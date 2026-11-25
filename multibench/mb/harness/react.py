"""The plainest harness: show the instructions, then loop observation -> one model call -> action.

It is the seed every bench starts from, not a method. The model sees the last WINDOW exchanges
(BALROG's default history length) and, when the bench provides them and the list is short, the
admissible actions. Where the bench's own parser searches the whole reply (env.reads_full_reply),
the reply is passed through unchanged, as the original protocols do; otherwise the action is the
text after a final `Action:` line if there is one, else the last non-empty line, with trailing
punctuation dropped. max_tokens is 8192: at 512, and on Minesweeper at 4096, this model's replies were cut off mid-reasoning.
"""
from __future__ import annotations

import re

WINDOW = 16
MAX_ADMISSIBLE = 60
SYSTEM = ("You are playing a text game. Read the instructions, then on every turn reply with "
          "exactly one action in the format the game asks for. You may think briefly first, but "
          "end your reply with a line of the form `Action: <action>`.")


def _with_admissible(text, env):
    adm = env.admissible()
    if adm and len(adm) <= MAX_ADMISSIBLE:
        text += "\n\nAdmissible actions: " + ", ".join(adm)
    return text


def extract_action(reply):
    m = re.findall(r"^\s*Action\s*:\s*(.+?)\s*$", reply or "", re.M | re.I)
    if m:
        a = m[-1]
    else:
        lines = [l.strip() for l in (reply or "").splitlines() if l.strip()]
        a = lines[-1] if lines else ""
    return a.strip().strip("`").rstrip(".!").strip()


def run_episode(env, chat, temperature=0.7, max_tokens=8192):
    first = {"role": "user", "content": _with_admissible(env.instructions, env)}
    turns, log = [], []
    while not env.done:
        msgs = [{"role": "system", "content": SYSTEM}, first] + turns[-2 * WINDOW:]
        try:
            reply = chat(msgs, temperature=temperature, max_tokens=max_tokens)
        except Exception as e:      # e.g. the prompt outgrew the server's context: end the episode, keep the record
            return {**env.result(), "steps_log": log, "error": repr(e)}
        action = reply if env.reads_full_reply else extract_action(reply)
        obs = env.act(action)
        log.append({"reply": reply, "action": action, "observation": obs})
        turns += [{"role": "assistant", "content": reply},
                  {"role": "user", "content": _with_admissible(obs, env)}]
    return {**env.result(), "steps_log": log}
