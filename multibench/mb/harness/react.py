"""The plainest harness: show the instructions, then loop observation -> one model call -> action.

It is the seed every bench starts from, not a method. The model sees the last WINDOW exchanges
(BALROG's default history length) and, when the bench provides them and the list is short, the
admissible actions. The action is the text after a final `Action:` line if there is one,
otherwise the last non-empty line of the reply.
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
        return m[-1].strip().strip("`")
    lines = [l.strip() for l in (reply or "").splitlines() if l.strip()]
    return lines[-1].strip("`") if lines else ""


def run_episode(env, chat, temperature=0.7, max_tokens=512):
    first = {"role": "user", "content": _with_admissible(env.instructions, env)}
    turns = []
    while not env.done:
        msgs = [{"role": "system", "content": SYSTEM}, first] + turns[-2 * WINDOW:]
        reply = chat(msgs, temperature=temperature, max_tokens=max_tokens)
        obs = env.act(extract_action(reply))
        turns += [{"role": "assistant", "content": reply},
                  {"role": "user", "content": _with_admissible(obs, env)}]
    return env.result()
