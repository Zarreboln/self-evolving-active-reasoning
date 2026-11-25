"""Experiment A's hand-written harness components: the react seed plus one component at a time.

A is a premise check, not a method. It asks whether the gain of a harness component depends on
the task in a way that is neither uniform (one global harness would do) nor split by bench (a
two-level harness would do). Each component is a small, fixed change to the seed:

  seed    the react seed, unchanged
  belief  before each action the model writes <belief>: the hypotheses still consistent with
          every observation. The latest one is stripped from what the env sees and shown again at
          the top of every later observation, so it survives the seed's 16-exchange window.
  plan    on the first turn the model writes a <plan>, and may rewrite it later; the latest plan
          is shown again every turn, like the belief.
  check   before any action that ends the episode (a final guess, answer, accusation, or the last
          step of a task) the model writes a <check> testing it against all observations, and
          gathers more information instead if the check fails. Not carried forward.
  brief   reasoning is capped at about 50 words per turn. No block.

Blocks are removed from the reply before the env sees it, because several benches parse the
whole reply (env.reads_full_reply) and would read e.g. a bracketed candidate list as a move.
Everything else -- the window, the admissible-action list, the action extraction, temperature,
max_tokens -- is the seed's (mb/harness/react.py).
"""
from __future__ import annotations

import re

from mb.harness.react import SYSTEM, WINDOW, _with_admissible, extract_action

_NO_BRACKETS = (" Use square brackets only in your action, never inside a <{tag}> block.")

STRATEGIES = {
    "seed": {"system": "", "carry": None},
    "belief": {
        "system": ("\n\nBefore each action, write a <belief>...</belief> block listing the hypotheses "
                   "about the hidden state that are still consistent with every observation so far, "
                   "most likely first, and what you are still unsure about. Your latest belief is "
                   "shown back to you each turn; update it rather than starting over."
                   + _NO_BRACKETS.format(tag="belief")),
        "carry": "belief"},
    "plan": {
        "system": ("\n\nOn your first turn, write a <plan>...</plan> block: the steps you will take to "
                   "reach the goal, and how you will know when you are done. Your latest plan is shown "
                   "back to you each turn; follow it, and write a new <plan> block only when an "
                   "observation shows it must change." + _NO_BRACKETS.format(tag="plan")),
        "carry": "plan"},
    "check": {
        "system": ("\n\nBefore any action that ends the game -- a final guess, answer or accusation, or "
                   "the last step of the task -- first write a <check>...</check> block that tests "
                   "that action against every observation so far. If any observation contradicts it, "
                   "or another answer is equally consistent, take an information-gathering action "
                   "instead." + _NO_BRACKETS.format(tag="check")),
        "carry": None},
    "brief": {
        "system": ("\n\nKeep your reasoning to at most 50 words per turn, then give the action."),
        "carry": None},
}
_BLOCK = re.compile(r"<(belief|plan|check)>(.*?)</\1>", re.S | re.I)


def split_blocks(reply):
    """(reply with the strategy blocks removed, {tag: last block's text})."""
    found = {}
    for m in _BLOCK.finditer(reply or ""):
        found[m.group(1).lower()] = m.group(2).strip()
    return _BLOCK.sub("", reply or "").strip(), found


def run_episode(env, chat, strategy="seed", temperature=0.7, max_tokens=8192):
    s = STRATEGIES[strategy]
    system = SYSTEM + s["system"]
    first = {"role": "user", "content": _with_admissible(env.instructions, env)}
    turns, log, carried = [], [], None
    while not env.done:
        msgs = [{"role": "system", "content": system}, first] + turns[-2 * WINDOW:]
        try:
            reply = chat(msgs, temperature=temperature, max_tokens=max_tokens)
        except Exception as e:      # e.g. the prompt outgrew the server's context
            return {**env.result(), "strategy": strategy, "steps_log": log, "error": repr(e)}
        stripped, blocks = split_blocks(reply)
        if s["carry"] and s["carry"] in blocks:
            carried = blocks[s["carry"]]
        action = stripped if env.reads_full_reply else extract_action(stripped)
        obs = env.act(action)
        log.append({"reply": reply, "action": action, "observation": obs, "blocks": blocks})
        shown = _with_admissible(obs, env)
        if carried:
            shown = f"Your latest <{s['carry']}>:\n{carried}\n\n{shown}"
        turns += [{"role": "assistant", "content": reply}, {"role": "user", "content": shown}]
    return {**env.result(), "strategy": strategy, "steps_log": log}
