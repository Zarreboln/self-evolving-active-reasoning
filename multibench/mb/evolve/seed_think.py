"""Candidate c000: the `think` arm of experiment A (mb/harness/strategies.py), written against the
candidate interface. The react seed plus a free <reasoning> block before each action; the block is
removed before the game sees the reply."""
import re

from harness_lib import SYSTEM, WINDOW, extract_action, with_admissible

THINK = ("\n\nBefore each action, reason step by step about the situation and what to do "
         "next inside a <reasoning>...</reasoning> block. Use square brackets only in your action, "
         "never inside a <reasoning> block.")
BLOCK = re.compile(r"<(belief|plan|check|reasoning)>(.*?)</\1>", re.S | re.I)


def run_episode(env, chat):
    system = SYSTEM + THINK
    first = {"role": "user", "content": with_admissible(env.instructions, env)}
    turns = []
    while not env.done:
        msgs = [{"role": "system", "content": system}, first] + turns[-2 * WINDOW:]
        reply = chat(msgs, temperature=0.7, max_tokens=8192)
        stripped = BLOCK.sub("", reply).strip()
        action = stripped if env.reads_full_reply else extract_action(stripped)
        obs = env.act(action)
        turns += [{"role": "assistant", "content": reply},
                  {"role": "user", "content": with_admissible(obs, env)}]
