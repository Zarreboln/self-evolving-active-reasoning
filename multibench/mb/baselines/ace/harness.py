"""The playing agent: a seed harness with ACE's playbook inserted into its prompt.

ACE's generator is the task's own agent with the playbook in its context. In ACE's agent setting
(ace-appworld, experiments/prompts/appworld_react_generator_prompt.txt) the ReAct agent's first
(instruction) message carries the playbook between `### PLAYBOOK BEGIN` / `### PLAYBOOK END`, introduced
by a fixed sentence; every later call keeps that message. We do the same to each seed: every model
call the seed makes gets the block put in front of its first message (the system message when there
is one, else the first user message). Nothing else of the seed changes: its prompts, history, sampling
and how it turns replies into actions are untouched.

On a retry of the same problem before deployment (ACE's Reflector refinement rounds), ACE's
generator also gets the reflection on the previous attempt (ace/prompts/generator.py: "**Reflection:**");
it is appended to the block then, and is "(empty)" otherwise, as in ACE.

The result is one self-contained harness file that passes mb/evolve/check.py.
"""
from __future__ import annotations

import re
from pathlib import Path

SEEDS = Path(__file__).resolve().parents[2] / "evolve" / "seeds"

# Default starting agent per benchmark: the benchmark's official default agent, the same seeds the
# per-benchmark methods start from (docs/official_baselines.md, notes/exp_b_design.md).
OFFICIAL_SEED = {"wordle": "textarena_standard", "hangman": "textarena_standard",
                 "mastermind": "textarena_standard", "minesweeper": "textarena_standard",
                 "twentyq": "textarena_standard", "babyai": "balrog_naive",
                 "sciworld": "swiftsage_react", "dc": "arbench_zero_shot",
                 "alfworld": "react_alfworld"}          # not in seeds/ yet: add the official ReAct agent, or pass --seed

# ace-appworld appworld_react_generator_prompt.txt, lines 28-33, with "API-specific information, "
# dropped (there are no APIs here).
INTRO = ("You are also provided with a curated playbook of strategies, common mistakes, and proven solutions "
         "to help you solve the task effectively.\n\n"
         "**Playbook**: - Read the **Playbook** first, then execute the task by explicitly leveraging each "
         "relevant section:\n### PLAYBOOK BEGIN\n")
OUTRO = "\n### PLAYBOOK END\n"
REFLECTION_HEAD = ("\n**Reflection** (a reflection that goes over the diagnosis of all previous mistakes made "
                   "while answering the question):\n")


def block(playbook: str, reflection: str | None = None) -> str:
    b = INTRO + playbook + OUTRO
    if reflection and reflection != "(empty)":
        b += REFLECTION_HEAD + reflection + "\n"
    return b + "\n"


WRAPPER = '''

# ---- ACE: the playbook goes in front of the first message of every model call ----
ACE_BLOCK = {block!r}


def ace_insert(messages):
    out = [dict(m) for m in messages]
    if out:
        out[0]["content"] = ACE_BLOCK + str(out[0].get("content") or "")
    return out


def run_episode(env, chat):
    def chat_with_playbook(messages, *args, **kwargs):
        return chat(ace_insert(messages), *args, **kwargs)
    return seed_run_episode(env, chat_with_playbook)
'''


def seed_for(cell: str, override: str | None = None) -> str:
    return override or OFFICIAL_SEED[cell.split(":", 1)[0]]


def build(seed_name: str, playbook: str, reflection: str | None = None) -> str:
    path = SEEDS / f"{seed_name}.py"
    if not path.exists():
        raise FileNotFoundError(f"seed {path} does not exist; pass --seed or add the official agent there")
    src = path.read_text()
    renamed, n = re.subn(r"^def run_episode\(", "def seed_run_episode(", src, flags=re.M)
    if n != 1 or re.search(r"\brun_episode\(", renamed.replace("seed_run_episode(", "")):
        raise ValueError(f"{seed_name}: expected exactly one run_episode and no other reference to it")
    return renamed + WRAPPER.format(block=block(playbook, reflection))


def strip_block(text: str, playbook: str, reflection: str | None = None) -> str:
    """What the method's Reflector reads of a trace: the block it inserted itself is cut out."""
    return text.replace(block(playbook, reflection), "[ACE playbook block omitted]\n")
