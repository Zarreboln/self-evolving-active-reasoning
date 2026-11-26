"""The official Harness² code we use unchanged: its prompts and its mission wrapper.

Everything the editor, composer, selector and reflector read is built by the official functions in
third_party_b/harness2/harness2/core/prompts.py (propose, aggregate, compose_round, select,
experience) and core/context.py (mission, ContextTree.layout). What differs is only data, through the
official seams: a `BenchPrompts` value for our tasks, and one more entry in the per-substrate surface
table (`_SURFACE_TEXTS`) describing our component tree (surface.py) in place of OpenCode's.

The surface block is assembled from slices of the official OpenCode block wherever the text applies
unchanged (the cost ordering of forms, the edit contract, KEEP_BASE); the table of components, the
easy-to-get-wrong notes and the rules that discard a proposal are rewritten for our runtime, because
the OpenCode ones (TypeScript plugins, Jinja braces, spreadsheet deliverables) do not exist here.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]            # ~/Desktop/ar_harness
OFFICIAL = ROOT / "third_party_b" / "harness2"
OFFICIAL_COMMIT = "6868817518876de8fd170c6a8354daca273b6191"
if str(OFFICIAL) not in sys.path:
    sys.path.insert(0, str(OFFICIAL))

from harness2.core import context as official_context  # noqa: E402
from harness2.core import prompts as P  # noqa: E402
from harness2.core.spec import BenchPrompts  # noqa: E402

SURFACE_NAME = "mbpy"

_HEAD = """\
# WHAT YOU CAN CHANGE

The harness is a tree of components around a fixed agent loop. The loop is in
`context/runtime/harness_runtime.py`: read it once to see exactly when each component reaches the
model. You cannot change the loop; you change the components. Each has a different runtime cost and
a different reach:

| path | what it is | when it is in context |
|---|---|---|
| `systemprompt.md` | role and procedure | ALWAYS ON -- every token competes with the agent's own output budget |
| `guardrails.md` | standing rules | ALWAYS ON -- same cost |
| `LongTermMEMORY.md` | durable domain facts | ALWAYS ON -- same cost |
| `tool_descriptions/tool_guidance.md` | how to use the tools | ALWAYS ON -- same cost |
| `skills/<name>/SKILL.md` | a reusable workflow | loaded ON DEMAND, when the model replies `TOOL: skill <name>` |
| `sub_agents/<name>/AGENT.md` | a delegated subagent | runs in its OWN context (`TOOL: subagent <name> <input>`), returns text |
| `scripts/<name>.py` | a deterministic helper | run on demand (`TOOL: script <name> <input>`), or called by a plugin |
| `plugin/<name>.py` | a plugin | ALWAYS REGISTERED, and acts INSIDE the loop on every turn |

The agent's model is a small open model. On every turn the loop sends it the always-on text (plus a
list of the on-demand tools, when there are any) as the system message and the environment's latest
observation as the user message; its reply is sent to the environment as the action. A reply whose
last line is `TOOL: ...` is answered by the tool instead, and the model replies again (at most three
tool uses per turn).

The three that are easy to get wrong:

`skills/` -- frontmatter needs `name:` matching the directory and a `description:` (the description
is what the model sees in the tool list). Two further rules below decide whether it survives
validation at all: it must be named in the always-on text, and any script it calls must exist.

`sub_agents/` -- frontmatter needs `description:`; the body is the subagent's system prompt. It sees
only its own file and the `<input>` text, never the episode, and it returns text. So it is worth it
for exactly one shape of problem: a bounded sub-task whose INPUT is large and whose ANSWER is small.
Each use costs one extra model call.

`plugin/` -- a Python file defining `on_observation(obs, ctx)` and/or `on_action(action, ctx)`.
`on_observation` may return a replacement for the observation the model is shown; `on_action` may
return a replacement for the action sent to the environment; returning None changes nothing. `ctx`
is a dict: `instructions`, `turn`, `history` (list of (action, observation)), `admissible` (the
valid actions when the environment lists them, else None), `max_steps`, `state` (a dict the plugin
may keep its own notes in), `scripts` (name -> function) and `subagent(name, text)`. It is the only
component that acts in the loop rather than in the prompt, and therefore the only way to enforce
something the model cannot skip. Prose can be stopped reading; a hook cannot."""

_RULES = """\
THREE RULES THAT DISCARD THE WHOLE PROPOSAL. Validation runs before any rollout, and a harness
that fails it is thrown away entire.

1. EVERY SKILL NEEDS A TRIGGER LINE. If you create or keep `skills/<name>/SKILL.md`, its directory
   name must appear VERBATIM in `systemprompt.md` or `guardrails.md`, in the SAME edit:

       When <the situation this skill handles>, load the `<name>` skill.

2. PYTHON COMPONENTS RUN IN A SEALED PROCESS. `scripts/<name>.py` must define `main(text)` returning
   a string; `plugin/<name>.py` must define at least one of the two hooks. Both may import only
   `re`, `json`, `math`, `itertools`, `collections`, `random`, `string`, `functools`, `heapq`,
   `typing`, `dataclasses`; builtins that open files, import or execute code (`open`, `eval`,
   `exec`, `getattr`, `type`, ...) and any name or attribute starting with `_` are rejected. They
   cannot read files, the network or the environment's internals.

3. EVERY SCRIPT YOU NAME MUST EXIST. A `scripts/<name>.py` or `TOOL: script <name>` mentioned in any
   component must ship in the same proposal.

NO NAMES OF SPECIFIC GAMES OR TASK COLLECTIONS, AND NO ANSWERS. The harness you produce is stored
in a library that every later task reads, including tasks of kinds you have not seen. Describe a
situation by its shape (what is hidden, what each action reveals, how many tries remain), never by
the name of a game, a benchmark or a task collection, and never write down the answer or solution
of a specific instance. A proposal that does either is rejected.

THE ACTION FORMAT BELONGS TO THE ENVIRONMENT, NOT TO YOU. A harness changes HOW the agent decides;
it must never change the FORMAT the environment asks for. Where the instructions name a syntax
(brackets, a command vocabulary, a choice letter), every action sent must follow it exactly."""


def _surface_md() -> str:
    oc = P._SURFACE_OPENCODE
    middle = P._surface_part(oc, "Change only what a trajectory shows", "THREE RULES THAT DISCARD")
    middle = middle.replace(
        "trimming noisy or oversized tool output so it stops crowding context",
        "trimming noisy or oversized observations so they stop crowding context")
    how = P._surface_part(oc, "# HOW TO MAKE A CHANGE")
    return "\n\n".join([_HEAD, middle, _RULES, how])


# Harness²'s GENERALIZE block tells the editor to write verified facts of the current workspace into
# the harness, because there the harness is re-run on the same task. Here the facts of an instance
# are its answer (the secret word, the hidden code), which the shared-store rule forbids, and at test
# time each rollout of a step plays a different problem. The first paragraph is therefore replaced;
# the rest of the block is the official text (from "`context/library/` HOLDS ...").
_GENERALIZE_HEAD = """\
# SHAPE IT FOR THE {DOMAIN_WORD_UPPER}, NOT FOR THIS INSTANCE

The harness you produce is stored in this {domain_word}'s library, where every later task reads it
as an example -- later tasks are other instances, and often other kinds of task. Facts that hold
only for this instance (its hidden answer, the particular layout or values it happened to have)
are wrong for every other one and must not be written into the harness. What travels is
PROCEDURE: how to read this kind of feedback, how to choose an informative action, how to keep
track of what has been ruled out, when to commit. Write it for the {domain_word}: procedure that
would be wrong for a different task of the same shape is scoped too narrowly.

  instance fact    "the hidden answer here is <what this instance revealed>"
                                                           -- never; it is this instance's answer
  procedure        "after each feedback, list the letters ruled in and out, and only guess words
                    consistent with all of them"
  too generic      "be thorough"                           -- changes no behaviour

"""


def _generalize() -> str:
    tail = P.GENERALIZE[P.GENERALIZE.index("`context/library/` HOLDS"):]
    tail = tail.replace("RE-VERIFY its workspace facts against the evidence in front of you: an earlier\n"
                        "task's mapping or threshold is a hypothesis here, not a fact.",
                        "RE-VERIFY it against the evidence in front of you: an earlier task's procedure\n"
                        "is a hypothesis here, not a fact.")
    return _GENERALIZE_HEAD + tail


def register_surface() -> None:
    if SURFACE_NAME in P._SURFACE_TEXTS:
        return
    P.GENERALIZE = _generalize()
    P._SURFACE_TEXTS[SURFACE_NAME] = P._SurfaceText(
        surface_md=_surface_md(),
        inv=P._INV_TEMPLATE.replace(
            "__LOOP_LEVER__",
            "`plugin/<name>.py` hook -- for example, repairing or refusing an action that is not in "
            "the format the environment asks for"),
        dom=P._DOM_TEMPLATE.replace(
            "__TRIGGER_SENTENCE__",
            "A skill exists only once its name appears in the always-on text: write that trigger line "
            "in the same edit."),
        tts_surface="")


# The benchmark-shaped slots of the official prompts, for our tasks. BenchPrompts itself rejects the
# words benchmark/evaluation/grader/rubric/score; game and benchmark names are kept out by us.
PROMPTS = BenchPrompts(
    work=(
        "Interactive text tasks with hidden information. The agent receives instructions, then acts turn "
        "by turn and gets one text observation back per action: guesses and their feedback, questions and "
        "their answers, or moves and commands in a described world. It must gather what it needs through "
        "its own actions and reach the goal within a limited number of turns. The environment's replies "
        "are the only feedback; nobody else checks the work."
    ),
    domain_word="area",
    deliverable_word="sequence of actions",
    inv_levers=(
        "- keep what earlier observations established in front of the model, so no action repeats one "
        "whose result is already known\n"
        "- parse feedback that has a fixed format mechanically, with a script, instead of re-reading it "
        "in prose every turn\n"
        "- check every reply is exactly one action in the syntax the instructions require before it is "
        "sent, with one bounded repair when it is not\n"
        "- budget the turns: know how many remain and leave room to commit to an answer before they run out"
    ),
    dom_levers=(
        "- encode how a skilled player of this kind of task chooses the most informative next action, "
        "narrows the remaining possibilities, and decides when to stop gathering and commit\n"
        "- state rules the agent can apply and check against the observations, not background reading\n"
        "- describe situations by their shape (what is hidden, what an action reveals), never by the name "
        "of a particular game, and never record the solution of a particular instance"
    ),
    yardstick=(
        "Did the run follow the required action syntax on every turn, use every piece of feedback the "
        "environment returned, avoid actions whose outcome it already knew, and commit to an answer "
        "before the turns ran out?"
    ),
    selection_checks=(
        "Prefer the run whose environment replies show the goal reached; then the one that broke no "
        "format rule, used the feedback it received, and wasted fewer turns. Judge only from what the "
        "environment's replies in the transcript say."
    ),
)

register_surface()

propose = P.propose
aggregate = P.aggregate
compose_round = P.compose_round
select = P.select
experience = P.experience
mission = official_context.mission
ContextTree = official_context.ContextTree
GRADER_VERDICT_MARKERS = official_context.GRADER_VERDICT_MARKERS
