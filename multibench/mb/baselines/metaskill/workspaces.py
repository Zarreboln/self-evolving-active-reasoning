"""What each Builder session may read: its workspace, built here and nowhere else.

Build workspace (paper eq. (2)/(3): H_x = B(H0, x, S)):
    TASK.md                 the released Builder prompt (BUILDER_SYSTEM + first-build instruction,
                            verbatim) + how the bundle is written in this interface
    card.json               public_benchmark_card: what the environment shows before the first action
    primitive_tools.json    primitive_tools: act + the three workbench tools
    skills.json             learned_support_skills: the full current bank (paper: full-bank mode)
    harness/components.py   H0, to be edited;  runner.py  the fixed runner (read only)
    build_harness.py, interface_check.py, check_harness.py, harness_lib.py, mock_game.py   checks
Reflect workspace (S_{j+1} = Revise_B(S_j, {(H_x, F(e_x))})):
    TASK.md                 the released reflection prompt (verbatim) + the batch adaptation
    skills.json             current_skills;  harness/components.py, harness/runner.py  generated_bundle
    batch.json              the episode IDs of this batch;  scores.json  per-episode results
    episodes/<id>/trace.md, result.json    public_episode: what the Target saw and did, and the result
Never: names of tasks in file or directory names, instance keys, selection or test splits, other
methods' runs, our private logs.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from mb.baselines.metaskill import assemble, official
from mb.baselines.metaskill.config import (EVOLVE, HERE, INTERFACE_VERSION, SKILL_FIELD_BYTES, SKILL_TOKENS,
                                           TARGET_MODEL)

PRIMITIVE_TOOLS = [
    {"type": "function", "function": {"name": "act", "description": "Take one action in the environment; "
     "returns the next observation. The environment ends the episode (win, loss or step limit).",
     "parameters": {"type": "object", "properties": {"action": {"type": "string"}}, "required": ["action"],
                    "additionalProperties": False}}},
    {"type": "function", "function": {"name": "workbench_write", "description": "Write a text file in task-local "
     "scratch space, not the official deliverable workspace.", "parameters": {"type": "object", "properties": {
         "path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"],
         "additionalProperties": False}}},
    {"type": "function", "function": {"name": "workbench_read", "description": "Read a task-local scratch file.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"],
                    "additionalProperties": False}}},
    {"type": "function", "function": {"name": "workbench_list", "description": "List task-local scratch files.",
     "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
]

BUILD_ADAPTATION = """
## How the bundle is written here (this section takes precedence over the format details above)

You are the Builder. The Target is {target}. It sees the instruction and the fixed runner boundary as
its system message, the task as its first user message, and one observation per action. It cannot see
this workspace.

- **The bundle is a Python file, not JSON.** Edit `harness/components.py`; it holds the same seven
  components: `INSTRUCTION` (instruction), `MEMORY`, `CONTEXT`, `TOOLS`, `CONTROLLER`, `VERIFICATION`,
  `WORKSPACE`. A disabled component is `None` (instead of JSON null) and `TOOLS = []`; an enabled one is a
  dict literal that names top-level functions of the file (instead of {{"source": ...}}). The callback
  contracts are at the top of `harness/components.py`; `runner.py` is the fixed runner that calls them
  (read it, do not edit it). It is appended to your file to make the harness.
- **The Target decides every action.** Every action sent to the environment must come from a model reply in that step. Code may build the prompt, check or repair the format of the model's answer, track state, or choose among candidates the model proposed; it must not decide the action without the model. A harness that plays an episode without calling the model is rejected.
- **One callback differs.** There is no finish tool: the environment ends the episode, and any action
  may end it. So `verification.before_finish(answer, snapshot)` is `before_action(action, snapshot)`
  here and gates every action (at most 2 rejections per action, then the action is sent).
- **Primitive tools** (`primitive_tools.json`): `act` takes one environment action; the workbench tools
  are scratch files. The Target calls tools with a `Tool: <name> <JSON>` line and acts with an
  `Action: <action>` line.
- **Code** is plain Python under the static rules in `check_harness.py` (instead of the bounded policy
  language): only the listed imports, no files, network, eval or exec, no names starting with `_` or
  `rt_`. Callbacks make no model calls.
- **Payload.** `public_benchmark_card` is `card.json` (what the environment shows before the first
  action{scope}); `primitive_tools` is `primitive_tools.json`; `current_bundle` is null: start from the
  neutral harness in `harness/components.py`; `learned_support_skills` is `skills.json` ({n_skills}
  skills).
- **Check** with `python build_harness.py` (interface check, static rules, a mock episode). It must print
  `ok`. Do not try to play the real environment; you have no access to it. Write only
  `harness/components.py`.
"""

REFLECT_ADAPTATION = """
## How the payload is given here (this section takes precedence where it differs)

- `current_skills` is `skills.json`. `generated_bundle` is `harness/components.py` (your harness for this
  task; `harness/runner.py` is the fixed runner it runs in).
- `public_episode` is a **batch**: every episode of one task played with this harness, one per problem.
  Per episode, `episodes/<id>/trace.md` has every message the Target saw, its replies, its actions and
  the observations, and `result.json` its result (`score` in [0, 1] is the environment's own score).
  `scores.json` has the batch's mean score and every episode's result.
- "This episode ID" means the IDs of this batch, listed in `batch.json`. A new or revised skill cites in
  `evidence_ids` at least one ID of this batch, and only IDs of this batch or IDs already cited in
  `skills.json`. A revision keeps every evidence ID of the skill it revises.
- **Write the one decision** as a JSON object to `decision.json` (instead of returning it):
  `{{"operation": "keep"}}`, or `{{"operation": "add", "skill": {{...}}}}`, or
  `{{"operation": "revise", "prior_skill_id": "...", "skill": {{...}}}}`. The skill has exactly the fields
  `skill_id` (letters, digits and `._:-`, starting with a letter or digit), `interface_version`
  (`"{version}"`), `when`, `provide`, `use`, `evidence_ids`. You may add a top-level `"rationale"` string.
- Size: `when` + `provide` + `use` together at most {tokens} tokens (about {tokens} words and
  punctuation marks), each at most {field_bytes} bytes.
- **The bank is shared.** It is read when harnesses are built for other tasks, including tasks the bank
  has never seen. So a skill must not name any game, benchmark, task or dataset, and must not contain any
  answer, secret, target word or other fact about a specific problem, including answers the environment
  revealed in a trace. Describe the observable condition, the support and its use in general terms.
- Do not play the environment. Write only `decision.json`.
"""


def _copy_tools(ws: Path):
    shutil.copy(EVOLVE / "check.py", ws / "check_harness.py")
    shutil.copy(EVOLVE / "ws_template" / "harness_lib.py", ws / "harness_lib.py")
    shutil.copy(HERE / "interface_check.py", ws / "interface_check.py")
    shutil.copy(HERE / "mock_game.py", ws / "mock_game.py")


def fill_build(ws: Path, card: dict, skills: list[dict], scope: str, repair: str | None = None,
               components: str | None = None):
    """scope: 'task' (one harness serves every problem of the task; the card shows its first problem)
    or 'problem'. repair: the interface error of the previous attempt (paper B.2)."""
    if ws.exists() and repair is None:
        shutil.rmtree(ws)
    (ws / "harness").mkdir(parents=True, exist_ok=True)
    _copy_tools(ws)
    shutil.copy(HERE / "ws_build_harness.py", ws / "build_harness.py")
    (ws / "runner.py").write_text(assemble.runner_source())
    if components is not None or not (ws / "harness" / "components.py").exists():
        (ws / "harness" / "components.py").write_text(components or assemble.neutral_components())
    (ws / "card.json").write_text(json.dumps({"public_benchmark_card": {
        "initial_task_prompt": card["instructions"], "initial_observation": card.get("observation", ""),
        "max_actions": card["max_steps"], "game_parses_whole_reply": card["reads_full_reply"],
        "valid_actions_at_start": card.get("valid_actions"),
        "success": "the environment's own score in [0, 1] when the episode ends",
        "target": TARGET_MODEL}}, indent=1))
    (ws / "primitive_tools.json").write_text(json.dumps(PRIMITIVE_TOOLS, indent=1))
    (ws / "skills.json").write_text(json.dumps(skills, indent=1))
    what = ("; the harness serves every problem of this task, and the card shows its first problem"
            if scope == "task" else "; the harness serves this one problem")
    task = ("# Build a harness for a Target agent\n\n" + official.builder_system() + "\n\n"
            + official.build_instruction() + "\n" + BUILD_ADAPTATION.format(
                target=TARGET_MODEL, scope=what, n_skills=len(skills)))
    if repair:
        task += ("\n## Repair\n\n" + official.repair_message(repair)
                 + "\n\n(Here: fix `harness/components.py`; a disabled component is `None`. Run "
                   "`python build_harness.py` until it prints `ok`.)\n")
    (ws / "TASK.md").write_text(task)


def fill_reflect(ws: Path, skills: list[dict], components: str, episodes: list[dict], ids: list[str],
                 repair: str | None = None):
    if ws.exists() and repair is None:
        shutil.rmtree(ws)
    (ws / "harness").mkdir(parents=True, exist_ok=True)
    (ws / "harness" / "components.py").write_text(components)
    (ws / "harness" / "runner.py").write_text(assemble.runner_source())
    (ws / "skills.json").write_text(json.dumps(skills, indent=1))
    (ws / "batch.json").write_text(json.dumps({"episode_ids": ids}, indent=1))
    per = {}
    for eid, e in zip(ids, episodes):
        d = ws / "episodes" / eid
        d.mkdir(parents=True, exist_ok=True)
        res = {k: e.get(k) for k in ("score", "success", "steps", "end", "error", "model_calls")}
        res["episode_id"] = eid
        (d / "result.json").write_text(json.dumps(res, indent=1))
        (d / "trace.md").write_text(e.get("trace", ""))
        per[eid] = res
    sc = [r["score"] for r in per.values() if isinstance(r["score"], (int, float))]
    (ws / "scores.json").write_text(json.dumps({"episodes": len(per), "mean_score": sum(sc) / len(sc) if sc else None,
                                               "per_episode": per}, indent=1))
    task = ("# Update the support-skill bank\n\n" + official.reflection_system() + "\n" + REFLECT_ADAPTATION.format(
        version=INTERFACE_VERSION, tokens=SKILL_TOKENS, field_bytes=SKILL_FIELD_BYTES))
    if repair:
        task += ("\n## Your previous decision was invalid\n\n" + repair + "\n\nWrite a corrected `decision.json` "
                 "(or `{\"operation\": \"keep\"}`).\n")
        if (ws / "decision.json").exists():
            (ws / "decision.json").rename(ws / "previous_decision.json")
    (ws / "TASK.md").write_text(task)
