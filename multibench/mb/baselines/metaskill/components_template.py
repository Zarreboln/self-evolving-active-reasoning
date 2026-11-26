# Harness components: the neutral environment H0 (every optional component disabled).
#
# Seven component families (paper Table 1; released interface "modular-v2.0"). The fixed runner
# (runner.py, appended below this file to make harness.py) calls them; you edit only this file.
# A disabled component is None. An enabled one is a dict literal that maps each callback name to a
# top-level function defined in this file. Callbacks are deterministic code: they receive and return
# JSON values, make no model calls, and keep no state outside what the runner passes back to them.
#
#   INSTRUCTION   str, always present: put first in the Target's system message.
#   MEMORY        {"write": f(event, records) -> {"upsert": [record, ...], "delete": [id, ...]},
#                  "retrieve": g(query, records) -> [id, ...]}
#                 records are JSON objects with a string "id"; at most 256 records.
#   CONTEXT       {"select_context": f(groups, retrieved) -> [group id, ...]}
#                 groups: [{"id", "roles", "characters", "preview"}] of the exchanges after the task
#                 message; return ids in increasing order and always keep the newest one.
#   TOOLS         list of {"name", "description", "parameters" (JSON schema), "next_action": f}
#                 f(arguments, state, previous) -> {"state": {...}, "call": {"tool": primitive,
#                 "arguments": {...}}} or {"state": {...}, "return": value}; "previous" is the result
#                 of the last primitive call. Primitives: act {"action": str} (one environment step),
#                 workbench_write {"path", "content"}, workbench_read {"path"}, workbench_list {}.
#   CONTROLLER    {"on_event": f(event, snapshot) -> {"state"?, "phase"?, "visible_tools"?, "instruction"?}}
#                 called on every event; visible_tools None means all; act is always available.
#   VERIFICATION  {"before_action": f(action, snapshot) -> {"allow": bool, "feedback": str, "actions"?: [...]},
#                  optional "on_failure": g(failure, snapshot) -> {"actions": [...], "feedback": str}}
#                 before_action gates every environment action (any action may end the episode);
#                 at most 2 rejections per action, then the action is sent.
#   WORKSPACE     {"setup": f(task) -> {"files": [{"path", "content"}], "actions": [...]}}
#                 called once at the start; task = {"prompt", "initial_observation", "public_metadata"}.
#
# Events: round_start, before_model, primitive_result {tool, arguments, origin, result}, action_check_result,
# model_output_error. Snapshot: {"controller", "phase", "memory", "recent_events", "remaining"}.
# "actions" lists hold {"tool": primitive other than act, "arguments": {...}}.
# Allowed imports: re, json, math, itertools, collections, random, string, functools, heapq, typing,
# dataclasses, harness_lib. No names starting with "_" or "rt_"/"Rt"/"RT_"; no open/eval/exec/getattr/type.

INSTRUCTION = "@@NEUTRAL_INSTRUCTION@@"
MEMORY = None
CONTEXT = None
TOOLS = []
CONTROLLER = None
VERIFICATION = None
WORKSPACE = None
