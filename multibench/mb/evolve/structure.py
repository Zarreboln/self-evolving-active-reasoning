"""Soft structured task profiles and retrieval by soft structure (user, 2026-11-28).

Every task, and every node of the graph, is described on the same fixed dimensions, but a value is a degree, not a
choice: a task may be 0.7 "enumerable" and 0.3 "structured" in its hypothesis space. A new task is compared with the
graph dimension by dimension, as the overlap of two soft values, not by the wording of free-text descriptions.

Dimensions (three groups):
- problem: the five dimensions the user set on 2026-11-24 for organising these games (goal and action, hypothesis
  space, observation channel, information cost, stopping condition); they decide the skeleton (main loop);
- interface: action form, whether each observation already holds the history, episode length; they decide parsing,
  memory and context;
- needs: how the plain agent fails, and which parts of a harness the game needs (role labels); they decide slots.

Two kinds of dimension:
- "one of": the options share one unit of weight (a distribution), e.g. hypothesis_space;
- "each": every option has its own degree in [0, 1], e.g. which roles a game needs.

A node's soft structure is derived, not written: on each dimension, the mean of its member tasks' soft values.
"""
from __future__ import annotations

import json
import math

ROLES = ("instruction", "memory", "context", "tools", "controller", "verification", "workspace")

# name: (group, kind, options); kind "one" = a distribution over the options, "each" = a degree per option
DIMS = {
    "goal": ("problem", "one", ("answer", "state")),
    "action_kind": ("problem", "each", ("query", "intervene", "persistent_change")),
    "hypothesis_space": ("problem", "one", ("enumerable", "structured", "open")),
    "observation_channel": ("problem", "each", ("per_element_feedback", "natural_language_answer", "local_percept",
                                                "full_state")),
    "info_cost": ("problem", "one", ("free", "total_budget", "per_step_cost")),
    "stopping": ("problem", "each", ("budget_exhausted", "agent_submits", "goal_reached")),
    "action_form": ("interface", "one", ("fixed_list", "word_or_letter", "command_with_args", "free_text")),
    "obs_has_history": ("interface", "one", ("yes", "no")),
    "episode_length": ("interface", "one", ("short", "medium", "long")),
    "base_agent_failures": ("needs", "each", ("format_rejected", "repeats_no_effect", "forgets", "wrong_reasoning",
                                              "other")),
    "needed_roles": ("needs", "each", ROLES),
}

DESCRIBE_STRUCTURE = """Write `structure.json`: the game's structure as its instructions and traces show it, as degrees, not single
choices. For a dimension marked "one of", give each option a weight; the weights add up to 1 (a game between two
options splits the weight). For a dimension marked "each", give every option its own degree from 0 to 1. Use the
option names exactly as spelled; leave out options with no weight. Example:
{"hypothesis_space": {"enumerable": 0.7, "structured": 0.3}, "needed_roles": {"memory": 0.9, "verification": 0.6}}
Dimensions:
- "goal" (one of): the end is giving an answer ("answer") or reaching a state of the world ("state");
- "action_kind" (each): actions ask about the hidden thing ("query"), act on the world temporarily ("intervene"),
  change it for the rest of the episode ("persistent_change");
- "hypothesis_space" (one of): what is being found is one of a list one can write down ("enumerable"), has structure
  but no list ("structured"), is open-ended ("open");
- "observation_channel" (each): structured feedback per element of a guess ("per_element_feedback"), answers in
  natural language ("natural_language_answer"), what is around the agent ("local_percept"), the whole state
  ("full_state");
- "info_cost" (one of): information is free ("free"), limited by a total budget ("total_budget"), costs per step
  ("per_step_cost");
- "stopping" (each): the episode ends when the budget runs out ("budget_exhausted"), when the agent submits
  ("agent_submits"), when the goal is reached ("goal_reached");
- "action_form" (one of): one of a fixed list of commands ("fixed_list"), a word or letter ("word_or_letter"), a command
  with arguments ("command_with_args"), free text ("free_text");
- "obs_has_history" (one of): each observation already repeats what happened before ("yes") or not ("no");
- "episode_length" (one of): up to 20 turns ("short"), 20 to 100 ("medium"), more than 100 ("long");
- "base_agent_failures" (each): how often the plain agent's replies are not accepted as actions ("format_rejected"),
  it repeats actions that change nothing ("repeats_no_effect"), it forgets earlier information ("forgets"), it
  reasons wrongly ("wrong_reasoning"), other ("other");
- "needed_roles" (each): how much a good harness for this game needs each part: instruction, memory, context, tools,
  controller, verification, workspace.
"""


def clean(s):
    """A soft structure with only known dimensions and options: "one of" dimensions normalised to sum 1, "each"
    dimensions clipped to [0, 1]. A plain option name (or a list of names) is read as full weight on it."""
    out = {}
    if not isinstance(s, dict):
        return out
    for d, (_, kind, opts) in DIMS.items():
        v = s.get(d)
        if isinstance(v, str):
            v = {v: 1.0}
        elif isinstance(v, list):
            v = {x: 1.0 for x in v if isinstance(x, str)}
        if not isinstance(v, dict):
            continue
        w = {}
        for o, x in v.items():
            try:
                if o in opts and float(x) > 0:
                    w[o] = min(float(x), 1.0)
            except (TypeError, ValueError):
                pass
        if not w:
            continue
        if kind == "one":
            z = sum(w.values())
            w = {o: x / z for o, x in w.items()}
        out[d] = {o: round(x, 3) for o, x in w.items()}
    return out


def node_structure(nd, structures, weights=None):
    """A node's soft structure: per dimension, the mean of its member tasks' soft values (weighted by the member
    edge weights when given)."""
    acc, tot = {}, {}
    for x in nd.get("members", {}):
        s = structures.get(x) or {}
        wx = (weights or {}).get(x, 1.0)
        for d, v in s.items():
            tot[d] = tot.get(d, 0.0) + wx
            for o, q in v.items():
                acc.setdefault(d, {})[o] = acc.setdefault(d, {}).get(o, 0.0) + wx * q
    return {d: {o: q / tot[d] for o, q in v.items()} for d, v in acc.items()}


def dim_match(a, b, kind):
    """Overlap of two soft values on one dimension, in [0, 1]: for "one of" the shared mass sum(min(p, q)); for
    "each" one minus the mean absolute difference over the options either side has."""
    if not a or not b:
        return None
    if kind == "one":
        return sum(min(a.get(o, 0.0), b.get(o, 0.0)) for o in set(a) | set(b))
    opts = set(a) | set(b)
    return 1.0 - sum(abs(a.get(o, 0.0) - b.get(o, 0.0)) for o in opts) / len(opts)


def match(task_struct, node_struct, groups):
    """Mean overlap over the dimensions of the given groups that both sides have; and the per-dimension detail."""
    detail = {}
    for d, (grp, kind, _) in DIMS.items():
        if grp in groups:
            m = dim_match(task_struct.get(d), node_struct.get(d), kind)
            if m is not None:
                detail[d] = round(m, 2)
    return (sum(detail.values()) / len(detail) if detail else 0.0), detail


def soft(scores, tau):
    """Softmax of scores at temperature tau (tau small: only the best; large: all alike)."""
    if not scores:
        return {}
    m = max(scores.values())
    e = {k: math.exp((v - m) / tau) for k, v in scores.items()}
    z = sum(e.values())
    return {k: v / z for k, v in e.items()}


def parse_file(path):
    try:
        return clean(json.loads(path.read_text()))
    except Exception:
        return {}
