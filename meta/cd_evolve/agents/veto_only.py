"""Iteration 1, exploration of WHEN the model sees the state: only when it errs.

enum15 appends its summary after every answer: count, determined outputs,
joint tables. In the so1 traces that text is what doubles the input tokens
(63k vs 32k per episode, one context overflow), while the model mostly copies
the determined values into its own notes and keeps scanning entries in
canonical order. Here the model sees nothing beyond the environment's lines
on turns where its query is informative; the state reaches it only at the
moment it is about to waste a query.

filter_action: when the reply asks an output on which every remaining
configuration agrees (a repeat is one such case), or contains no parsable
query, it is not passed to the environment. The model is told the output is
already determined and to what value, is shown the outputs not determined yet
in canonical order, and replies again (at most twice per turn; no query is
spent). Nothing is ranked or recommended: the list is the complement of the
determined set. Once a single configuration remains every reply is accepted.
Rejected exchanges are removed from the context once the turn is settled, so
the transcript stays as short as bare's.

The final hint presents the configurations still consistent with the answers
(when at most 15 remain), written in the required <guess> order; with one
left that is the answer, which the model cannot assemble on its own (bare:
0% exact).
"""

from harness_lib import determined

NAME = "veto_only"
PARENT = "bare"
LIST_CAP = 15
LABELS = "ABCDEFGH"


def init_state(view):
    return {"vetoes": [], "n_veto": 0, "n_format": 0}


def build_feedback(view, state):
    return ""


def filter_action(view, state, text, q):
    """Send the reply back if it asks a determined output or contains no query."""
    if len(view.alive) <= 1:
        return None
    fixed = dict(determined(view))
    asked = dict(view.history)
    if q is None:
        state["n_format"] += 1
        msg = ("Your reply contains no query in the required format, and the circuits are not "
               f"pinned down yet: {len(view.alive)} configurations are still possible. ")
    elif q in asked:
        state["n_veto"] += 1
        msg = (f"You already asked {view.fmt(q)} and the answer was {asked[q]}; "
               "asking it again would waste one of your queries. ")
    elif q in fixed:
        state["n_veto"] += 1
        msg = (f"{view.fmt(q)} is already determined: every circuit configuration consistent "
               f"with the inventory and your answers so far gives {view.fmt(q)} = {fixed[q]}, "
               "so asking it would waste one of your queries. ")
    else:
        return None
    still = [view.fmt(x) for x in range(view.c << view.k) if x not in fixed]
    msg += ("Outputs not determined yet: " + ", ".join(still) + ". Reply with one query for an "
            "output that is not determined yet, in the format <query>X(a, b, c)</query>.")
    state["vetoes"].append(msg)
    return msg


def compose_messages(view, state, msgs):
    """Drop settled rejections: each re-ask message and the reply it rejected."""
    vetoes = set(state["vetoes"])
    if not vetoes:
        return msgs
    last = len(msgs) - 1
    drop = set()
    for i in range(1, last):
        if msgs[i]["role"] == "user" and msgs[i]["content"] in vetoes:
            drop.add(i)
            if msgs[i - 1]["role"] == "assistant":
                drop.add(i - 1)
    return [m for i, m in enumerate(msgs) if i not in drop]


def final_hint(view, state):
    n = len(view.alive)
    if n > LIST_CAP:
        return ""
    tables = sorted("".join(str(int(b)) for b in row) for row in view.H[view.alive].tolist())
    order = (", then ".join(LABELS[:view.c]) + f"; inputs {'0' * view.k} to {'1' * view.k} "
             "within each circuit")
    if n == 1:
        return ("Exactly one configuration of the circuits is consistent with all your answers. "
                f"Its outputs in the required order ({order}): <guess>{tables[0]}</guess>")
    return (f"{n} configurations of the circuits are still consistent with all your answers. "
            f"Their outputs in the required order ({order}): "
            + ", ".join(f"<guess>{t}</guess>" for t in tables) + ".")
