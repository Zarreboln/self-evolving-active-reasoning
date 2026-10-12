"""Iteration 1, exploitation of oracle_hint: a copyable directive with a safe fallback.

oracle_hint appends the max-entropy query after enum15's negative list and
joint tables. Here the state part is replaced by the outputs that are still
unknown (in canonical order, query format) and the recommendation is the
exact <query> tag to send, followed by an instruction to reply with it alone.
The point of keeping a list next to the directive: on turns where the model
does not copy, it falls back on its scanning habit, and if that scan runs
over the unknown outputs it is still informative. A CPU simulation on the
search split puts uniq@end at 98-100% for copy rates >= 0.7 when misses are
informative, against 72% at 0.7 when misses are wasted turns. Compare with
copy_query (directive alone) and open_entries (list alone).

After the collapse the message says so and asks for a fixed harmless query;
the final hint hands over the remaining configuration as a ready <guess>
string."""

from harness_lib import best_query, determined

NAME = "copy_open"
PARENT = "oracle_hint"
LABELS = "ABCDEFGH"


def _qtag(view, q):
    """Query index -> '<query>B(1, 0, 0)</query>', the reply format of the system prompt."""
    row = q % (1 << view.k)
    bits = ", ".join(str((row >> (view.k - 1 - i)) & 1) for i in range(view.k))
    return f"<query>{LABELS[q // (1 << view.k)]}({bits})</query>"


def init_state(view):
    return {}


def build_feedback(view, state):
    n = len(view.alive)
    if n <= 1:
        return ("Your answers now determine every truth table; no query can add information. "
                f"Reply with only {_qtag(view, 0)}")
    fixed = {q for q, _ in determined(view)}
    unknown = [view.fmt(q) for q in range(view.c << view.k) if q not in fixed]
    q = best_query(view)
    ones = int(view.H[view.alive, q].sum())
    return (f"{n} circuit configurations are still possible. Outputs still unknown: "
            + ", ".join(unknown) + ". Every other output is already fixed by your answers. "
            f"Most informative next query: {_qtag(view, q)} ({n - ones} of the {n} give 0, "
            f"{ones} give 1). Reply with exactly this query and nothing else.")


def final_hint(view, state):
    n = len(view.alive)
    bits = "".join(str(int(b)) for b in view.H[view.alive[0]])
    lead = ("Your answers leave exactly one possible set of truth tables." if n == 1 else
            f"Your answers leave {n} possible sets of truth tables; here is one of them.")
    order = ", then ".join(LABELS[:view.c])
    rows = f"inputs {'0' * view.k} to {'1' * view.k} within each"
    return (f"{lead} In the required order ({order}; {rows}) it is: "
            f"<guess>{bits}</guess>\nSubmit exactly this string.")
