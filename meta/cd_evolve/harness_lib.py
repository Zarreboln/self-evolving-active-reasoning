"""Text helpers shared by CD harness candidates.

Everything here is computed from public information only: the hypothesis
space H (derived from the inventory the model is told), the feasible-set
indices `alive` (derived from the model's own query/answer history) and the
history itself. The runner never passes the truth table into a harness, so
a candidate cannot leak it even by accident -- see runner/cd_harness_run.py.
"""

import numpy as np

LABELS = "ABCDEFGH"


def fmt(q, k):
    """Joint truth-table index -> 'A(0,1,1)'."""
    lb = LABELS[q // (1 << k)]
    row = q % (1 << k)
    bits = ",".join(str((row >> (k - 1 - i)) & 1) for i in range(k))
    return f"{lb}({bits})"


def determined(view):
    """Entries forced to a single value by every remaining hypothesis:
    [(q, value), ...] in canonical order."""
    sub = view.H[view.alive]
    out = [(int(q), 1) for q in np.flatnonzero((sub == 1).all(axis=0))]
    out += [(int(q), 0) for q in np.flatnonzero((sub == 0).all(axis=0))]
    return sorted(out)


def count_line(view):
    n = len(view.alive)
    return (f"{n} circuit configuration{'s' if n != 1 else ''} remain"
            f"{'s' if n == 1 else ''} consistent with every answer given so far.")


def determined_text(view):
    det = determined(view)
    if not det:
        return ""
    return ("Already determined outputs (do not ask these again): "
            + ", ".join(f"{fmt(q, view.k)}={v}" for q, v in det) + ".")


def enumeration_text(view, cap):
    """List every remaining joint truth table once there are at most `cap`."""
    n = len(view.alive)
    if not (1 < n <= cap):
        return ""
    opts = sorted("".join(map(str, row)) for row in view.H[view.alive].tolist())
    return ("All circuit configurations still consistent with every answer "
            f"(as {view.c * (1 << view.k)}-bit joint truth tables): "
            + ", ".join(opts) + ".")


def observations_text(view):
    """The model's own observations restated compactly, no inference."""
    if not view.history:
        return ""
    return ("Your observations so far: "
            + ", ".join(f"{fmt(q, view.k)}={b}" for q, b in view.history) + ".")


def split_table_text(view):
    """For every still-undetermined entry, how the remaining configurations
    split between output 0 and output 1. Canonical order, not sorted by
    informativeness: the model still has to compare the numbers."""
    sub = view.H[view.alive]
    n = len(view.alive)
    if n <= 1:
        return ""
    ones = sub.sum(axis=0)
    cells = [(int(q), int(n - ones[q]), int(ones[q]))
             for q in range(sub.shape[1]) if 0 < ones[q] < n]
    if not cells:
        return ""
    return ("Undetermined entries and how the remaining configurations split "
            "between output 0 and output 1 (count0/count1): "
            + ", ".join(f"{fmt(q, view.k)}: {z}/{o}" for q, z, o in cells) + ".")


def best_query(view):
    """Max-entropy query over the feasible set, or None if nothing is left."""
    bits, q = view.game.available_bits(view.alive)
    return None if q is None or bits <= 0 else int(q)
