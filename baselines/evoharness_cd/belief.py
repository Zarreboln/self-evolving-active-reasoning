"""CircuitDecoding belief: the exact feasible set, rebuilt from the agent's own
observations. Never touches the truth table: the inventory comes from the
objective text and every answer from the observation line ``X(a, b, c) = b``."""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from ...belief import Belief
from .env import C, K, LABELS, fmt, game_for

OBS_RE = re.compile(r"\b([A-H])\(\s*([01])\s*,\s*([01])\s*,\s*([01])\s*\)\s*=\s*([01])")
INV_RE = re.compile(r"(\d+)\s+(AND|OR|NOT)\s+gate", re.I)
LIST_CAP = 15


class CircuitDecodingBelief(Belief):
    def __init__(self) -> None:
        self.game = None
        self.reset("", "")

    def reset(self, objective: str, observation: str) -> None:
        comps = {g.upper(): int(n) for n, g in INV_RE.findall(objective or "")}
        self.game = game_for(comps) if comps else None
        self.alive = np.arange(len(self.game.H)) if self.game is not None else np.arange(0)
        self.history: list[tuple[int, int]] = []

    # ------------------------------------------------------------ update --
    def update(self, action: str, observation: str, actions: list[str], won: bool) -> dict[str, Any]:
        if self.game is None:
            return None
        for lb, a, b, c, bit in OBS_RE.findall(observation or ""):
            q = LABELS.index(lb) * (1 << K) + int(a + b + c, 2)
            if (q, int(bit)) not in self.history:
                self.history.append((q, int(bit)))
                self.alive = self.alive[self.game.H[self.alive, q] == int(bit)]
        n = int(len(self.alive))
        return {"status": "complete" if won or n == 1 else "partial",
                "satisfied": ["circuits pinned down"] if n == 1 else [],
                "missing": [] if n == 1 else [f"{n} configurations still possible"],
                "complete": bool(won)}

    # ------------------------------------------------------------- views --
    def _determined(self) -> list[tuple[int, int]]:
        sub = self.game.H[self.alive]
        if len(sub) == 0:
            return []
        out = [(int(q), 1) for q in np.flatnonzero((sub == 1).all(axis=0))]
        out += [(int(q), 0) for q in np.flatnonzero((sub == 0).all(axis=0))]
        return sorted(out)

    def _tables(self) -> list[str]:
        return sorted("".join(str(int(b)) for b in row) for row in self.game.H[self.alive].tolist())

    def track(self, query: str) -> str:
        if self.game is None:
            return "TRACKED: no inventory known."
        q = (query or "").strip().lower()
        n = int(len(self.alive))
        det = dict(self._determined())
        if q and q[0] in "abc" and len(q) == 1:
            lb = q.upper(); base = LABELS.index(lb) * (1 << K)
            rows = []
            for r in range(1 << K):
                v = det.get(base + r)
                rows.append(f"{fmt(base + r)}={'?' if v is None else v}")
            return f"TRACKED circuit {lb}: " + ", ".join(rows)
        parts = [f"{n} configuration{'s' if n != 1 else ''} consistent with every answer so far."]
        if det:
            parts.append("Determined outputs (asking these again gains nothing): "
                         + ", ".join(f"{fmt(k)}={v}" for k, v in sorted(det.items())) + ".")
        open_cells = [fmt(x) for x in range(C << K) if x not in det]
        if open_cells and n > 1:
            parts.append("Outputs not determined yet: " + ", ".join(open_cells) + ".")
        if 1 < n <= LIST_CAP or (n == 1):
            parts.append("Remaining joint truth tables (A then B then C, rows 000..111): " + ", ".join(self._tables()) + ".")
        return "TRACKED: " + " ".join(parts)

    def render(self, focus: list[str]) -> str:
        if self.game is None:
            return ""
        return self.track("")[len("TRACKED: "):]

    def to_dict(self) -> dict[str, Any]:
        return {"n_alive": int(len(self.alive)), "history": [list(h) for h in self.history]}
