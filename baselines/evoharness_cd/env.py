"""CircuitDecoding (Multi-Turn Puzzles, arXiv 2508.10142) as an EvoHarness ``Env``.

Protocol follows the paper's transcript: one query per turn, the environment
answers ``A(0, 0, 1) = 1`` and counts the budget down; after the 18th answer the
policy must submit the 24-bit joint truth table. A reply with no parsable action
burns one of the 18 queries, exactly as in ar_harness/runner/cd_harness_run.py.

Data and the hypothesis-space code are taken from the ar_harness checkout given
by $AR_HARNESS_ROOT (default: the parent of this repository).
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ...domain import Step, Task

ROOT = Path(os.environ.get("AR_HARNESS_ROOT", Path(__file__).resolve().parents[4]))
if str(ROOT / "stage0") not in sys.path:
    sys.path.insert(0, str(ROOT / "stage0"))
from cd_oracle import CDGame  # noqa: E402  (public hypothesis space, no truth)

LABELS = "ABCDEFGH"
BUDGET = 18
C, K = 3, 3
QUERY_RE = re.compile(r"query\s*([a-h])\s*\(\s*([01])\s*,\s*([01])\s*,\s*([01])\s*\)", re.I)
GUESS_RE = re.compile(r"guess\s*([01][01\s,]*)", re.I)
KICKOFF = "Please start with your first query."
ILLEGAL = ("Your message did not contain a valid action. Reply with exactly one action: "
           "query X(a, b, c), or the final guess when asked.")

_GAMES: dict[tuple, CDGame] = {}


def game_for(components: dict) -> CDGame:
    key = tuple(sorted(components.items()))
    if key not in _GAMES:
        _GAMES[key] = CDGame(dict(key))
    return _GAMES[key]


def fmt(q: int) -> str:
    lb = LABELS[q // (1 << K)]
    row = q % (1 << K)
    return f"{lb}({', '.join(str((row >> (K - 1 - i)) & 1) for i in range(K))})"


def inventory_text(components: dict) -> str:
    return ", ".join(f"{v} {g} gate{'s' if v != 1 else ''}" for g, v in sorted(components.items()) if v)


def load_instances(path: Path) -> list[dict]:
    out = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        c = json.loads(line)["controller"]
        out.append({"components": c["components"],
                    "truth": np.array([int(ch) for ch in c["circuit_joint_lookup"]], dtype=np.uint8)})
    return out


def query_commands() -> list[str]:
    return [f"query {fmt(q)}" for q in range(C << K)]


class CircuitDecodingEnv:
    """One instance at a time; ``reset`` picks the instance by ``task.data`` (index)."""

    def __init__(self, data_path: str | None = None, **_: Any) -> None:
        self.instances = load_instances(Path(data_path) if data_path else ROOT / "mt_puzzles/cd_test.jsonl")
        self.inst = None

    def reset(self, task: Task) -> Step:
        self.inst = self.instances[int(task.data)]
        self.truth = self.inst["truth"]
        self.game = game_for(self.inst["components"])
        self.alive = np.arange(len(self.game.H))
        self.history: list[tuple[int, int]] = []
        self.used = 0          # queries consumed (legal or burned)
        self.finished = False
        self.objective = (f"Deduce circuits A, B, C. Inventory: {inventory_text(self.inst['components'])}, "
                          f"distributed across the 3 circuits. {BUDGET} queries, then the final guess.")
        return Step(observation=KICKOFF, actions=query_commands(), objective=self.objective,
                    info={"task_type": self._task_type()})

    def _task_type(self) -> str:
        c = self.inst["components"]
        return f"and{c.get('AND', 0)}_or{c.get('OR', 0)}_not{c.get('NOT', 0)}"

    def _actions(self) -> list[str]:
        return query_commands() if self.used < BUDGET else ["guess <24 bits in the required order>"]

    def step(self, action: str) -> Step:
        text = (action or "").strip()
        if self.finished:
            return Step(observation="The episode is over.", actions=[], objective=self.objective,
                        done=True, won=False, valid=False)
        if self.used < BUDGET:
            m = QUERY_RE.search(text)
            if not m:
                self.used += 1            # a wasted turn, as in the reference runner
                return self._after(ILLEGAL, valid=False)
            q = LABELS.index(m.group(1).upper()) * (1 << K) + int(m.group(2) + m.group(3) + m.group(4), 2)
            bit = int(self.truth[q])
            self.history.append((q, bit))
            self.alive = self.alive[self.game.H[self.alive, q] == bit] if len(self.alive) else self.alive
            self.used += 1
            return self._after(f"{fmt(q)} = {bit}", valid=True)
        # final guess
        self.finished = True
        m = GUESS_RE.search(text)
        bits = re.sub(r"[^01]", "", m.group(1)) if m else ""
        if len(bits) != C << K:
            cand = re.findall(r"[01]{%d}" % (C << K), re.sub(r"\s", "", text))
            bits = cand[-1] if cand else ""
        guess = np.array([int(ch) for ch in bits], dtype=np.uint8) if len(bits) == C << K else None
        won = guess is not None and bool((guess == self.truth).all())
        ckt = 0.0
        if guess is not None:
            ckt = sum(int((guess[i * 8:(i + 1) * 8] == self.truth[i * 8:(i + 1) * 8]).all()) for i in range(C)) / C
        info = {"task_type": self._task_type(), "n_alive_end": int(len(self.alive)),
                "unique_at_end": int(len(self.alive)) == 1, "parsed_guess": guess is not None,
                "n_queries": len(self.history), "queries": [list(h) for h in self.history]}
        return Step(observation=f"Final guess received. Correct: {won}.", actions=[],
                    objective=self.objective, done=True, won=won, score=ckt, valid=guess is not None, info=info)

    def _after(self, line: str, valid: bool) -> Step:
        remaining = BUDGET - self.used
        if remaining > 0:
            obs = f"{line}\nPlease send your next query. {remaining} queries remaining."
        else:
            obs = (f"{line}\nPlease provide the truth tables for all circuits in the format requested. "
                   "Make sure to enclose your guess in the <action>guess ...</action> markers.")
        return Step(observation=obs, actions=self._actions(), objective=self.objective, valid=valid,
                    info={"task_type": self._task_type(), "n_alive": int(len(self.alive))})

    def close(self) -> None:
        pass


def list_tasks(split: str, split_file: str | None = None, data_path: str | None = None) -> list[Task]:
    n = len(load_instances(Path(data_path) if data_path else ROOT / "mt_puzzles/cd_test.jsonl"))
    if split == "all":
        keys = list(range(n))
    else:
        sf = Path(split_file) if split_file else ROOT / "splits/cd_split_v2.json"
        keys = json.loads(sf.read_text())[split]
    return [Task(id=f"cd{k:03d}", data=k) for k in keys]
