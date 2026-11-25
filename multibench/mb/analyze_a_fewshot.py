"""Experiment A, amendment 1 (notes/exp_a_prereg.md): leave-one-task-out few-shot prediction.

Does a structure fitted on the other tasks let a new task be characterised from a few episodes?
For each held-out task and each k, the estimators below predict the task's 4 component gains from
k paired instances; the target is the gain on the task's remaining instances.

    python -m mb.analyze_a_fewshot --out ~/results/mb/exp_a
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from mb.analyze_a import CELLS, GROUP, delta, load

KS = (1, 2, 3, 5, 8, 12)


def predict(D9, groups9, g_t, dk, est, r=1):
    """D9: [4, 9] gains on the other tasks; dk: [4] gains on t's k instances (or None)."""
    a = D9.mean(axis=1)
    if est == "G":
        return a
    if est == "H":
        same = groups9 == g_t
        return D9[:, same].mean(axis=1) if same.any() else a
    if est == "S":
        return dk
    U = np.linalg.svd(D9 - a[:, None], full_matrices=False)[0][:, :r]
    if est == "L":
        w = np.linalg.lstsq(U, dk - a, rcond=None)[0]
        return a + U @ w
    if est == "Lprobe":                       # only belief (row 0) measured
        w = (dk[0] - a[0]) / U[0, 0]
        return a + U[:, 0] * w
    raise ValueError(est)


def run(data, cells, draws, rng):
    names = [f"{b}/{s}" for b, s in cells]
    groups = np.array([GROUP[b] for b, _ in cells])
    full = np.stack([delta(data[c]) for c in cells], axis=1)
    rows = {}
    ests = [("G", 1, 0), ("H", 1, 0), ("S", 1, 5), ("L", 1, 5), ("L", 2, 5), ("Lprobe", 1, 2)]
    for k in KS:
        err = {e: [] for e in ests}
        for t, c in enumerate(cells):
            x = data[c]
            n = x.shape[1]
            if n - k < 3:
                continue
            others = [j for j in range(len(cells)) if j != t]
            D9 = full[:, others]
            for _ in range(draws):
                p = rng.permutation(n)
                dk, target = delta(x, p[:k]), delta(x, p[k:])
                for e in ests:
                    pred = predict(D9, groups[others], groups[t], dk, e[0], e[1])
                    err[e].append(((pred - target) ** 2).mean())
        rows[k] = {e: float(np.mean(v)) for e, v in err.items()}
    print(f"{'k':>3s} " + " ".join(f"{e[0] + (str(e[1]) if e[0] == 'L' else ''):>8s}({e[2]}k)" for e in ests))
    for k, r in rows.items():
        print(f"{k:3d} " + " ".join(f"{r[e]:12.4f}" for e in ests))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--draws", type=int, default=200)
    a = ap.parse_args()
    data, _ = load(a.out)
    print("held-out MSE of the predicted 4-vector of gains; (Nk) = episodes spent on the new task\n")
    print("== all 10 tasks")
    run(data, CELLS, a.draws, np.random.default_rng(0))
    print("\n== without gn")
    run(data, [c for c in CELLS if c[0] != "gn"], a.draws, np.random.default_rng(0))


if __name__ == "__main__":
    main()
