"""Experiment A analysis, exactly as pre-registered in notes/exp_a_prereg.md.

  1. Δ[s,t] = paired mean of score(s) − score(seed) over instances, 95% bootstrap CI (10k).
  2. Sign test: components with at least one significantly positive and one significantly
     negative cell.
  3. Split-half cross-validation (200 random halves of each cell's instances, shared by all
     strategies so the pairing holds): fit on one half, squared error on the other, for
     G (a_s), H (a_s + b_{s,group}), L1 / L2 (a_s + rank-1 / rank-2 residual), S (per cell).
  4. Pass = sign part (≥ 2 components) AND best of L1/L2 beats G, H and S.

Also printed, as the first use of notes/method_formulas.md (not part of the pass rule): the DPM
τ-softmax assignment γ(t,c) over {seed, components} at several τ, and the cosine similarity of
the tasks' gain vectors Δ(t,·).

    python -m mb.analyze_a --out ~/results/mb/exp_a
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os

import numpy as np

STRATS = ["seed", "belief", "plan", "check", "brief"]
COMPS = STRATS[1:]
CELLS = [("gn", "all"), ("dc", "all"), ("mastermind", "basic"), ("hangman", "basic"),
         ("twentyq", "basic-things"), ("wordle", "basic5"), ("blicket", "n5"),
         ("sciworld", "test-conductivity"), ("alfworld", "pick_and_place_simple"),
         ("babyai", "GoToLocal")]
GROUP = {"gn": 0, "dc": 0, "mastermind": 0, "hangman": 0, "twentyq": 0, "wordle": 0,
         "blicket": 1, "sciworld": 1, "alfworld": 2, "babyai": 2}
GROUP_NAMES = ["pure", "intermediate", "world"]


def load(out, keep_errors=False):
    """{cell: array [strategy, instance]} on the instances every strategy finished cleanly. With
    keep_errors (sensitivity check, not the pre-registered analysis), an errored episode counts
    with the score it had when it stopped, or 0 if it has none."""
    data, dropped = {}, {}
    for bench, sub in CELLS:
        per = []
        for s in STRATS:
            d = {}
            for f in glob.glob(os.path.join(out, s, bench, sub.replace("/", "_"), "*.json")):
                r = json.load(open(f))
                sc = r.get("score")
                ok = isinstance(sc, (int, float)) and not math.isnan(sc)
                if not r.get("error") and ok:
                    d[os.path.basename(f)] = float(sc)
                elif keep_errors:
                    d[os.path.basename(f)] = float(sc) if ok else 0.0
            per.append(d)
        common = sorted(set.intersection(*(set(d) for d in per)))
        dropped[f"{bench}/{sub}"] = max(len(d) for d in per) - len(common)
        data[(bench, sub)] = np.array([[d[i] for i in common] for d in per])
    return data, dropped


def delta(x, idx=None):
    """[4] mean paired gains of the components over seed on instances idx."""
    x = x if idx is None else x[:, idx]
    return (x[1:] - x[0]).mean(axis=1)


def fit_predict(D, model):
    """D: [4 components, 10 tasks] gains from one half; returns the model's fitted matrix."""
    a = D.mean(axis=1, keepdims=True)
    if model == "G":
        return np.repeat(a, D.shape[1], axis=1)
    if model == "H":
        g = np.array([GROUP[b] for b, _ in CELLS])
        out = np.empty_like(D)
        for k in range(3):
            out[:, g == k] = D[:, g == k].mean(axis=1, keepdims=True)
        return out
    if model == "S":
        return D.copy()
    r = int(model[1])
    U, sv, Vt = np.linalg.svd(D - a, full_matrices=False)
    return a + (U[:, :r] * sv[:r]) @ Vt[:r]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=10_000)
    ap.add_argument("--halves", type=int, default=200)
    ap.add_argument("--keep-errors", action="store_true", help="sensitivity: score errored episodes")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    data, dropped = load(a.out, a.keep_errors)
    names = [f"{b}/{s}" for b, s in CELLS]

    # 1. gains and bootstrap CIs
    D = np.zeros((4, len(CELLS)))
    lo, hi = np.zeros_like(D), np.zeros_like(D)
    print(f"{'cell':34s} {'n':>3s} {'seed':>6s} " + " ".join(f"{c:>22s}" for c in COMPS))
    for j, c in enumerate(CELLS):
        x = data[c]
        n = x.shape[1]
        D[:, j] = delta(x)
        bs = np.stack([delta(x, rng.integers(0, n, n)) for _ in range(a.boot)])
        lo[:, j], hi[:, j] = np.percentile(bs, 2.5, axis=0), np.percentile(bs, 97.5, axis=0)
        cells = []
        for i in range(4):
            star = "+" if lo[i, j] > 0 else "-" if hi[i, j] < 0 else " "
            cells.append(f"{D[i, j]:+.3f} [{lo[i, j]:+.2f},{hi[i, j]:+.2f}]{star}")
        print(f"{names[j]:34s} {n:3d} {x[0].mean():6.3f} " + " ".join(f"{s:>22s}" for s in cells))
    print("dropped (unpaired/errored):", {k: v for k, v in dropped.items() if v})

    # 2. sign test
    both = [COMPS[i] for i in range(4) if (lo[i] > 0).any() and (hi[i] < 0).any()]
    print(f"\nsign: components with significant cells of both signs: {both} ({len(both)}/4)")
    for i, comp in enumerate(COMPS):
        print(f"  {comp:6s} + {[names[j] for j in np.where(lo[i] > 0)[0]]}  - {[names[j] for j in np.where(hi[i] < 0)[0]]}")

    # 3. split-half CV
    models = ["G", "H", "L1", "L2", "S"]
    err = {m: [] for m in models}
    for _ in range(a.halves):
        DA, DB = np.zeros_like(D), np.zeros_like(D)
        for j, c in enumerate(CELLS):
            n = data[c].shape[1]
            p = rng.permutation(n)
            DA[:, j], DB[:, j] = delta(data[c], p[: n // 2]), delta(data[c], p[n // 2:])
        for m in models:
            err[m].append(((fit_predict(DA, m) - DB) ** 2).mean())
    print("\nsplit-half held-out MSE (mean ± sd over halves):")
    for m in models:
        print(f"  {m:3s} {np.mean(err[m]):.5f} ± {np.std(err[m]):.5f}")
    best_L = min(("L1", "L2"), key=lambda m: np.mean(err[m]))
    beats = {m: float(np.mean(np.array(err[best_L]) < np.array(err[m]))) for m in ("G", "H", "S")}
    structure = all(np.mean(err[best_L]) < np.mean(err[m]) for m in ("G", "H", "S"))
    print(f"  best low-rank = {best_L}; fraction of halves it beats G/H/S: {beats}")

    a_s = D.mean(axis=1, keepdims=True)
    U, sv, Vt = np.linalg.svd(D - a_s, full_matrices=False)
    print(f"  singular values of the interaction: {np.round(sv, 4).tolist()}")
    print("  rank-1 task loadings v_t:", {names[j]: round(float(Vt[0, j]), 3) for j in np.argsort(Vt[0])})
    print("  rank-1 component loadings u_s:", {COMPS[i]: round(float(U[i, 0]), 3) for i in range(4)})

    passed = len(both) >= 2 and structure
    print(f"\nPASS = {passed}  (sign {len(both) >= 2}, structure {structure})")

    # first use of the method formulas (not part of the pass rule)
    L = np.vstack([np.array([data[c][0].mean() for c in CELLS]), D + np.array([data[c][0].mean() for c in CELLS])])
    print("\nDPM τ-softmax γ(t, ·) over", STRATS)
    for tau in (0.01, 0.05, 0.1, 0.3, 1.0):
        print(f"  τ={tau}")
        for j in range(len(CELLS)):
            z = L[:, j] / tau
            g = np.exp(z - z.max()); g /= g.sum()
            print(f"    {names[j]:34s} " + " ".join(f"{v:.2f}" for v in g))
    V = D.T / (np.linalg.norm(D.T, axis=1, keepdims=True) + 1e-12)
    print("\ncosine similarity of task gain vectors Δ(t,·):")
    print(" " * 34 + " ".join(f"{n.split('/')[0][:6]:>6s}" for n in names))
    for j in range(len(CELLS)):
        print(f"  {names[j]:32s} " + " ".join(f"{v:+.2f}".rjust(6) for v in V[j] @ V.T))
    json.dump({"cells": names, "components": COMPS, "delta": D.tolist(), "lo": lo.tolist(),
               "hi": hi.tolist(), "cv": {m: [float(np.mean(err[m])), float(np.std(err[m]))] for m in models},
               "sign_both": both, "pass": passed},
              open(os.path.join(a.out, "analysis_a" + ("_keep_errors" if a.keep_errors else "") + ".json"), "w"), indent=1)


if __name__ == "__main__":
    main()
