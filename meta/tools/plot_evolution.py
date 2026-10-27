"""The two views of a harness search: accuracy against iteration, and accuracy against cost.

Laid out after VideoHarness-RSI's Figures 2 and 3, with one change forced by the search itself.
That paper keeps a single parent and replaces it only on a strict improvement, so one step line is
the whole story. Meta-Harness keeps a Pareto frontier over (accuracy, cost), so a candidate can be
kept while scoring below the best: the step line here is the running best accuracy on the left and
the frontier on the right, and a point can be a diamond on one panel and a star on the other.

Labels are placed by a greedy pass in display coordinates rather than by hand, because the three
tasks differ too much -- Guess Number stacks three candidates on one accuracy, Detective Cases
spreads seven over a decade of cost -- and the data changes under us while the runs are going.

    python meta/tools/plot_evolution.py scratch/ds_candidates.json figures/
"""
from __future__ import annotations

import json
import math
import sys
import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, NullFormatter

PARENT, REJECT, FRONTIER, SEED, RULE, MUTE = (
    "#1f4e79", "#6b7585", "#2f7d8c", "#9aa3b0", "#c8cdd5", "#5c636f")

TASKS = [("gn", "Guess Number", "exact match"),
         ("dc", "Detective Cases", "right suspect of five"),
         ("sp", "Situation Puzzles", "judged equivalent by jev-1.14.0")]
CHANCE = {"dc": 20.0}


def style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8.5, "axes.labelsize": 9, "axes.titlesize": 9.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#3c424c", "axes.linewidth": 0.8,
        "xtick.color": "#3c424c", "ytick.color": "#3c424c",
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.frameon": False, "figure.dpi": 200, "savefig.dpi": 200,
    })


def mark(ax, x, y, kind, z=3):
    """One candidate. A star is on the frontier, a diamond was dominated, a circle is a seed."""
    if kind == "seed":
        ax.plot(x, y, "o", ms=5.2, mfc=SEED, mec=SEED, zorder=z)
    elif kind == "frontier":
        ax.plot(x, y, "*", ms=13, mfc=FRONTIER, mec="white", mew=0.8, zorder=z + 1)
    else:
        ax.plot(x, y, "D", ms=4.6, mfc=REJECT, mec=REJECT, zorder=z)


def place(ax, items, pad_pt=10.0):
    """Greedy placement with a widening spiral, all of it in device pixels.

    Two things make this worth doing properly. The box is measured from the drawn text rather than
    assumed, because the names run from `draft_verify` to `witness_first_stop` and a fixed
    half-width either overlaps or wastes room. And the offset is applied in pixels, not points:
    at 200 dpi a point is 2.78 pixels, so geometry worked out in pixels and handed to
    `offset points` puts every label almost three times further out than it was placed, which is
    how labels end up outside the axes while the collision test says they fit.
    """
    fig = ax.figure
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    px_per_pt = fig.dpi / 72.0
    pad = pad_pt * px_per_pt
    placed = []

    angles = [90, 270, 0, 180, 45, 135, 315, 225, 60, 120, 300, 240]
    offsets = [(0.0, pad), (0.0, -pad - 5 * px_per_pt)]
    for radius in (1.0, 1.7, 2.5, 3.4, 4.5, 5.8):
        for a in angles:
            offsets.append((math.cos(math.radians(a)) * pad * radius * 1.9,
                            math.sin(math.radians(a)) * pad * radius))

    # the markers are obstacles too, so a label never lands on another system's point
    half = 7.5 * px_per_pt                 # the largest marker (a 13pt star) in pixels
    for x, y, _t, _c in items:
        mx, my = ax.transData.transform((x, y))
        placed.append((mx - half, my - half, mx + half, my + half))

    area = ax.get_window_extent(renderer=rend)
    inside = (area.x0 + 1, area.y0 + 1, area.x1 - 1, area.y1 - 1)

    for x, y, text, colour in items:
        mx, my = ax.transData.transform((x, y))
        # measured in axes fraction: (0, 0) in data coordinates is invalid on a log axis and the
        # extent that comes back there is degenerate
        probe = ax.annotate(text, (0.5, 0.5), xycoords="axes fraction", fontsize=6.6, alpha=0)
        bb = probe.get_window_extent(renderer=rend)
        probe.remove()
        hw, hh = bb.width / 2 + 3.0 * px_per_pt, bb.height / 2 + 2.0 * px_per_pt
        chosen, fallback = None, None
        for dx, dy in offsets:
            box = (mx + dx - hw, my + dy - hh, mx + dx + hw, my + dy + hh)
            if box[0] < inside[0] or box[2] > inside[2] \
                    or box[1] < inside[1] or box[3] > inside[3]:
                continue
            if any(box[0] < t[2] and t[0] < box[2] and box[1] < t[3] and t[1] < box[3]
                   for t in placed):
                if fallback is None:
                    fallback = (dx, dy)
                continue
            chosen = (dx, dy)
            break
        if chosen is None:                 # nowhere clear inside: the least bad that still fits
            chosen = fallback if fallback is not None else (0.0, pad)
        dx, dy = chosen
        placed.append((mx + dx - hw, my + dy - hh, mx + dx + hw, my + dy + hh))
        far = math.hypot(dx, dy) > pad * 1.4
        kw = {"arrowprops": dict(arrowstyle="-", lw=0.55, color=RULE, shrinkA=1, shrinkB=5)} \
            if far else {}
        ax.annotate(text, (x, y), textcoords="offset pixels", xytext=(dx, dy),
                    ha="center", va="center", fontsize=6.6, color=colour,
                    linespacing=1.15, zorder=6, **kw)


def by_task(rows):
    out = {}
    for r in rows:
        if r["accuracy"] is None or r["tokens"] is None:
            continue
        out.setdefault(r["task_id"], []).append(r)
    return out


def panel_iteration(ax, rows, title, metric):
    """Accuracy against the candidate's place in the search. One slot per candidate rather than per
    iteration: three candidates share an iteration here, and stacking them on one x hides both the
    markers and the order the proposer wrote them in. The step line is the best reached so far."""
    seq = sorted(rows, key=lambda r: (r["iteration"], -(r["accuracy"] or 0), r["system"]))
    xs = list(range(len(seq)))
    run, best = -1.0, []
    for r in seq:
        run = max(run, r["accuracy"])
        best.append(run)
    ax.step(xs + [xs[-1] + 0.5], best + best[-1:], where="post",
            color=PARENT, lw=1.6, zorder=2, solid_capstyle="round")
    labels = []
    for x, r in zip(xs, seq):
        kind = "seed" if r["kind"] == "seed" else ("frontier" if r["frontier"] == "yes" else "x")
        mark(ax, x, r["accuracy"], kind)
        colour = SEED if kind == "seed" else (FRONTIER if kind == "frontier" else MUTE)
        labels.append((x, r["accuracy"], f"{r['system']}\n{r['accuracy']:.1f}", colour))

    # iteration bands: a rule between groups, the number centred under each
    ax.set_xticks([])
    groups = {}
    for x, r in zip(xs, seq):
        groups.setdefault(r["iteration"], []).append(x)
    for it, gx in sorted(groups.items()):
        if min(gx) > 0:
            ax.axvline(min(gx) - 0.5, color=RULE, lw=0.7, zorder=1)
        ax.annotate("seeds" if it == 0 else f"iteration {it}",
                    ((min(gx) + max(gx)) / 2, 0), xycoords=("data", "axes fraction"),
                    xytext=(0, -15), textcoords="offset points",
                    ha="center", va="top", fontsize=7.2, color=MUTE)
    ax.set_xlabel("Candidates, in the order the proposer wrote them", labelpad=30)
    ax.set_title(title, loc="left", pad=9, color="#171a20")
    ax.text(0, 1.015, metric, transform=ax.transAxes, fontsize=7, color=MUTE, va="bottom")
    ax.set_xlim(-0.6, xs[-1] + 0.6)
    return labels


def panel_cost(ax, rows, task_id):
    """Accuracy against what one episode costs; the step line is the Pareto frontier."""
    fr = sorted([r for r in rows if r["frontier"] == "yes"], key=lambda r: r["tokens"])
    if fr:
        xs, ys = [], []
        for r in fr:
            xs += [r["tokens"], r["tokens"]]
            ys += [ys[-1] if ys else r["accuracy"], r["accuracy"]]
        ax.plot(xs[1:], ys[1:], color=PARENT, lw=1.4, ls=(0, (5, 2.5)), zorder=2)
    labels = []
    for r in rows:
        kind = "seed" if r["kind"] == "seed" else ("frontier" if r["frontier"] == "yes" else "x")
        mark(ax, r["tokens"], r["accuracy"], kind)
        colour = SEED if kind == "seed" else (FRONTIER if kind == "frontier" else MUTE)
        labels.append((r["tokens"], r["accuracy"],
                       f"{r['system']}\n{r['tokens']:,.0f}", colour))
    ax.set_xscale("log")
    ax.set_xlabel("Total tokens per episode (input + output)")
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda v, _: f"{v/1000:.0f}k" if v >= 1000 else f"{v:.0f}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    lo = min(r["tokens"] for r in rows)
    hi = max(r["tokens"] for r in rows)
    ax.set_xlim(lo / 3.4, hi * 3.4)
    return labels


def main(data_path, out_dir):
    rows = json.load(open(data_path))
    groups = by_task(rows)
    style()
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    out = Path(out_dir)

    for which, builder, fname, caption in [
        ("iteration", panel_iteration, "evolution_trajectory",
         "Every candidate the proposer wrote, by the iteration that wrote it. The step line is the "
         "best selection accuracy reached so far; a star is on the accuracy-cost frontier, a "
         "diamond was dominated, a circle is a seed the search started from."),
        ("cost", panel_cost, "evolution_cost",
         "The same candidates against what one episode costs to run. The dashed line is the Pareto "
         "frontier; cost is input plus output tokens, so a harness that buys accuracy by "
         "generating more text pays for it here."),
    ]:
        fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.8))
        pending = []
        for ax, (tid, name, metric) in zip(axes, TASKS):
            rs = groups.get(tid, [])
            if not rs:
                ax.set_visible(False)
                continue
            if which == "iteration":
                labels = builder(ax, rs, name, metric)
            else:
                labels = builder(ax, rs, tid)
                ax.set_title(name, loc="left", pad=9, color="#171a20")
                ax.text(0, 1.015, metric, transform=ax.transAxes, fontsize=7,
                        color=MUTE, va="bottom")
            lo = min(r["accuracy"] for r in rs)
            hi = max(r["accuracy"] for r in rs)
            span = max(hi - lo, 8)
            floor = lo - span * 0.16
            if lo >= 0:                      # no accuracy is negative; don't draw room for one
                floor = max(floor, -span * 0.04)
            ax.set_ylim(floor, hi + span * 0.34)
            if which == "cost" and tid in CHANCE and CHANCE[tid] > ax.get_ylim()[0]:
                ax.axhline(CHANCE[tid], color=RULE, lw=0.9, ls=(0, (2, 2)), zorder=1)
                ax.annotate("chance", (0.995, CHANCE[tid]), xycoords=("axes fraction", "data"),
                            xytext=(0, 3), textcoords="offset points",
                            ha="right", va="bottom", fontsize=6.4, color=MUTE)
            ax.grid(axis="y", color=RULE, lw=0.5, alpha=0.5)
            ax.set_axisbelow(True)
            pending.append((ax, labels))
        axes[0].set_ylabel("Selection accuracy (%)")
        handles = [
            Line2D([], [], marker="*", ls="", ms=12, mfc=FRONTIER, mec="white",
                   label="On the frontier"),
            Line2D([], [], marker="D", ls="", ms=4.6, color=REJECT, label="Dominated"),
            Line2D([], [], marker="o", ls="", ms=5.2, color=SEED, label="Seed"),
            Line2D([], [], color=PARENT, lw=1.6,
                   label="Best so far" if which == "iteration" else "Pareto frontier"),
        ]
        fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.005),
                   handletextpad=0.5, columnspacing=1.8, fontsize=8)
        fig.text(0.008, 0.012, f"AR-Bench, frozen Qwen2.5-7B-Instruct, 100 selection episodes "
                               f"per point  ·  run1  ·  {stamp}", fontsize=6.6, color=MUTE)
        fig.tight_layout(rect=(0, 0.055, 1, 0.945))
        for ax, labels in pending:        # after the layout, so the axes box is the final one
            place(ax, labels)
        for ext in ("png", "pdf"):
            fig.savefig(out / f"{fname}.{ext}", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  {fname}.png / .pdf  — {caption[:60]}…")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "figures")
