"""Where the search went, against where the published methods already were.

Two figures, both on the selection split with the same frozen policy, so AR-Bench's own reasoning
methods and the harnesses the proposer wrote sit on one frontier and the question "did the search
beat what was already there" is read off the picture rather than argued.

  evolution_frontier     accuracy against what one episode costs. The frontier is recomputed over
                         the union, so a published method keeps its place only if nothing the
                         search wrote dominates it.
  evolution_trajectory   the same candidates in the order the proposer wrote them, with the step
                         line showing the best reached so far.

Uncertainty-of-Thoughts is held out of the seed sets and is not drawn. Everything else AR-Bench
ships is here.

    python meta/tools/plot_evolution.py notes/arbench_mh/evolution_run1_snapshot.json figures/
"""
from __future__ import annotations

import json
import math
import sys
import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter

FRONT, DOM, RULE, MUTE, INK = "#b4530f", "#8b93a1", "#c9ced6", "#7b8290", "#1c1f26"

TASKS = [("gn", "Guess Number", "exact match"),
         ("dc", "Detective Cases", "right suspect of five"),
         ("sp", "Situation Puzzles", "partially correct, judged by jev-1.14.0")]
CHANCE = {"dc": 20}
PUBLISHED = Path("notes/arbench_mh/published_baselines.json")


def style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9.5, "axes.labelsize": 9.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#3c424c", "axes.linewidth": 0.9,
        "xtick.color": "#3c424c", "ytick.color": "#3c424c",
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "figure.dpi": 200, "savefig.dpi": 200,
    })


def tick(v, _):
    if v >= 1e6:
        return "%gM" % (v / 1e6)
    if v >= 1e3:
        return "%gk" % (v / 1e3)
    return "%g" % v


def load(snapshot):
    """The published methods and the evolved candidates, keyed by task."""
    rows = {}
    for r in json.load(open(PUBLISHED))["rows"]:
        rows.setdefault(r["task_id"], []).append(
            {"system": r["system"], "accuracy": r["accuracy"], "tokens": r["tokens"],
             "origin": "prior", "iteration": -1})
    for r in json.load(open(snapshot)):
        if r["accuracy"] is None or r["tokens"] is None:
            continue
        rows.setdefault(r["task_id"], []).append(
            {"system": r["system"], "accuracy": r["accuracy"], "tokens": r["tokens"],
             "origin": "prior" if r["kind"] == "seed" else "search",
             "iteration": r["iteration"]})
    for rs in rows.values():
        for r in rs:
            r["front"] = not any(
                (o["accuracy"] > r["accuracy"] and o["tokens"] <= r["tokens"]) or
                (o["accuracy"] >= r["accuracy"] and o["tokens"] < r["tokens"])
                for o in rs if o["system"] != r["system"])
    return rows


def draw_point(ax, x, y, r):
    """Fill says whether it survives on the frontier; shape says where it came from."""
    shape = "o" if r["origin"] == "prior" else "D"
    size = 78 if shape == "o" else 58
    if r["front"]:
        ax.scatter(x, y, s=size, marker=shape, color=FRONT, zorder=4)
        return FRONT
    ax.scatter(x, y, s=size * 0.82, marker=shape, facecolors="white", edgecolors=DOM,
               lw=1.5, zorder=3)
    return DOM


def place(ax, items, pad_pt=9.0, avoid=()):
    """Greedy placement with a widening spiral, all of it in device pixels.

    The box is measured from the drawn text rather than assumed, since the names run from `tot` to
    `baseline_no_questions`. Three things have to be right. The probe is measured in axes fraction,
    because (0, 0) in data coordinates is invalid on a log axis and the extent that comes back is
    degenerate. The offset is applied in pixels: at 200 dpi a point is 2.78 pixels, so geometry
    worked out in pixels and handed to `offset points` lands almost three times too far out. And
    placement runs after the layout is final, which moves the axes under any earlier measurement.
    """
    fig = ax.figure
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    scale = fig.dpi / 72.0
    pad = pad_pt * scale
    placed = []

    angles = [0, 180, 90, 270, 45, 315, 135, 225, 20, 160, 200, 340]
    offsets = [(pad * 2.4, 0.0), (-pad * 2.4, 0.0), (0.0, pad * 1.3), (0.0, -pad * 1.5)]
    for radius in (1.0, 1.6, 2.3, 3.2, 4.3, 5.6):
        for a in angles:
            offsets.append((math.cos(math.radians(a)) * pad * radius * 2.4,
                            math.sin(math.radians(a)) * pad * radius * 1.25))

    half = 7.0 * scale                       # the largest marker, in pixels
    for x, y, _t, _c, _d in items:
        mx, my = ax.transData.transform((x, y))
        placed.append((mx - half, my - half, mx + half, my + half))
    for x, y in avoid:                       # points along the frontier line
        mx, my = ax.transData.transform((x, y))
        placed.append((mx - 2.0, my - 2.0, mx + 2.0, my + 2.0))

    area = ax.get_window_extent(renderer=rend)
    inside = (area.x0 + 1, area.y0 + 1, area.x1 - 1, area.y1 - 1)

    for x, y, text, colour, droppable in sorted(items, key=lambda it: it[3] != FRONT):
        mx, my = ax.transData.transform((x, y))
        probe = ax.annotate(text, (0.5, 0.5), xycoords="axes fraction", fontsize=8.6, alpha=0)
        bb = probe.get_window_extent(renderer=rend)
        probe.remove()
        hw, hh = bb.width / 2 + 3.0 * scale, bb.height / 2 + 2.0 * scale
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
        if chosen is None:
            # A panel fills up as the search adds candidates. The points that carry the result --
            # the frontier and the published arms -- keep their labels whatever it costs; a
            # dominated candidate with nowhere clear to go gives its label up and stays a marker,
            # which is better than two names written over each other.
            if droppable:
                continue
            chosen = fallback if fallback is not None else (pad * 2.4, 0.0)
        dx, dy = chosen
        placed.append((mx + dx - hw, my + dy - hh, mx + dx + hw, my + dy + hh))
        kw = {"arrowprops": dict(arrowstyle="-", lw=0.6, color=RULE, shrinkA=1, shrinkB=6)} \
            if math.hypot(dx / 2.2, dy) > pad * 1.5 else {}
        ax.annotate(text, (x, y), textcoords="offset pixels", xytext=(dx, dy),
                    ha="center", va="center", fontsize=8.6, color=colour, zorder=6,
                    path_effects=[pe.withStroke(linewidth=2.6, foreground="white")], **kw)


def heading(ax, name, metric):
    ax.set_title(name, loc="left", pad=26, fontsize=11.5, color=INK)
    ax.text(0, 1.028, metric, transform=ax.transAxes, fontsize=10, color="#454b56", va="bottom")


def figure_frontier(rows, out, stamp):
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.0))
    pending = []
    for ax, (tid, name, metric) in zip(axes, TASKS):
        rs = rows[tid]
        ax.set_xscale("log")
        xs = [r["tokens"] for r in rs]
        ax.set_xlim(min(xs) * 0.42, max(xs) * 2.8)
        top = max(r["accuracy"] for r in rs) * 1.26 + 2
        ax.set_ylim(-top * 0.09, top)
        ax.xaxis.set_major_formatter(FuncFormatter(tick))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.grid(axis="y", color=RULE, lw=0.6, alpha=0.6)
        ax.set_axisbelow(True)
        heading(ax, name, metric)
        ax.set_xlabel("total tokens per episode")

        avoid = []
        if tid in CHANCE:
            ax.axhline(CHANCE[tid], color=RULE, ls="--", lw=1.2, zorder=1)
            ax.text(max(xs) * 2.6, CHANCE[tid] + top * 0.02, "chance %d%%" % CHANCE[tid],
                    ha="right", fontsize=8.6, color=MUTE)
            lo, hi = ax.get_xlim()           # the rule is drawn, so nothing should sit on it
            for f in range(0, 21):
                avoid.append((lo * (hi / lo) ** (f / 20), CHANCE[tid]))
        front = sorted([r for r in rs if r["front"]], key=lambda r: r["tokens"])
        if len(front) > 1:
            sx, sy = [], []
            for i, r in enumerate(front):
                if i:
                    sx.append(r["tokens"]); sy.append(front[i - 1]["accuracy"])
                sx.append(r["tokens"]); sy.append(r["accuracy"])
            ax.plot(sx, sy, color=FRONT, lw=1.4, ls=":", zorder=2)
            for i in range(len(sx) - 1):     # sample the segments as obstacles
                for f in (0.2, 0.4, 0.6, 0.8):
                    avoid.append((sx[i] * (1 - f) + sx[i + 1] * f,
                                  sy[i] * (1 - f) + sy[i + 1] * f))

        labels = []
        for r in rs:
            colour = draw_point(ax, r["tokens"], r["accuracy"], r)
            labels.append((r["tokens"], r["accuracy"], r["system"], colour,
                           not r["front"] and r["origin"] == "search"))
        pending.append((ax, labels, avoid))

    axes[0].set_ylabel("accuracy %")
    fig.text(0.006, 0.975, "AR-Bench harness evolution, run 1   ·   selection split, 100 episodes "
             "each, frozen Qwen2.5-7B-Instruct", fontsize=10, color="#454b56", va="top")
    fig.text(0.006, 0.935, "filled = on the frontier, hollow = dominated   ·   circle = published "
             "method or the no-questions anchor, diamond = written by the search",
             fontsize=10, color="#454b56", va="top")
    fig.text(0.006, 0.012, "Uncertainty-of-Thoughts held out  ·  an unlabelled marker is a "
             f"dominated candidate with no room for its name  ·  {stamp}",
             fontsize=8, color=MUTE)
    fig.tight_layout(rect=(0, 0.035, 1, 0.885))
    for ax, labels, avoid in pending:
        place(ax, labels, avoid=avoid)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"evolution_frontier.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  evolution_frontier.png / .pdf")


def figure_trajectory(rows, out, stamp):
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.0))
    pending = []
    for ax, (tid, name, metric) in zip(axes, TASKS):
        rs = [r for r in rows[tid] if r["iteration"] >= 0]      # the published arms have no place
        seq = sorted(rs, key=lambda r: (r["iteration"], -r["accuracy"], r["system"]))
        xs = list(range(len(seq)))
        run, best = -1.0, []
        for r in seq:
            run = max(run, r["accuracy"])
            best.append(run)
        ax.step(xs + [xs[-1] + 0.5], best + best[-1:], where="post",
                color=FRONT, lw=1.5, zorder=2, solid_capstyle="round")
        labels = []
        for x, r in zip(xs, seq):
            colour = draw_point(ax, x, r["accuracy"], r)
            labels.append((x, r["accuracy"], r["system"], colour,
                           not r["front"] and r["origin"] == "search"))

        ax.set_xticks([])
        groups = {}
        for x, r in zip(xs, seq):
            groups.setdefault(r["iteration"], []).append(x)
        for it, gx in sorted(groups.items()):
            if min(gx) > 0:
                ax.axvline(min(gx) - 0.5, color=RULE, lw=0.8, zorder=1)
            ax.annotate("anchor" if it == 0 else f"iteration {it}",
                        ((min(gx) + max(gx)) / 2, 0), xycoords=("data", "axes fraction"),
                        xytext=(0, -16), textcoords="offset points",
                        ha="center", va="top", fontsize=9, color=MUTE)
        lo = min(r["accuracy"] for r in seq)
        hi = max(r["accuracy"] for r in seq)
        span = max(hi - lo, 8)
        ax.set_ylim(max(lo - span * 0.16, -span * 0.06), hi + span * 0.30)
        ax.set_xlim(-0.6, xs[-1] + 0.6)
        ax.grid(axis="y", color=RULE, lw=0.6, alpha=0.6)
        ax.set_axisbelow(True)
        heading(ax, name, metric)
        ax.set_xlabel("candidates, in the order the proposer wrote them", labelpad=30)
        pending.append((ax, labels))

    axes[0].set_ylabel("accuracy %")
    fig.text(0.006, 0.975, "What each iteration produced   ·   the step line is the best selection "
             "accuracy reached so far", fontsize=10, color="#454b56", va="top")
    fig.text(0.006, 0.935, "filled = on the frontier against every published method too, hollow = "
             "dominated   ·   circle = anchor, diamond = written by the search",
             fontsize=10, color="#454b56", va="top")
    fig.text(0.006, 0.012, "Uncertainty-of-Thoughts held out  ·  an unlabelled marker is a "
             f"dominated candidate with no room for its name  ·  {stamp}",
             fontsize=8, color=MUTE)
    fig.tight_layout(rect=(0, 0.04, 1, 0.885))
    for ax, labels in pending:
        place(ax, labels)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"evolution_trajectory.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  evolution_trajectory.png / .pdf")


def main(snapshot, out_dir):
    style()
    rows = load(snapshot)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    out = Path(out_dir)
    figure_frontier(rows, out, stamp)
    figure_trajectory(rows, out, stamp)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "figures")
