"""Per-iteration attribution for a meta-harness evolution run: what each step proposed, what
later steps kept, and which signal was actually live.

Three things the loop records separately are joined here:
  evolution_summary.jsonl                     what each candidate scored on accuracy
  logs/<task>/<name>/<model>/val.json         its cost and LLM-call count
  pending_eval.json, as written inside the    its parent and its component tags
    proposer session's own tool logs
  reports/iteration_N.md                      the signal the proposer said it acted on

Two distinctions the raw summary does not make, and that change every conclusion drawn from it:

  * A candidate with no val.json scored 0.0 because the benchmark crashed, not because it was
    tried and lost. In runs/gn_fixed/logs/neutral, iterations 3-5 are all of this kind.
  * `avg_val` carries accuracy only. On GN accuracy saturates in iteration 1 and never moves
    again, so every later selection was made on cost, which lives only in val.json.

Usage:
    python lineage.py <run-log-dir> [task]      # e.g. runs/gn_fixed/logs/neutral1 gn
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def pending_blobs(run: Path) -> dict[str, list[dict]]:
    """Every pending_eval.json the proposer wrote, recovered from its tool logs.

    The file on disk holds only the last iteration, but each proposer session logged the Write
    (or heredoc Bash) that produced it, so the whole series is recoverable. A later write in the
    same iteration supersedes an earlier one.
    """
    out: dict[str, list[dict]] = {}
    for sess in sorted((run / "claude_sessions").glob("*")):
        for tool in sorted((sess / "tools").glob("*.txt")):
            txt = tool.read_text(errors="replace")
            if "pending_eval.json" not in txt or '"candidates"' not in txt:
                continue
            for m in re.finditer(r"\{", txt):
                try:
                    obj, _ = json.JSONDecoder().raw_decode(txt[m.start():])
                except ValueError:
                    continue
                if isinstance(obj, dict) and "candidates" in obj:
                    out[str(obj.get("iteration", ""))] = obj["candidates"]
                    break
    return out


def measured(run: Path, task: str) -> dict[str, dict]:
    """Each candidate's own val.json, keyed by harness name."""
    out: dict[str, dict] = {}
    if not (run / task).exists():
        return out
    for d in (run / task).rglob("val.json"):
        try:
            out[d.parent.parent.name] = json.loads(d.read_text())
        except ValueError:
            pass
    return out


def load(run: Path, task: str) -> list[dict]:
    rows = [json.loads(l) for l in (run / "evolution_summary.jsonl").read_text().splitlines()
            if l.strip()]
    meta = {(it, c["name"]): c for it, cands in pending_blobs(run).items() for c in cands}
    done = measured(run, task)
    for r in rows:
        m = meta.get((str(r["iteration"]), r["system"]), {})
        v = done.get(r["system"], {})
        r["base_system"] = m.get("base_system", "?")
        r["measured"] = r["system"] in done
        r["tokens"] = v.get("total_tokens")
        r["llm_calls"] = v.get("llm_calls")
    return rows


def report(run: Path, task: str) -> None:
    rows = load(run, task)
    unmeasured = [r for r in rows if not r["measured"]]
    rows = [r for r in rows if r["measured"]]
    if not rows:
        print(f"{run}: nothing was benchmarked ({len(unmeasured)} candidates proposed)")
        return

    by_it: dict[int, list[dict]] = {}
    for r in rows:
        by_it.setdefault(r["iteration"], []).append(r)
    its = sorted(by_it)
    winners = {it: min(by_it[it], key=lambda r: r["tokens"]) for it in its}
    final = min(rows, key=lambda r: (r["tokens"], r["iteration"]))

    print(f"=== {run} ===")
    if unmeasured:
        names = ", ".join(f"{r['system']}(i{r['iteration']})" for r in unmeasured)
        print(f"NOT BENCHMARKED, excluded: {names}\n")

    print(f"{'iter':<5} {'cheapest of iteration':<26} {'acc':>5} {'tok/ep':>9} {'calls/100ep':>12} {'vs best':>9}")
    best = None
    for it in its:
        w = winners[it]
        rel = "—" if best is None else f"{(w['tokens'] - best['tokens']) / best['tokens'] * 100:+.0f}%"
        print(f"{it:<5} {w['system']:<26} {w['avg_val']:>4.0f}% {w['tokens']:>9.1f} "
              f"{str(w['llm_calls']):>12} {rel:>9}")
        if best is None or w["tokens"] < best["tokens"]:
            best = w

    accs = {r["avg_val"] for r in rows}
    if len(accs) == 1:
        print(f"\nsignal: accuracy is {accs.pop():.0f}% for all {len(rows)} measured candidates, so it "
              f"carried no signal after iteration {its[0]}. Every later step was selected on cost.")

    origin: dict[str, tuple] = {}
    for it in its:
        for r in sorted(by_it[it], key=lambda r: r["system"]):
            for c in r["components"]:
                origin.setdefault(c, (it, r["system"], r["tokens"]))

    print(f"\nFINAL WINNER {final['system']} ({final['avg_val']:.0f}%, {final['tokens']:.1f} tok/ep, "
          f"{final['llm_calls']} call(s)/100ep)")
    print(f"  built on: {final['base_system']}")
    print("  where each of its components came from:")
    for c in sorted(final["components"]):
        it, who, tok = origin[c]
        print(f"    {c:<30} iter {it}, {who} ({tok:.1f} tok/ep)"
              f"{' (its own)' if who == final['system'] else ''}")

    print("\n  what earlier winners carried that the final one does not:")
    for it in its:
        w = winners[it]
        if w["system"] == final["system"]:
            continue
        gone = sorted(set(w["components"]) - set(final["components"]))
        if gone:
            print(f"    iter {it} {w['system']} ({w['tokens']:.1f} tok/ep): {', '.join(gone)}")

    # A tag reused by several later candidates is the search's own vote of confidence. A tag that
    # never reappears was abandoned -- except in the last iteration, where there is no later step
    # to reappear in, so silence there means nothing.
    print("\n  reused by >=2 later candidates:")
    reuse = []
    for c, (it, _who, _tok) in origin.items():
        n = sum(1 for r in rows if r["iteration"] > it and c in r["components"])
        if n >= 2:
            reuse.append((n, it, c))
    for n, it, c in sorted(reuse, reverse=True):
        print(f"    {c:<32} debut iter {it}, reused {n}x")

    dead = [c for c, (it, _w, _t) in origin.items()
            if it < its[-1] and not any(r["iteration"] > it and c in r["components"] for r in rows)]
    print(f"\n  abandoned after debut ({len(dead)} of {len(origin)} tags; iteration-{its[-1]} debuts "
          f"excluded, they had no later step):\n    {', '.join(sorted(dead))}")

    for it in its:
        rp = run / "reports" / f"iteration_{it}.md"
        if not rp.exists():
            continue
        tk = re.search(r"\*\*Takeaway\.\*\*(.+?)(?:\n\n|\Z)", rp.read_text(), re.S)
        if tk:
            print(f"\n  iter {it} stated takeaway: {' '.join(tk.group(1).split())}")


if __name__ == "__main__":
    report(Path(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "gn")
