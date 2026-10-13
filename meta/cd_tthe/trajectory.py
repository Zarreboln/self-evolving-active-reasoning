"""The evolution trajectory of one cd_tthe run (Circuit Decoding or Situation Puzzles): per batch, which harness governed, whether the
judge committed a change, what the change was (from the proposal card), and the label-free and
gold scores. Table to stdout, figure to <run>/trajectory.png.

    python trajectory.py logs/<run>
"""

import json
import sys
from pathlib import Path


def main(run):
    run = Path(run)
    rows = [json.loads(l) for l in (run / "opt_log.jsonl").open()]
    cards = {p.stem: json.loads(p.read_text()) for p in (run / "proposal_cards").glob("*.json")} if (run / "proposal_cards").exists() else {}
    # metrics by task: the row's prequential dict tells which task produced the run
    sp = "f1_char" in rows[0]["prequential"]
    mets = [("f1_char", "F1 (char)"), ("keypoint_recall", "key-q recall"), ("health", "run health")] if sp else \
           [("circuit_wise_accuracy", "cwa"), ("joint_truth_table_correct", "joint"), ("uniq_end", "uniq")]
    print(f"{'batch':>5} {'incoming':<26} {'committed':<26} " + " ".join(f"{'preq ' + l:>14}" for _, l in mets)
          + " " + " ".join(f"{'trans ' + l:>15}" for _, l in mets[:2]) + "  change")
    for r in rows:
        changed = r["committed"] != r["incoming"]
        card = cards.get(r["committed"], {})
        desc = (card.get("hypothesis") or "")[:70] if changed else ""
        print(f"{r['batch']:>5} {r['incoming'][:26]:<26} {r['committed'][:26]:<26} "
              + " ".join(f"{100*r['prequential'][m]:>13.0f}%" for m, _ in mets) + " "
              + " ".join(f"{100*r['transductive'][m]:>14.0f}%" for m, _ in mets[:2]) + f"  {'* ' + desc if changed else ''}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        x = [r["batch"] for r in rows]
        fig, ax = plt.subplots(figsize=(8, 3.6))
        main_lab = "F1, char-level (AR-Bench's metric)" if sp else "circuit-wise accuracy (the benchmark's metric)"
        ax.plot(x, [100 * r["prequential"][mets[0][0]] for r in rows], "o-", label=f"prequential {main_lab}")
        ax.plot(x, [100 * r["prequential"][mets[1][0]] for r in rows], "s-", label=f"prequential {mets[1][1]}" + (" (referee-judged key questions)" if sp else ""))
        ax.plot(x, [100 * r["prequential"][mets[2][0]] for r in rows], "^:", alpha=0.6, label=f"prequential {mets[2][1]}" + ("" if sp else " (ours)"))
        ax.plot(x, [100 * r["transductive"][mets[0][0]] for r in rows], "o--", alpha=0.5, label=f"transductive {mets[0][1]}")
        for r in rows:
            if r["committed"] != r["incoming"]:
                ax.axvline(r["batch"], color="gray", lw=0.6, ls=":")
        ax.set_xlabel("batch"); ax.set_ylabel("%"); ax.set_ylim(0, 102); ax.legend(fontsize=8)
        ax.set_title(f"{run.name}: harness evolution along the stream (dotted = judge committed a new harness)", fontsize=9)
        fig.tight_layout(); fig.savefig(run / "trajectory.png", dpi=150)
        print("->", run / "trajectory.png")
    except Exception as e:
        print("(no figure:", e, ")")


if __name__ == "__main__":
    main(sys.argv[1])
