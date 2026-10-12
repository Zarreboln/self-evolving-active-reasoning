"""The evolution trajectory of one cd_tthe run: per batch, which harness governed, whether the
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
    print(f"{'batch':>5} {'incoming':<26} {'committed':<26} {'preq cwa':>9} {'preq joint':>10} {'preq uniq':>9} {'trans cwa':>9} {'trans joint':>11}  change")
    for r in rows:
        changed = r["committed"] != r["incoming"]
        card = cards.get(r["committed"], {})
        desc = (card.get("hypothesis") or "")[:70] if changed else ""
        print(f"{r['batch']:>5} {r['incoming'][:26]:<26} {r['committed'][:26]:<26} "
              f"{100*r['prequential']['circuit_wise_accuracy']:>8.0f}% {100*r['prequential']['joint_truth_table_correct']:>9.0f}% {100*r['prequential']['uniq_end']:>8.0f}% "
              f"{100*r['transductive']['circuit_wise_accuracy']:>8.0f}% {100*r['transductive']['joint_truth_table_correct']:>10.0f}%  {'* ' + desc if changed else ''}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        x = [r["batch"] for r in rows]
        fig, ax = plt.subplots(figsize=(8, 3.6))
        ax.plot(x, [100 * r["prequential"]["circuit_wise_accuracy"] for r in rows], "o-", label="prequential circuit-wise accuracy (deployed; the benchmark's metric)")
        ax.plot(x, [100 * r["prequential"]["joint_truth_table_correct"] for r in rows], "s-", label="prequential joint truth table correct")
        ax.plot(x, [100 * r["prequential"]["uniq_end"] for r in rows], "^:", alpha=0.6, label="prequential uniq@end (ours)")
        ax.plot(x, [100 * r["transductive"]["circuit_wise_accuracy"] for r in rows], "o--", alpha=0.5, label="transductive circuit-wise accuracy")
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
