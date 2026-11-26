"""Repair published evaluation results whose export is stale (cluster side).

An export can be missing or stale while its summary is current: e.g. a broker export that failed on an
episode record and left the files of an earlier run. For every published result (results/<key>/summary.json)
this re-runs `broker export` for each task's sub-run under <runs>/<key>/out and republishes the export when
any task's scores.json is missing, says "rejected", or disagrees with that task's summary.

    python -m mb.evolve.reexport --mailbox ~/mailbox_w1 --runs ~/results/mb/evo_b
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from mb.evolve.remote import Mailbox


def subruns(out: Path):
    """(relative export dir, sub-run out dir) for a single-task or a multi-task request."""
    if (out / "cells").is_dir():
        return [(Path("cells") / c.name, c) for c in sorted((out / "cells").iterdir()) if c.is_dir()]
    return [(Path("."), out)]


def stale(exp: Path, out: Path) -> bool:
    for rel, sub in subruns(out):
        if not (sub / "summary.json").exists():
            continue
        f = exp / rel / "scores.json"
        if not f.exists():
            return True
        s = json.loads(f.read_text())
        want = json.loads((sub / "summary.json").read_text())
        if "rejected" in s or abs(s.get("mean_score", -1) - want.get("mean_score", -2)) > 1e-9:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mailbox", required=True)
    ap.add_argument("--runs", required=True)
    a = ap.parse_args()
    box, runs = Mailbox(Path(a.mailbox).expanduser()), Path(a.runs).expanduser()
    box.pull()
    fixed = []
    for summ in sorted((box.repo / "results").rglob("summary.json")):
        res = summ.parent
        key = str(res.relative_to(box.repo / "results"))
        if res.parent.name == "export" or "/export/" in key:
            continue
        work = runs / key
        if not (work / "out").is_dir() or not (res / "export").is_dir():
            continue
        if not stale(res / "export", work / "out"):
            continue
        for rel, sub in subruns(work / "out"):
            if (sub / "summary.json").exists():
                subprocess.run([sys.executable, "-m", "mb.evolve.broker", "export", "--out", str(sub),
                                "--dest", str(work / "export" / rel)], capture_output=True, text=True)
        with box.lock():
            shutil.rmtree(res / "export", ignore_errors=True)
            shutil.copytree(work / "export", res / "export")
        fixed.append(key)
    if fixed:
        box.push("re-export: " + ", ".join(fixed))
    print("re-exported:", fixed or "none")


if __name__ == "__main__":
    main()
