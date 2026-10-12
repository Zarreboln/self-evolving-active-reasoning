"""Fix the CircuitDecoding search / held-out split once.

300 released instances = 3 inventories x 100. The search split (what the
proposer sees and candidates are selected on) takes 50, stratified 17/17/16
over the inventories; the other 250 are held out and are only ever scored at
a frozen finalize step. Also writes the search instances to the proposer
workspace so prototyping has real inputs without the held-out file.

    python runner/cd_make_split.py            # idempotent; refuses to overwrite a different split
"""

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "mt_puzzles/cd_test.jsonl"
OUT = ROOT / "splits/cd_split.json"
WS_DATA = ROOT / "meta/cd_evolve/data/cd_search.jsonl"
SEED, N_SEARCH = 20261010, 50


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-search", type=int, default=N_SEARCH)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--extend", default=None,
                    help="existing split whose search keys are kept and extended (superset split)")
    a = ap.parse_args()
    out_path = Path(a.out)
    ws_data = WS_DATA.parent / (out_path.stem.replace("cd_split", "cd_search") + ".jsonl")
    base = set(json.loads(Path(a.extend).read_text())["search"]) if a.extend else set()
    lines = DATA.read_text().splitlines()
    by_inv = {}
    for i, line in enumerate(lines):
        comp = json.loads(line)["controller"]["components"]
        by_inv.setdefault(json.dumps(comp, sort_keys=True), []).append(i)
    rng = random.Random(SEED)
    keys = sorted(by_inv)
    quota = [a.n_search // len(keys) + (1 if j < a.n_search % len(keys) else 0) for j in range(len(keys))]
    search = []
    for key, n in zip(keys, quota):
        kept = sorted(set(by_inv[key]) & base)
        pool = sorted(set(by_inv[key]) - base)
        rng.shuffle(pool)
        search += sorted(kept + pool[:max(0, n - len(kept))])
    search = sorted(search)
    heldout = [i for i in range(len(lines)) if i not in set(search)]
    split = {"seed": SEED, "extends": a.extend, "n_search": len(search), "n_heldout": len(heldout),
             "search": search, "heldout": heldout,
             "search_by_inventory": {k: sorted(set(v) & set(search)) for k, v in by_inv.items()}}
    if out_path.exists():
        old = json.loads(out_path.read_text())
        if old["search"] != split["search"]:
            sys.exit(f"{out_path} exists with a DIFFERENT split; refusing to overwrite")
        print("split unchanged")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(split, indent=1))
    ws_data.parent.mkdir(parents=True, exist_ok=True)
    with ws_data.open("w") as f:
        for i in search:
            d = json.loads(lines[i])
            d["orig_index"] = i
            f.write(json.dumps(d) + "\n")
    print(f"search {len(search)} / heldout {len(heldout)} -> {out_path}; workspace copy -> {ws_data}")
    for k, v in split["search_by_inventory"].items():
        print(f"  {k}: {len(v)}")


if __name__ == "__main__":
    main()
