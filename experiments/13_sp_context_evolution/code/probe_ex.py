"""Answer-step memory, first test: can past examples in context help the frozen 7B USE what it found?
No new questions are asked: the 7B re-writes the final story for the saved TEST transcripts (seed 0) of a0 and r1,
with 2 TRAIN examples placed before the final prompt. Each example = a TRAIN puzzle's surface, a question-answer
record from an earlier TRAIN episode (the best trajectory saved in runs/evolve/traj_s*/experience_r*.json) and the
TRUE hidden story (TRAIN labels; TEST labels are never used). Examples are retrieved by TF-IDF similarity of surfaces.
Condition name: ex7b. Scored exactly like probe_use.py (guess_eval match 0-6 under Haiku and Gemini, char-F1).
Usage: python -I code/probe_ex.py   -> runs/probe_use/<arm>_ex7b.jsonl; then probe_use.report() includes it.
"""
import glob
import json
import math
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sp  # noqa: E402
import evaluate  # noqa: E402
import probe_use as pu  # noqa: E402
from llm import chat, spent_usd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TRAIN = {p["_id"]: p for p in sp.load("train")}
EX_TURNS = 15
INTRO = ("Before you answer, here are two solved examples from OTHER puzzles. Each shows a question-answer record and "
         "the true hidden story it was pointing to. Study how the confirmed facts were combined into one coherent "
         "story. Do not copy their content: your puzzle has a different story.\n\n")


def build_bank():
    bank = {}
    for f in glob.glob(str(ROOT / "runs" / "evolve" / "traj_s*" / "experience_r*.json")):
        for g in json.loads(Path(f).read_text(encoding="utf-8")):
            pid = g["puzzle"]
            if pid in TRAIN and pid not in bank:
                bank[pid] = {"surface": g["surface"], "qa": g["best"]["questions"], "bottom": TRAIN[pid]["bottom"]}
    return list(bank.values())


BANK = build_bank()


def _toks(s):
    return re.findall(r"[a-z']+", s.lower())


_DOCS = [Counter(_toks(b["surface"])) for b in BANK]
_DF = Counter(w for d in _DOCS for w in d)
_IDF = {w: math.log((1 + len(_DOCS)) / (1 + c)) + 1 for w, c in _DF.items()}


def _vec(c):
    v = {w: tf * _IDF.get(w, math.log(1 + len(_DOCS)) + 1) for w, tf in c.items()}
    n = math.sqrt(sum(x * x for x in v.values())) or 1.0
    return {w: x / n for w, x in v.items()}


_VECS = [_vec(d) for d in _DOCS]


def examples(p, k=2):
    q = _vec(Counter(_toks(p["surface"])))
    best = sorted(range(len(BANK)), key=lambda i: -sum(q.get(w, 0) * x for w, x in _VECS[i].items()))[:k]
    parts = []
    for j, i in enumerate(best, 1):
        b = BANK[i]
        qa = "\n".join(f"Q: {x['q']} -> {x['a']}" for x in b["qa"][:EX_TURNS])
        parts.append(f"Example {j}. Puzzle: {b['surface']}\n{qa}\nTrue hidden story: {b['bottom']}")
    return INTRO + "\n\n".join(parts) + "\n\nNow, for YOUR puzzle: "


def main():
    print(f"bank: {len(BANK)} solved TRAIN examples", flush=True)
    for arm in ("a0", "r1"):
        rows = [json.loads(x) for x in (ROOT / "runs" / "test" / f"{arm}_s0.jsonl").read_text(encoding="utf-8")
                .splitlines()]
        pb = evaluate.playbook_for(arm, 0)

        def one(r):
            p = pu.PUZ[r["episode"]["id"]]
            m = pu.msgs_for(p, pb, r["episode"]["history"]) + [
                {"role": "user", "content": examples(p) + sp.FINAL_PROMPT}]
            pred = chat(sp.ACTOR, m, role="actor", temperature=0.7, max_tokens=512, seed=7000, extra={"top_p": 0.7},
                        tag=p["_id"])
            return {"arm": arm, "cond": "ex7b", "id": p["_id"], "pred": pred, "haiku": sp.answer_match(p, pred),
                    "gemini": pu.gem_match(p, pred), "f1_char": sp.f1(pred, p["bottom"])}
        with ThreadPoolExecutor(12) as ex:
            res = list(ex.map(one, rows))
        (pu.OUT / f"{arm}_ex7b.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in res) + "\n",
                                                 encoding="utf-8")
        g = [x["gemini"] for x in res if x["gemini"] is not None]
        print(f"{arm} ex7b: haiku {mean(x['haiku'] for x in res):.2f} gemini {mean(g):.2f} "
              f"f1 {mean(x['f1_char'] for x in res):.3f} spent ${spent_usd():.2f}", flush=True)
    pu.CONDS = pu.CONDS + ("ex7b",)
    pu.report()


if __name__ == "__main__":
    main()
