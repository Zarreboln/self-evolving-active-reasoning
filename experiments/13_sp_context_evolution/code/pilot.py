"""Pilot evaluation of the second batch of context methods on PILOT = VAL[:60].

PILOT puzzles come from the TRAIN file's VAL split. None of the new arms uses them as experience or for selection:
  bo8        best-of-8 self-selection distilled into rules, judge-free, ungated     (runs/evolve/bo8_s*)
  bo8g       same experience, judge-free pairwise-preference gate on VAL[60:75]     (runs/evolve/bo8g_s*)
  ex         exemplar memory: the 2 most similar TRAIN puzzles' best-of-8 question sequences (judge-free bank)
  hyb        bo8g rules + the ex exemplars
  traj_cons  trajectory contrast with a consensus-judged gate on 150 TRAIN puzzles     (runs/evolve/traj_cons_s*)
Baselines on the same puzzles and seeds: a0 (empty playbook) and r1 (test-time best-of-8, ~5x calls).
Scores per episode: key-point recall under Haiku (per question), Gemini (batched) and consensus (both on the same
question), plus the final story's char-F1 (no judge). Seeds 0 and 1.
Usage: python -I code/pilot.py bank | run <arm> <seed> | report
"""
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
import rescore_test as rt  # noqa: E402
from llm import spent_usd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "pilot"
EVO = ROOT / "runs" / "evolve"
BANK = OUT / "exemplar_bank.json"
BANK_N = 200
EX_K = 2
EX_TURNS = 15
WORKERS = 12

_TR, _VA = sp.split_train(sp.load("train"))
PILOT = _VA[:60]
_VA_BY_ID = {p["_id"]: p for p in _VA}


# ----------------------------------------------------------------------------- exemplar memory (ex, hyb)

def build_bank():
    """Best-of-8 self-selected episodes (the r1 procedure, no judge) on 200 TRAIN puzzles, seed 0."""
    ps = _TR[:BANK_N]

    def one(p):
        ep = sp.run_episode(p, playbook="", seed=0, choose=evaluate.r1_choose(p, 0))
        return {"id": p["_id"], "surface": p["surface"], "history": ep["history"]}
    with ThreadPoolExecutor(WORKERS) as ex:
        rows = list(ex.map(one, ps))
    OUT.mkdir(parents=True, exist_ok=True)
    BANK.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"bank: {len(rows)} episodes; spent ${spent_usd():.2f}", flush=True)


def _toks(s):
    return re.findall(r"[a-z']+", s.lower())


_IDX = None


def _index():
    global _IDX
    if _IDX is None:
        bank = json.loads(BANK.read_text(encoding="utf-8"))
        docs = [Counter(_toks(b["surface"])) for b in bank]
        df = Counter(w for d in docs for w in d)
        n = len(docs)
        idf = {w: math.log((1 + n) / (1 + c)) + 1 for w, c in df.items()}

        def vec(c):
            v = {w: tf * idf.get(w, math.log(1 + n) + 1) for w, tf in c.items()}
            norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
            return {w: x / norm for w, x in v.items()}
        _IDX = (bank, [vec(d) for d in docs], vec)
    return _IDX


def exemplars(p):
    bank, vecs, vec = _index()
    q = vec(Counter(_toks(p["surface"])))
    sims = sorted(((sum(q.get(w, 0) * x for w, x in v.items()), i) for i, v in enumerate(vecs)), reverse=True)
    parts = ["Solved examples from OTHER puzzles (different hidden stories). They show how a careful player "
             "questions; copy the questioning strategy, not the content."]
    for k, (_, i) in enumerate(sims[:EX_K], 1):
        b = bank[i]
        qa = "\n".join(f"Q{j}: {sp.clean_q(q)} -> {a}" for j, (q, a) in enumerate(b["history"][:EX_TURNS], 1))
        parts.append(f"Example {k}. Puzzle: {b['surface']}\n{qa}")
    return "\n\n".join(parts)


# ----------------------------------------------------------------------------- teacher exemplar memory (U-ask)
# Added 2026-11-07 late, after the strong-asker run: Qwen3-235B as asker doubled key-point recall on TEST and the 7B
# wrote 2.4x better stories from its transcripts. tex = 2 retrieved TRAIN transcripts of that strong asker plus the
# TRAIN puzzles' true stories, in the playbook slot (in-context distillation of a stronger asker; no judge involved).

TBANK = OUT / "teacher_bank.json"
TBANK_N = 150
TEX_TURNS = 20
TRAIN_BY_ID = {p["_id"]: p for p in sp.load("train")}


# Teacher banks (same 150 TRAIN puzzles, seed 0, empty playbook, same referee). "" = the original Qwen3-235B bank
# (teacher = referee model: a possible confound); the others are non-Qwen teachers added 2026-11-08.
TEACHERS = {"": sp.REFEREE, "g54m": "openai/gpt-5.4-mini", "g54": "openai/gpt-5.4"}


def tbank_path(tag=""):
    return TBANK if not tag else OUT / f"teacher_bank_{tag}.json"


def build_tbank(tag=""):
    ps = _TR[:TBANK_N]
    model = TEACHERS[tag]

    def one(p):
        ep = sp.run_episode(p, playbook="", seed=0, actor_model=model)
        return {"id": p["_id"], "surface": p["surface"], "history": ep["history"], "bottom": p["bottom"],
                "pred": ep["pred"]}
    with ThreadPoolExecutor(WORKERS) as ex:
        rows = list(ex.map(one, ps))
    bad = sum(1 for r in rows if not r["history"] or any(not q.strip() or not a.strip() for q, a in r["history"]))
    OUT.mkdir(parents=True, exist_ok=True)
    tbank_path(tag).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"teacher bank {tag or 'qwen235'} ({model}): {len(rows)} episodes, {bad} with empty turns; "
          f"spent ${spent_usd():.2f}", flush=True)


_TIDX = {}


def _tindex(tag):
    if tag not in _TIDX:
        bank = json.loads(tbank_path(tag).read_text(encoding="utf-8"))
        docs = [Counter(_toks(b["surface"])) for b in bank]
        df = Counter(w for d in docs for w in d)
        idf = {w: math.log((1 + len(docs)) / (1 + c)) + 1 for w, c in df.items()}

        def vec(c):
            v = {w: tf * idf.get(w, math.log(1 + len(docs)) + 1) for w, tf in c.items()}
            nrm = math.sqrt(sum(x * x for x in v.values())) or 1.0
            return {w: x / nrm for w, x in v.items()}
        _TIDX[tag] = (bank, [vec(d) for d in docs], vec)
    return _TIDX[tag]


def texemplars(p, tag="", mode="full", k=EX_K):
    """Teacher-exemplar memory. mode: full (Q/A + true story), nos (Q/A only, no story), sto (puzzle + true story,
    no Q/A), rand (full, but k random bank items chosen by a hash of the puzzle id instead of TF-IDF nearest)."""
    bank, vecs, vec = _tindex(tag)
    if mode == "rand":
        import random as _r
        best = _r.Random(p["_id"]).sample(range(len(bank)), k)
    else:
        q = vec(Counter(_toks(p["surface"])))
        best = sorted(range(len(bank)), key=lambda i: -sum(q.get(w, 0) * x for w, x in vecs[i].items()))[:k]
    n_word = {1: "one", 2: "two", 3: "three", 4: "four"}.get(k, str(k))
    if mode == "sto":
        parts = [f"Examples of {n_word} OTHER situation puzzles and their true hidden stories, to show what a complete "
                 "explanation looks like. Your puzzle has a different story."]
    elif mode == "nos":
        parts = [f"Examples of how a strong player questioned {n_word} OTHER puzzles. Copy the questioning strategy "
                 "(which hypotheses it tested first, how it narrowed down after a Yes, when it switched topic), not "
                 "the content: your puzzle has a different story."]
    else:
        parts = [f"Examples of how a strong player questioned {n_word} OTHER puzzles, and the true hidden story each "
                 "record pointed to. Copy the questioning strategy (which hypotheses it tested first, how it narrowed "
                 "down after a Yes, when it switched topic), not the content: your puzzle has a different story."]
    for j, i in enumerate(best, 1):
        b = bank[i]
        qa = "\n".join(f"Q{t}: {sp.clean_q(q_)} -> {a}" for t, (q_, a) in enumerate(b["history"][:TEX_TURNS], 1))
        if mode == "sto":
            parts.append(f"Example {j}. Puzzle: {b['surface']}\nTrue hidden story: {b['bottom']}")
        elif mode == "nos":
            parts.append(f"Example {j}. Puzzle: {b['surface']}\n{qa}")
        else:
            parts.append(f"Example {j}. Puzzle: {b['surface']}\n{qa}\nTrue hidden story: {b['bottom']}")
    return "\n\n".join(parts)


# tex variants for the mechanism / teacher ablations (2026-11-08): arm -> (teacher tag, mode, k)
TEX_ARMS = {"tex": ("", "full", 2), "tex_nos": ("", "nos", 2), "tex_sto": ("", "sto", 2),
            "tex_rand": ("", "rand", 2), "tex_k4": ("", "full", 4),
            "tex_g54m": ("g54m", "full", 2), "tex_g54": ("g54", "full", 2)}


# ----------------------------------------------------------------------------- arms

def rules_of(mech, seed):
    return json.loads((EVO / f"{mech}_s{seed}" / "final_playbook.json").read_text(encoding="utf-8"))["playbook"]


def context_for(arm, seed, p):
    if arm in ("a0", "r1"):
        return ""
    if arm in ("bo8", "bo8g", "traj_cons"):
        return rules_of(arm, seed)
    if arm == "ex":
        return exemplars(p)
    if arm in TEX_ARMS:
        tag, mode, k = TEX_ARMS[arm]
        return texemplars(p, tag=tag, mode=mode, k=k)
    if arm == "hyb":
        rules = rules_of("bo8g", seed)
        return (rules + "\n\n" + exemplars(p)) if rules else exemplars(p)
    raise ValueError(arm)


def score(p, ep):
    sc = sp.score_episode(p, ep)
    qs = [sp.clean_q(q) for q, _ in ep["history"]]
    hb = [sp.key_hits(p, q) for q in qs]
    gb = rt.batched_hits(p, qs)
    n = len(sp.keypoints_of(p))
    from probe_use import gem_match
    import time
    story_g = None
    for attempt in range(5):   # gem_match returns None on a rate-limited/unparsed reply: retry, never score as 0
        story_g = gem_match(p, ep["pred"]) if (ep["pred"] or "").strip() else None
        if story_g is not None or not (ep["pred"] or "").strip():
            break
        time.sleep(6 * (attempt + 1))
    out = {"haiku": len(set().union(*hb)) / n, "f1_char": sc["f1_char"], "answer_match": sc["answer_match"],
           "story_g": story_g}
    if gb is not None:
        out["gemini"] = len(set().union(*gb)) / n
        out["consensus"] = len(set().union(*[a & b for a, b in zip(hb, gb)])) / n
    return out


def run(arm, seed):
    def one(p):
        ctx = context_for(arm, seed, p)
        ep = sp.run_episode(p, playbook=ctx, seed=seed, choose=evaluate.r1_choose(p, seed) if arm == "r1" else None)
        return {"arm": arm, "seed": seed, "id": p["_id"], "context_words": len(ctx.split()), "episode": ep,
                "score": score(p, ep)}
    with ThreadPoolExecutor(WORKERS) as ex:
        rows = list(ex.map(one, PILOT))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{arm}_s{seed}.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                                              encoding="utf-8")
    m = {k: mean(r["score"][k] for r in rows if k in r["score"]) for k in ("haiku", "gemini", "consensus")}
    print(f"{arm} s{seed}: " + " ".join(f"{k} {v:.3f}" for k, v in m.items()) + f" spent ${spent_usd():.2f}",
          flush=True)


KEYS = ("haiku", "gemini", "consensus", "f1_char", "answer_match", "story_g")   # last two: final story (0-6)


def load(arm):
    per = {}
    for f in sorted(OUT.glob(f"{arm}_s*.jsonl")):
        if not re.fullmatch(rf"{re.escape(arm)}_s\d+\.jsonl", f.name):
            continue
        for line in f.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            per.setdefault(r["id"], []).append(r["score"])
    return {pid: {k: mean(s[k] for s in v if s.get(k) is not None) for k in KEYS
                  if any(s.get(k) is not None for s in v)} | {"n": len(v)}
            for pid, v in per.items()}


def report():
    arms = [a for a in ("a0", "r1", "bo8", "bo8g", "ex", "hyb", "traj_cons", *TEX_ARMS)
            if any(re.fullmatch(rf"{re.escape(a)}_s\d+\.jsonl", g.name) for g in OUT.glob(f"{a}_s*.jsonl"))]
    data = {a: load(a) for a in arms}
    lines = ["| arm | seeds | haiku | gemini | consensus | char-F1 | story (Haiku) | story (Gemini) |",
             "|---|---|---|---|---|---|---|---|"]
    for a in arms:
        d = data[a]
        lines.append(f"| {a} | {max(v['n'] for v in d.values())} | " +
                     " | ".join(f"{mean(v[k] for v in d.values() if k in v):.3f}" if any(k in v for v in d.values())
                                else "" for k in KEYS) + " |")
    for base in ("a0", "r1"):
        if base not in data:
            continue
        lines += ["", f"Paired difference vs {base} (95% bootstrap CI over the 60 pilot puzzles, seeds averaged):", ""]
        for a in arms:
            if a == base:
                continue
            cells = []
            for k in KEYS:
                common = [p for p in data[a] if p in data[base] and k in data[a][p] and k in data[base][p]]
                if common:
                    m, lo, hi = evaluate.boot([data[a][p][k] - data[base][p][k] for p in common])
                    cells.append(f"{k} {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
            lines.append(f"- {a} - {base}: " + "; ".join(cells))
    txt = "\n".join(lines)
    (OUT / "pilot_report.md").write_text(txt + "\n", encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    if sys.argv[1] == "bank":
        build_bank()
    elif sys.argv[1] == "tbank":
        build_tbank(sys.argv[2] if len(sys.argv) > 2 else "")
    elif sys.argv[1] == "backfill":     # add the Gemini story score to pilot rows scored before it existed
        from probe_use import gem_match
        for f in sorted(OUT.glob("*_s*.jsonl")):
            rows = [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines()]
            if all("story_g" in r["score"] for r in rows):
                continue
            with ThreadPoolExecutor(WORKERS) as ex:
                gs = list(ex.map(lambda r: r["score"].get("story_g") if "story_g" in r["score"]
                                 else gem_match(_VA_BY_ID[r["id"]], r["episode"]["pred"]), rows))
            for r, g in zip(rows, gs):
                r["score"]["story_g"] = g
            f.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
            print("backfilled", f.name, flush=True)
    elif sys.argv[1] == "run":
        run(sys.argv[2], int(sys.argv[3]))
    else:
        report()
