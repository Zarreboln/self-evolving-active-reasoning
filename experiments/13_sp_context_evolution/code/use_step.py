"""Use-step context, multi-seed (2026-11-08): does context placed at the ANSWER step help the frozen 7B use what it
already found? No new questions: the 7B re-writes the final story for saved TEST transcripts of an arm, seeds 0-4.
Conditions (all retrieved by TF-IDF of TRAIN surfaces; TRAIN labels only, TEST labels never used):
  ex7b     2 solved TRAIN examples from the 7B's own evolution experience (probe_ex bank) + true story, right before the
           final prompt (same prompt as probe_ex.py; seed 7000 + run seed, so arm a0 seed 0 replays probe_ex from cache)
  tex7b    the same placement, but the 2 examples come from the teacher bank (Qwen3-235B transcripts + true story)
  facts7b  the 7B first lists confirmed / ruled-out facts, then writes the story (probe_use.py)
Scores: AR-Bench guess_eval key points matched (0-6) by Haiku and by Gemini (retried; failures stay None).
Usage: python -I code/use_step.py <arm> <cond> [seeds]   e.g.  python -I code/use_step.py a0 ex7b 0,1,2,3,4
       python -I code/use_step.py --report"""
import json
import random
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sp  # noqa: E402
import evaluate  # noqa: E402
import probe_use as pu  # noqa: E402
import probe_ex  # noqa: E402
import pilot  # noqa: E402
from llm import chat, spent_usd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TEST = ROOT / "runs" / "test"
OUT = ROOT / "runs" / "use_step"
TEX_INTRO = ("Before you answer, here are two solved examples from OTHER puzzles. Each shows how a strong player "
             "questioned it and the true hidden story the record was pointing to. Study how the confirmed facts were "
             "combined into one coherent story. Do not copy their content: your puzzle has a different story.\n\n")


def tex_examples(p):
    bank, vecs, vec = pilot._tindex("")
    from collections import Counter
    q = vec(Counter(pilot._toks(p["surface"])))
    best = sorted(range(len(bank)), key=lambda i: -sum(q.get(w, 0) * x for w, x in vecs[i].items()))[:2]
    parts = []
    for j, i in enumerate(best, 1):
        b = bank[i]
        qa = "\n".join(f"Q: {sp.clean_q(q_)} -> {a}" for q_, a in b["history"][:probe_ex.EX_TURNS])
        parts.append(f"Example {j}. Puzzle: {b['surface']}\n{qa}\nTrue hidden story: {b['bottom']}")
    return TEX_INTRO + "\n\n".join(parts) + "\n\nNow, for YOUR puzzle: "


FAR = 0.25   # ex7b_far: only TRAIN examples whose surface TF-IDF cosine to this puzzle is below FAR (no near variants)


def ex_variant(p, mode, k=2):
    """Leakage / mechanism controls for ex7b (2026-11-08). AR-Bench SP has near-variant puzzle families across TRAIN and
    TEST (e.g. several 'letter after her funeral' puzzles), and the ex7b gain is largest where the retrieved TRAIN story
    is most similar. Retrieval here uses TEST surfaces only, never TEST labels.
      rand     2 random bank examples (seeded by puzzle id): format / 'how to combine facts' effect, no content match
      far      the 2 most similar examples among those with surface cosine < FAR: no near-variant puzzle
      nostory  the usual 2 nearest examples, but without the true story (record only)"""
    q = probe_ex._vec(Counter(probe_ex._toks(p["surface"])))
    sims = [(sum(q.get(w, 0) * x for w, x in probe_ex._VECS[i].items()), i) for i in range(len(probe_ex.BANK))]
    if mode == "rand":
        best = random.Random(p["_id"]).sample(range(len(probe_ex.BANK)), k)
    elif mode == "far":
        best = [i for s, i in sorted(sims, reverse=True) if s < FAR][:k]
    else:
        best = [i for s, i in sorted(sims, reverse=True)][:k]
    parts = []
    for j, i in enumerate(best, 1):
        b = probe_ex.BANK[i]
        qa = "\n".join(f"Q: {x['q']} -> {x['a']}" for x in b["qa"][:probe_ex.EX_TURNS])
        tail = "" if mode == "nostory" else f"\nTrue hidden story: {b['bottom']}"
        parts.append(f"Example {j}. Puzzle: {b['surface']}\n{qa}{tail}")
    intro = NOSTORY_INTRO if mode == "nostory" else probe_ex.INTRO
    return intro + "\n\n".join(parts) + "\n\nNow, for YOUR puzzle: "


NOSTORY_INTRO = ("Before you answer, here are two question-answer records from OTHER puzzles, to show how a record is "
                 "read. Do not copy their content: your puzzle has a different story.\n\n")


def gem(p, pred):
    if not (pred or "").strip():
        return None
    for attempt in range(5):
        v = pu.gem_match(p, pred)
        if v is not None:
            return v
        time.sleep(6 * (attempt + 1))
    return None


def rewrite(cond, p, pb, hist, seed):
    if cond == "facts7b":
        return pu.story("facts7b", p, pb, hist, 7000 + seed)
    if cond == "plain7b":   # control: same transcript, same final prompt, new sample, no examples
        m = pu.msgs_for(p, pb, hist) + [{"role": "user", "content": sp.FINAL_PROMPT}]
        return chat(sp.ACTOR, m, role="actor", temperature=0.7, max_tokens=512, seed=7000 + seed,
                    extra={"top_p": 0.7}, tag=p["_id"])
    if cond.startswith("ex7b_"):
        intro = ex_variant(p, cond.split("_", 1)[1])
    else:
        intro = probe_ex.examples(p) if cond == "ex7b" else tex_examples(p)
    m = pu.msgs_for(p, pb, hist) + [{"role": "user", "content": intro + sp.FINAL_PROMPT}]
    return chat(sp.ACTOR, m, role="actor", temperature=0.7, max_tokens=512, seed=7000 + seed, extra={"top_p": 0.7},
                tag=p["_id"])


def run(arm, cond, seeds):
    OUT.mkdir(parents=True, exist_ok=True)
    for s in seeds:
        f = TEST / f"{arm}_s{s}.jsonl"
        rows = [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines()]

        def one(r):
            p = pu.PUZ[r["episode"]["id"]]
            pb = r_context(arm, s, p)
            pred = rewrite(cond, p, pb, r["episode"]["history"], s)
            return {"arm": arm, "seed": s, "cond": cond, "id": p["_id"], "pred": pred,
                    "haiku": sp.answer_match(p, pred) if (pred or "").strip() else None, "gemini": gem(p, pred),
                    "haiku_orig": r["score"]["answer_match"]}
        with ThreadPoolExecutor(8) as ex:
            res = list(ex.map(one, rows))
        (OUT / f"{arm}_s{s}_{cond}.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in res) + "\n",
                                                     encoding="utf-8")
        h = [x["haiku"] for x in res if x["haiku"] is not None]
        g = [x["gemini"] for x in res if x["gemini"] is not None]
        print(f"{arm} s{s} {cond}: haiku {mean(h):.2f} (orig {mean(x['haiku_orig'] for x in res):.2f}) "
              f"gemini {mean(g) if g else float('nan'):.2f} (n={len(g)}) spent ${spent_usd():.2f}", flush=True)


def r_context(arm, seed, p):
    """The context the episode was played with (so the rewrite sees the same system message)."""
    if arm in evaluate.PER_PUZZLE:
        return pilot.context_for(arm, seed, p)
    return evaluate.playbook_for(arm, seed)


def boot(d, b=10_000):
    rng = random.Random(0)
    n = len(d)
    ms = sorted(mean(d[rng.randrange(n)] for _ in range(n)) for _ in range(b))
    return mean(d), ms[int(.025 * b)], ms[int(.975 * b)]


def report():
    """Paired vs the original story of the same episode (Haiku: saved answer_match; Gemini: runs/test/story_gemini)."""
    lines = ["Use-step context: story key points (0-6) vs the original story of the same episodes; paired by puzzle,",
             "seeds averaged within puzzle; 95% bootstrap CI over puzzles.", ""]
    files = sorted(OUT.glob("*_s*_*.jsonl"))
    groups = {}
    for f in files:
        arm, rest = f.stem.split("_s", 1)
        s, cond = rest.split("_", 1)
        groups.setdefault((arm, cond), []).append((int(s), f))
    for (arm, cond), lst in sorted(groups.items()):
        hd, gd = {}, {}
        seeds = sorted(s for s, _ in lst)
        for s, f in lst:
            gorig_f = TEST / "story_gemini" / f"{arm}_s{s}.json"
            gorig = json.loads(gorig_f.read_text(encoding="utf-8")) if gorig_f.exists() else {}
            for line in f.read_text(encoding="utf-8").splitlines():
                x = json.loads(line)
                if x["haiku"] is not None:
                    hd.setdefault(x["id"], []).append(x["haiku"] - x["haiku_orig"])
                if x["gemini"] is not None and gorig.get(x["id"]) is not None:
                    gd.setdefault(x["id"], []).append(x["gemini"] - gorig[x["id"]])
        cells = []
        for name, d in (("Haiku", hd), ("Gemini", gd)):
            if d:
                m, lo, hi = boot([mean(v) for v in d.values()])
                cells.append(f"{name} {m:+.2f} [{lo:+.2f}, {hi:+.2f}] (n={len(d)})" + (" SIG" if lo > 0 or hi < 0 else ""))
        lines.append(f"- {arm} {cond} seeds {seeds}: " + "; ".join(cells))
    txt = "\n".join(lines)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "REPORT.md").write_text(txt + "\n", encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    if sys.argv[1] == "--report":
        report()
    else:
        seeds = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "0,1,2,3,4").split(",")]
        run(sys.argv[1], sys.argv[2], seeds)
        report()
