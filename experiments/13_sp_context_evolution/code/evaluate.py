"""Frozen TEST evaluation on the official AR-Bench SP test.json (100 puzzles). Run ONLY after playbooks are frozen.

Arms: a0 (empty playbook), a2 (reflector prior, no experience), and every evolved mechanism's final playbook.
Evolved arm seed s uses its own playbook with decoding seed s; a0/a2 use decoding seeds 0..2.
r1 = test-time repeated sampling: at every turn the 7B proposes K=8 questions and the 7B itself picks one
(same frozen model, no labels) - the "is it just repeated sampling?" baseline, with its extra calls counted.

Pre-registered primary endpoint: key-question recall at turn 25 (judge = sp.KEYJUDGE), paired by puzzle,
averaged over seeds within puzzle, 95% bootstrap CI over puzzles (10,000 resamples).
Secondary: AR-Bench answer key-point match (0..6), char-F1, word-F1, Unknown rate, answer length, calls/tokens.
Usage: python -I code/evaluate.py run <arm> <seed> [n]   |   python -I code/evaluate.py report
"""
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sp  # noqa: E402
from llm import spent_usd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
EVO = ROOT / "runs" / "evolve"
OUT = ROOT / "runs" / "test"
K = 8
WORKERS = 16


def playbook_for(arm, seed):
    if arm in ("a0", "r1"):
        return ""
    if arm == "a2":
        return json.loads((EVO / f"a2_s{seed}" / "final_playbook.json").read_text(encoding="utf-8"))["playbook"]
    return json.loads((EVO / f"{arm}_s{seed}" / "final_playbook.json").read_text(encoding="utf-8"))["playbook"]


SELECT = ("Here are candidate questions you could ask next in this situation puzzle:\n{cands}\n"
          "Which ONE question would reveal the most about the hidden story, given everything asked so far? "
          "Reply with its number only.")


def r1_choose(p, seed):
    # candidate/pick seeds depend on the run seed; before 2026-11-07 13:0x they did not, so r1 s1/s2 replayed s0's
    # cached candidates (identical recall 0.463 x3) and were discarded and rerun.
    def choose(msgs, t, hist):
        cands = [sp.actor_next(msgs, seed=70_000 + 10_000 * seed + 100 * j + t, tag=p["_id"]) for j in range(K)]
        listing = "\n".join(f"{j + 1}. {sp.clean_q(c)}" for j, c in enumerate(cands))
        pick = sp.actor_next(msgs + [{"role": "user", "content": SELECT.format(cands=listing)}],
                             seed=71_000 + 10_000 * seed + t, tag=p["_id"], temperature=0.0)
        m = re.search(r"\d+", pick or "")
        j = int(m.group(0)) - 1 if m else 0
        return cands[j if 0 <= j < K else 0]
    return choose


PER_PUZZLE = ("tex", "ex", "tex_nos", "tex_sto", "tex_rand", "tex_k4", "tex_g54m", "tex_g54")
# exemplar-memory arms: context retrieved per puzzle from TRAIN banks (code/pilot.py TEX_ARMS)


def run(arm, seed, n=100):
    ps = sp.load("test")[:n]
    pb = None if arm in PER_PUZZLE else playbook_for(arm, seed)

    def one(p):
        ctx = pb
        if ctx is None:
            import pilot
            ctx = pilot.context_for(arm, seed, p)
        ep = sp.run_episode(p, playbook=ctx, seed=seed, choose=r1_choose(p, seed) if arm == "r1" else None)
        sc = sp.score_episode(p, ep)
        return {"arm": arm, "seed": seed, "episode": ep, "score": sc}
    with ThreadPoolExecutor(WORKERS) as ex:
        rows = list(ex.map(one, ps))
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"{arm}_s{seed}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{arm} s{seed}: recall {mean(r['score']['key_recall'] for r in rows):.3f} "
          f"answer_match {mean(r['score']['answer_match'] for r in rows):.2f} spent ${spent_usd():.2f}", flush=True)


def load_arm(arm):
    per = {}
    for f in sorted(p for p in OUT.glob(f"{arm}_s*.jsonl") if re.fullmatch(rf"{re.escape(arm)}_s\d+\.jsonl", p.name)):
        for line in f.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            per.setdefault(r["score"]["id"], []).append(r["score"])
    return {pid: {k: mean(s[k] for s in v) for k in ("key_recall", "answer_match", "f1_char", "f1_word",
                                                     "unknown_rate", "pred_words")} | {"n_seeds": len(v)}
            for pid, v in per.items()}


def boot(diffs, b=10_000, seed=0):
    rng = random.Random(seed)
    n = len(diffs)
    ms = sorted(mean(diffs[rng.randrange(n)] for _ in range(n)) for _ in range(b))
    return mean(diffs), ms[int(0.025 * b)], ms[int(0.975 * b)]


def report():
    arms = sorted({f.name.rsplit("_s", 1)[0] for f in OUT.glob("*_s*.jsonl")})
    data = {a: load_arm(a) for a in arms}
    lines = ["| arm | seeds | key recall | answer match (0-6) | char-F1 | word-F1 | Unknown rate | answer words |",
             "|---|---|---|---|---|---|---|---|"]
    for a in arms:
        d = data[a]
        lines.append(f"| {a} | {max(v['n_seeds'] for v in d.values())} | {mean(v['key_recall'] for v in d.values()):.3f} | "
                     f"{mean(v['answer_match'] for v in d.values()):.2f} | {mean(v['f1_char'] for v in d.values()):.3f} | "
                     f"{mean(v['f1_word'] for v in d.values()):.3f} | {mean(v['unknown_rate'] for v in d.values()):.3f} | "
                     f"{mean(v['pred_words'] for v in d.values()):.0f} |")
    lines += ["", "Paired differences in key recall (95% bootstrap CI over puzzles):", ""]
    for a in arms:
        for b in arms:
            if a != b and b in ("a0", "a2", "traj", "cce_shuf", "r1"):
                common = sorted(set(data[a]) & set(data[b]))
                if common:
                    m, lo, hi = boot([data[a][p]["key_recall"] - data[b][p]["key_recall"] for p in common])
                    lines.append(f"- {a} - {b}: {m:+.3f} [{lo:+.3f}, {hi:+.3f}] (n={len(common)})")
    txt = "\n".join(lines)
    (OUT / "REPORT.md").write_text(txt + "\n", encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 100)
    else:
        report()
