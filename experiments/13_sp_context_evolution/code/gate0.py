"""Gate 0: is there room for context evolution to improve information acquisition (IA) on SP?

All probes run on TRAIN puzzles only (official sp/train.json); test.json is never touched here.
  baseline : bare official loop, 1 episode per puzzle
  branch   : at fixed turns, K candidate questions sampled from the SAME dialogue state; each scored by
             NEW key-question hits (privileged judge). Measures step-level headroom and how often
             candidates differ (credit signal exists only if they differ).
  oracle   : full episodes where each turn takes the best of K candidates by new key hits (upper bound of
             what selecting among the 7B's own questions could achieve; NOT deployable - uses labels)
  steer    : does the frozen 7B follow playbook text at all? Two mechanically checkable rules.
  prior    : privileged category hint (someone_dies / supernatural) in the playbook - upper bound of a
             learned prior over puzzle types.
Outputs: runs/gate0/<probe>.jsonl + runs/gate0/summary.json. Real numbers only when run with a key.
Usage: python -I code/gate0.py <probe> [n_puzzles]
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sp  # noqa: E402
from llm import spent_usd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "gate0"
BRANCH_TURNS = (1, 5, 10, 15, 20)
K = 8
WORKERS = 24


def puzzles(n):
    tr, _ = sp.split_train(sp.load("train"))
    return tr[:n]


def pmap(fn, items):
    with ThreadPoolExecutor(WORKERS) as ex:
        return list(ex.map(fn, items))


def save(name, rows):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"{name}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def hits_so_far(p, hist):
    seen = set()
    for q, _ in hist:
        seen |= sp.key_hits(p, sp.clean_q(q))
    return seen


def run_baseline(ps, playbook="", seed=0, tag="baseline"):
    def one(p):
        ep = sp.run_episode(p, playbook=playbook, seed=seed)
        return {"episode": ep, "score": sp.score_episode(p, ep)}
    rows = pmap(one, ps)
    save(tag, rows)
    return rows


def candidates(p, hist, t, k=K, playbook=""):
    msgs = [{"role": "system", "content": sp.system_message(p["surface"], playbook)}]
    for i, (q, a) in enumerate(hist, 1):
        msgs += [{"role": "user", "content": f"Turn {i}: please propose your next question."},
                 {"role": "assistant", "content": q}, {"role": "user", "content": a}]
    msgs.append({"role": "user", "content": f"Turn {t}: please propose your next question."})
    return [sp.actor_next(msgs, seed=10_000 + 100 * j + t, tag=p["_id"]) for j in range(k)]


def run_branch(ps, base_rows):
    by_id = {r["episode"]["id"]: r["episode"] for r in base_rows}

    def one(p):
        ep = by_id[p["_id"]]
        groups = []
        for t in BRANCH_TURNS:
            prefix = ep["history"][:t - 1]
            seen = hits_so_far(p, prefix)
            cands = candidates(p, prefix, t)
            new = [len(sp.key_hits(p, sp.clean_q(c)) - seen) for c in cands]
            groups.append({"t": t, "already": len(seen), "new_hits": new,
                           "realized_new": len(sp.key_hits(p, sp.clean_q(ep["history"][t - 1][0])) - seen),
                           "n_distinct": len({sp.clean_q(c).lower() for c in cands})})
        return {"id": p["_id"], "groups": groups}
    rows = pmap(one, ps)
    save("branch", rows)
    return rows


def run_oracle(ps, seed=0):
    def one(p):
        def choose(msgs, t, hist):
            seen = hits_so_far(p, hist)
            cands = [sp.actor_next(msgs, seed=20_000 + 100 * j + t, tag=p["_id"]) for j in range(K)]
            gains = [len(sp.key_hits(p, sp.clean_q(c)) - seen) for c in cands]
            return cands[max(range(K), key=lambda j: gains[j])]
        ep = sp.run_episode(p, seed=seed, choose=choose)
        return {"episode": ep, "score": sp.score_episode(p, ep)}
    rows = pmap(one, ps)
    save("oracle", rows)
    return rows


STEER_PB = ("- Begin every question with the exact words \"Is it true that\".\n"
            "- Your very first question must ask whether anyone in the story died.")


def run_steer(ps):
    rows = run_baseline(ps, playbook=STEER_PB, tag="steer")
    for r in rows:
        qs = [sp.clean_q(q) for q, _ in r["episode"]["history"]]
        r["compliance_prefix"] = mean(q.lower().startswith("is it true that") for q in qs)
        r["compliance_first"] = any(w in qs[0].lower() for w in ("die", "dead", "death", "killed")) if qs else False
    save("steer", rows)
    return rows


def run_prior(ps):
    out = []
    for p in ps:
        hint = (f"- For THIS puzzle: someone dies = {'yes' if p.get('someone_dies') else 'no'}; "
                f"supernatural elements = {'yes' if p.get('supernatural') else 'no'}.")
        out.append((p, hint))

    def one(ph):
        p, hint = ph
        ep = sp.run_episode(p, playbook=hint, seed=0)
        return {"episode": ep, "score": sp.score_episode(p, ep)}
    rows = pmap(one, out)
    save("prior", rows)
    return rows


def summarize():
    def load(name):
        f = OUT / f"{name}.jsonl"
        return [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines()] if f.exists() else []
    s = {}
    for name in ("baseline", "oracle", "prior", "steer"):
        rows = load(name)
        if rows:
            sc = [r["score"] for r in rows]
            s[name] = {"n": len(sc), "key_recall": mean(x["key_recall"] for x in sc),
                       "answer_match": mean(x["answer_match"] for x in sc),
                       "f1_char": mean(x["f1_char"] for x in sc), "unknown_rate": mean(x["unknown_rate"] for x in sc)}
            if name == "steer":
                s[name]["compliance_prefix"] = mean(r["compliance_prefix"] for r in rows)
                s[name]["compliance_first"] = mean(r["compliance_first"] for r in rows)
    br = load("branch")
    if br:
        gs = [g for r in br for g in r["groups"]]
        s["branch"] = {"groups": len(gs),
                       "frac_discriminative": mean(len(set(g["new_hits"])) > 1 for g in gs),
                       "mean_new_hits_single": mean(mean(g["new_hits"]) for g in gs),
                       "mean_new_hits_best_of_k": mean(max(g["new_hits"]) for g in gs),
                       "mean_realized_new": mean(g["realized_new"] for g in gs),
                       "mean_distinct_questions": mean(g["n_distinct"] for g in gs)}
    s["spent_usd_ledger"] = spent_usd()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(s, indent=1), encoding="utf-8")
    print(json.dumps(s, indent=1))
    return s


if __name__ == "__main__":
    probe = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    ps = puzzles(n)
    if probe == "baseline":
        run_baseline(ps)
    elif probe == "branch":
        base = [json.loads(x) for x in (OUT / "baseline.jsonl").read_text(encoding="utf-8").splitlines()]
        run_branch(ps, base)
    elif probe == "oracle":
        run_oracle(ps)
    elif probe == "steer":
        run_steer(ps)
    elif probe == "prior":
        run_prior(ps)
    summarize()
