"""Context evolution on AR-Bench SP: several mechanisms under one protocol, so that differences are attributable.

Evolved object: a short PLAYBOOK of state-conditioned questioning rules, read by the frozen Qwen2.5-7B through the
single playbook slot of the official loop (sp.system_message). Harness, parser, referee and budget never change.

Mechanisms (same reflector, same rule schema, same VAL gate, same number of rounds; only the EXPERIENCE differs):
  cce      step-level counterfactual credit: at fixed turns of TRAIN episodes, K questions are sampled from the SAME
           dialogue state (K-1 from the 7B + 1 from a stronger teacher model) and each is scored by the NEW key
           points it hits (privileged TRAIN labels) and by the referee's answer. The reflector sees these scored
           comparison groups.
  cce_shuf identical to cce but scores are permuted within each group before reflection (credit vs exposure).
  traj     trajectory-level contrast (Training-Free-GRPO style): G full episodes per TRAIN puzzle, each scored by
           key-question recall; the reflector compares best vs worst trajectories. G is set so that actor calls
           per round match cce.
  prior    hypothesis prior: the reflector reads TRAIN puzzles' surfaces, hidden bottoms and key questions plus the
           7B's questions and writes rules about which hypotheses/aspects to probe (learning from labels, no
           counterfactual sampling).
Controls run elsewhere: empty playbook (A0) and a Claude-written playbook from the task description only (A2).

Update: reflector proposes <=3 delta ops (ADD/MODIFY/DELETE); the candidate playbook replaces the incumbent only if
its paired VAL key-recall gain over the incumbent is > 0 on the same VAL puzzles and seeds. TEST is never used.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sp  # noqa: E402
from llm import chat, spent_usd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "evolve"
REFLECTOR = "anthropic/claude-opus-5.5"
TEACHER = "qwen/qwen3-235b-a22b-2507"
BRANCH_TURNS = (2, 6, 10, 14, 18)
K_STUDENT = 6
TRAIN_PER_ROUND = 10
VAL_N = 60
ROUNDS = 4
MAX_RULES = 12
MAX_WORDS = 400          # 260 rejected most rounds of the 10:04 runs (Opus writes long rules); see history.json
MIN_GAIN = 0.01          # accept a candidate only if paired VAL recall gain > MIN_GAIN (noise guard)
WORKERS = int(__import__("os").environ.get("MAS_WORKERS", "12"))


# ----------------------------------------------------------------------------- playbook

def render(rules: list[dict]) -> str:
    return "\n".join(f"- [{r['id']}] When {r['when'].rstrip('.')}: {r['do'].rstrip('.')}." for r in rules)


def words(rules):
    return len(render(rules).split())


# ----------------------------------------------------------------------------- data

def splits():
    tr, va = sp.split_train(sp.load("train"))
    return tr, va[:VAL_N]


def pmap(fn, items, workers=WORKERS):
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(fn, items))


# Judge used INSIDE the evolution loop (VAL selection and experience ranking). Default: sp.KEYJUDGE (Haiku,
# per-question). Mechanism names ending in "_gem" (e.g. traj_gem, added 2026-11-07 after the TEST judge check showed
# Haiku over-credits the questions traj playbooks produce) use the stricter Gemini judge with the batched protocol of
# code/rescore_test.py instead, so Haiku stays a held-out judge for those playbooks.
EVOLVE_JUDGE = "haiku"


def base_of(mech):
    """Experience type of a mechanism name: bo8g (pref-gated bo8) -> bo8; *_gem / *_cons -> the plain mechanism."""
    return "bo8" if mech == "bo8g" else re.sub(r"_(gem|cons)$", "", mech)


def recall_of(p, hist):
    qs = [sp.clean_q(q) for q, _ in hist]
    if EVOLVE_JUDGE == "consensus":   # traj_cons: a key point counts only if Haiku AND Gemini credit the same question
        from rescore_test import batched_hits
        gb = batched_hits(p, qs)
        if gb is None:
            raise RuntimeError(f"gemini judge failed on {p['_id']}")
        seen = set()
        for q, g in zip(qs, gb):
            seen |= sp.key_hits(p, q) & g
    elif EVOLVE_JUDGE == "gemini":
        from rescore_test import batched_hits
        hits = batched_hits(p, qs)
        if hits is None:
            raise RuntimeError(f"gemini judge failed on {p['_id']}")
        seen = set().union(*hits) if hits else set()
    else:
        seen = set()
        for q in qs:
            seen |= sp.key_hits(p, q)
    return len(seen) / len(sp.keypoints_of(p)), seen


def eval_playbook(rules, puzzles, seed):
    pb = render(rules)

    def one(p):
        ep = sp.run_episode(p, playbook=pb, seed=seed)
        try:
            r, _ = recall_of(p, ep["history"])
        except RuntimeError:          # judge reply unparseable after retries: this puzzle is left out of the comparison
            r = None
        return {"id": p["_id"], "recall": r, "unknown": mean(a.lower().startswith("unknown") for _, a in ep["history"])}
    return [x for x in pmap(one, puzzles) if x["recall"] is not None]


# ----------------------------------------------------------------------------- experience collection

def msgs_at(p, pb, hist, t):
    m = [{"role": "system", "content": sp.system_message(p["surface"], pb)}]
    for i, (q, a) in enumerate(hist, 1):
        m += [{"role": "user", "content": f"Turn {i}: please propose your next question."},
              {"role": "assistant", "content": q}, {"role": "user", "content": a}]
    m.append({"role": "user", "content": f"Turn {t}: please propose your next question."})
    return m


def teacher_question(p, hist, t):
    m = msgs_at(p, "", hist, t)
    q = chat(TEACHER, m, role="teacher", temperature=0.7, max_tokens=120, seed=t, tag=p["_id"])
    return q.strip()


def collect_cce(rules, puzzles, seed, shuffle=False):
    pb = render(rules)

    def one(p):
        ep = sp.run_episode(p, playbook=pb, seed=seed)
        groups = []
        for t in BRANCH_TURNS:
            prefix = ep["history"][:t - 1]
            _, seen = recall_of(p, prefix)
            m = msgs_at(p, pb, prefix, t)
            cands = [sp.actor_next(m, seed=50_000 + 97 * seed + 100 * j + t, tag=p["_id"]) for j in range(K_STUDENT)]
            cands.append(teacher_question(p, prefix, t))
            src = ["student"] * K_STUDENT + ["teacher"]
            scored = []
            for c, s in zip(cands, src):
                q = sp.clean_q(c)
                new = sorted(sp.key_hits(p, q) - seen)
                ans = sp.referee(p, c)
                scored.append({"q": q, "src": s, "new_keypoints": len(new), "referee": ans})
            if shuffle:
                vals = [(x["new_keypoints"], x["referee"]) for x in scored]
                rs = int(hashlib.sha256(f"{p['_id']}|{t}|{seed}".encode()).hexdigest()[:8], 16)
                random.Random(rs).shuffle(vals)
                for x, v in zip(scored, vals):
                    x["new_keypoints"], x["referee"] = v
            groups.append({"puzzle": p["_id"], "surface": p["surface"], "turn": t,
                           "recent": [{"q": sp.clean_q(q), "a": a} for q, a in prefix[-4:]],
                           "n_found_so_far": len(seen), "candidates": scored})
        return groups
    return [g for gs in pmap(one, puzzles) for g in gs]


def collect_traj(rules, puzzles, seed, g=3):
    pb = render(rules)

    def one(p):
        eps = []
        for j in range(g):
            ep = sp.run_episode(p, playbook=pb, seed=seed * 100 + j)
            try:
                r, _ = recall_of(p, ep["history"])
            except RuntimeError:      # unparseable judge reply: rank this trajectory by the Haiku-only recall
                r = len(set().union(*[sp.key_hits(p, sp.clean_q(q)) for q, _ in ep["history"]])) / len(sp.keypoints_of(p))
            eps.append({"recall": r, "questions": [{"q": sp.clean_q(q), "a": a} for q, a in ep["history"]]})
        eps.sort(key=lambda e: -e["recall"])
        return {"puzzle": p["_id"], "surface": p["surface"], "best": eps[0], "worst": eps[-1],
                "recalls": [e["recall"] for e in eps]}
    return pmap(one, puzzles)


def collect_prior(rules, puzzles, seed):
    pb = render(rules)

    def one(p):
        ep = sp.run_episode(p, playbook=pb, seed=seed)
        r, seen = recall_of(p, ep["history"])
        kps = sp.keypoints_of(p)
        return {"puzzle": p["_id"], "surface": p["surface"], "hidden_story": p["bottom"],
                "key_points": kps, "missed_key_points": [k for i, k in enumerate(kps, 1) if i not in seen],
                "student_questions": [sp.clean_q(q) for q, _ in ep["history"]], "recall": r}
    return pmap(one, puzzles)


BO8_K = 8
BO8_SELECT = ("Here are candidate questions you could ask next in this situation puzzle:\n{cands}\n"
              "Which ONE question would reveal the most about the hidden story, given everything asked so far? "
              "Reply with its number only.")   # identical to evaluate.SELECT (the r1 arm's self-selection prompt)


def collect_bo8(rules, puzzles, seed):
    """Amortizing test-time best-of-8 (added 2026-11-07 evening). At fixed turns of a TRAIN episode played with the
    current playbook, the 7B samples 8 questions from the SAME state and itself picks one (the r1 procedure); each
    candidate also gets the referee's answer (environment feedback, no labels). No key-point judge is used anywhere,
    so nothing in this mechanism can learn a judge's leniency."""
    pb = render(rules)

    def one(p):
        ep = sp.run_episode(p, playbook=pb, seed=seed)
        groups = []
        for t in BRANCH_TURNS:
            prefix = ep["history"][:t - 1]
            m = msgs_at(p, pb, prefix, t)
            cands = [sp.clean_q(sp.actor_next(m, seed=80_000 + 97 * seed + 100 * j + t, tag=p["_id"]))
                     for j in range(BO8_K)]
            listing = "\n".join(f"{j + 1}. {c}" for j, c in enumerate(cands))
            pick = sp.actor_next(m + [{"role": "user", "content": BO8_SELECT.format(cands=listing)}],
                                 seed=81_000 + 97 * seed + t, tag=p["_id"], temperature=0.0)
            mm = re.search(r"\d+", pick or "")
            j = int(mm.group(0)) - 1 if mm else 0
            j = j if 0 <= j < BO8_K else 0
            groups.append({"surface": p["surface"], "turn": t,
                           "recent": [{"q": sp.clean_q(q), "a": a} for q, a in prefix[-4:]],
                           "candidates": [{"q": c, "picked": i == j, "referee": sp.referee(p, c)}
                                          for i, c in enumerate(cands)]})
        return groups
    return [g for gs in pmap(one, puzzles) for g in gs]


GATE_PUZZLES = slice(60, 75)          # VAL[60:75]: never used as experience or as the pilot set (VAL[:60])
PREF_MARGIN = 0.55
COMPARATORS = (sp.ACTOR, TEACHER)     # the 7B's own side-by-side preference + a larger model (Qwen3-235B)
PAIR_SELECT = ("Here are two candidate questions you could ask next in this situation puzzle:\n1. {a}\n2. {b}\n"
               "Which ONE question would reveal the most about the hidden story, given everything asked so far? "
               "Reply with its number only.")


def _pick2(model, m, a, b, seed, tag):
    txt = chat(model, m + [{"role": "user", "content": PAIR_SELECT.format(a=a, b=b)}], role="comparator",
               temperature=0.0, max_tokens=8, seed=seed, tag=tag)
    mm = re.search(r"[12]", txt or "")
    return int(mm.group(0)) if mm else None


def pref_gate(old_rules, new_rules, puzzles, seed):
    """Judge-free acceptance test for bo8g. At 5 states of each gate puzzle (episodes of the OLD playbook), one
    question is written under the old and one under the new playbook (same decoding seed); each comparator sees both,
    in both orders, from a neutral context (no playbook, no hidden story) and picks the more revealing one.
    Returns the new playbook's win rate per comparator (ties and identical questions = 0.5)."""
    old_pb, new_pb = render(old_rules), render(new_rules)

    def one(p):
        ep = sp.run_episode(p, playbook=old_pb, seed=seed)
        rows = []
        for t in BRANCH_TURNS:
            prefix = ep["history"][:t - 1]
            s = 90_000 + 97 * seed + t
            qo = sp.clean_q(sp.actor_next(msgs_at(p, old_pb, prefix, t), seed=s, tag=p["_id"]))
            qn = sp.clean_q(sp.actor_next(msgs_at(p, new_pb, prefix, t), seed=s, tag=p["_id"]))
            neutral = msgs_at(p, "", prefix, t)
            row = {"puzzle": p["_id"], "turn": t, "old": qo, "new": qn}
            for model in COMPARATORS:
                if qo.strip().lower() == qn.strip().lower():
                    row[model] = 0.5
                    continue
                a = _pick2(model, neutral, qo, qn, seed, p["_id"])     # new question is option 2
                b = _pick2(model, neutral, qn, qo, seed, p["_id"])     # new question is option 1
                votes = [v for v in ((a == 2) if a else None, (b == 1) if b else None) if v is not None]
                row[model] = mean(votes) if votes else 0.5
            rows.append(row)
        return rows
    rows = [r for rs in pmap(one, puzzles) for r in rs]
    return {m: mean(r[m] for r in rows) for m in COMPARATORS}, rows


# ----------------------------------------------------------------------------- reflector

SCHEMA = """Return ONLY a JSON object:
{"analysis": "<=120 words on what distinguishes questions that found new key points",
 "ops": [ {"op": "ADD", "when": "<observable dialogue condition>", "do": "<concrete questioning action>"},
          {"op": "MODIFY", "id": "R3", "when": "...", "do": "..."},
          {"op": "DELETE", "id": "R5"} ]}
At most 3 ops. Each rule must be GENERAL (no names, objects or events specific to one puzzle), state-conditioned
(WHEN refers to things visible in the dialogue: turn number, recent answers, what is already confirmed), and
actionable for a small model writing ONE yes/no question per turn. Never tell the player to ask for the answer,
to ask compound or 'or' questions, or to state guesses as facts."""

INTRO = ("You improve a short questioning playbook for a small frozen language model that plays situation puzzles "
         "(lateral-thinking riddles): it sees a short surface story, asks 25 yes/no questions to a referee who knows "
         "the hidden story, then writes the hidden story. Quality of questioning is measured by how many of the "
         "puzzle's hidden key points its questions uncover.\n\nCurrent playbook:\n")


def reflect(mech, rules, experience, round_i, seed):
    base = base_of(mech)
    if base in ("cce", "cce_shuf"):
        task = ("Below are comparison groups. In each group, several candidate questions were proposed from the SAME "
                "dialogue state; each shows how many NEW key points it would uncover and the referee's answer. "
                "Infer which kinds of questions uncover key points in which situations, and edit the playbook.")
    elif base == "traj":
        task = ("Below, for each puzzle, the best and the worst of several full question sequences (by fraction of key "
                "points uncovered) are shown. Infer what the better sequences do differently and edit the playbook.")
    elif base == "bo8":
        task = ("Below are decision points from the small model's own games. At each, the model wrote 8 candidate "
                "questions from the SAME dialogue state, then compared them side by side and picked the one it judged "
                "most revealing (picked=true); the referee's answer to every candidate is shown. When the model "
                "compares candidates it asks better questions than when it writes a single question. Infer what the "
                "picked questions have in common versus the unpicked ones, in which situations, and edit the playbook "
                "so that a SINGLE question written by the model looks like the one it would have picked.")
    else:
        task = ("Below are training puzzles with their hidden stories, their key points, the key points the player "
                "missed and the player's questions. Infer which kinds of hypotheses and aspects the player should "
                "probe to uncover hidden key points in puzzles like these, and edit the playbook.")
    exp = json.dumps(experience, ensure_ascii=False)
    if len(exp) > 120_000:
        exp = exp[:120_000] + " ...(truncated)"
    budget = (f"\n\nLength budget: the playbook now has {words(rules)} words and {len(rules)} rules; it must stay under "
              f"{MAX_WORDS} words and {MAX_RULES} rules after your edits. Keep each rule under 45 words; prefer MODIFY "
              f"or DELETE over ADD when the budget is tight. An edit that breaks the budget is discarded.")
    prompt = (INTRO + (render(rules) or "(empty)") + f"\n\n{task}\n\n" + SCHEMA + budget +
              f"\n\nRound {round_i}. Experience:\n{exp}")
    txt = chat(REFLECTOR, [{"role": "user", "content": prompt}], role="reflector", temperature=0.7,
               max_tokens=12000, seed=seed * 10 + round_i, reasoning=True, tag=f"{mech}-s{seed}-r{round_i}")
    m = re.search(r"\{.*\}", txt or "", re.S)
    try:
        obj = json.loads(m.group(0)) if m else {}
    except Exception:
        obj = {}
    return obj, txt


_BOTTOM_NGRAMS = None


def specific_ngram(text, n=5):
    """True if the rule copies a 5-word span from any TRAIN hidden story (puzzle-specific content)."""
    global _BOTTOM_NGRAMS
    if _BOTTOM_NGRAMS is None:
        tr, _ = sp.split_train(sp.load("train"))
        _BOTTOM_NGRAMS = set()
        for p in tr:
            w = re.findall(r"[a-z']+", p["bottom"].lower())
            _BOTTOM_NGRAMS |= {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}
    w = re.findall(r"[a-z']+", text.lower())
    return any(" ".join(w[i:i + n]) in _BOTTOM_NGRAMS for i in range(len(w) - n + 1))


def apply_ops(rules, ops, round_i):
    rules = [dict(r) for r in rules]
    log = []
    nxt = 1 + max([int(r["id"][1:]) for r in rules] or [0])
    for op in (ops or [])[:3]:
        kind = str(op.get("op", "")).upper()
        before_op = [dict(r) for r in rules]
        if kind in ("ADD", "MODIFY"):
            when, do = str(op.get("when", "")).strip(), str(op.get("do", "")).strip()
            if not when or not do or specific_ngram(when + " " + do):
                log.append({"op": op, "status": "rejected_invalid_or_specific"})
                continue
        if kind == "ADD" and len(rules) < MAX_RULES:
            rules.append({"id": f"R{nxt}", "when": when, "do": do, "round": round_i})
            nxt += 1
            log.append({"op": op, "status": "applied"})
        elif kind == "MODIFY":
            hit = [r for r in rules if r["id"] == op.get("id")]
            if hit:
                hit[0].update(when=when, do=do, round=round_i)
                log.append({"op": op, "status": "applied"})
            else:
                log.append({"op": op, "status": "rejected_unknown_id"})
        elif kind == "DELETE":
            before = len(rules)
            rules = [r for r in rules if r["id"] != op.get("id")]
            log.append({"op": op, "status": "applied" if len(rules) < before else "rejected_unknown_id"})
        else:
            log.append({"op": op, "status": "rejected"})
        if words(rules) > MAX_WORDS:   # revert only the op that broke the length budget
            rules = before_op
            log[-1] = {"op": op, "status": "rejected_too_long", "words_if_applied": words(rules)}
    return rules, log


# ----------------------------------------------------------------------------- loop

def run(mech, seed, rounds=ROUNDS):
    out = OUT / f"{mech}_s{seed}"
    out.mkdir(parents=True, exist_ok=True)
    tr, va = splits()
    rng = random.Random(1000 + seed)
    order = tr[:]
    rng.shuffle(order)
    rules = []
    # Gate per mechanism: bo8 = none (judge-free, every valid edit kept); bo8g = judge-free pairwise preference test on
    # VAL[60:75] states; traj_cons = consensus-judged VAL gate on 150 TRAIN puzzles disjoint from its experience
    # batches; all others = the usual key-point-judged VAL[:60] gate.
    gate = {"bo8": "none", "bo8g": "pref"}.get(mech, "val")
    if mech == "traj_cons":
        va = order[rounds * TRAIN_PER_ROUND: rounds * TRAIN_PER_ROUND + 150]
    gate_puzzles = sp.split_train(sp.load("train"))[1][GATE_PUZZLES]
    gated = gate == "val"
    inc_val = eval_playbook(rules, va, seed=seed) if gated else None
    history = [{"round": 0, "rules": rules, "accepted": True, "gate": gate, "val_n": len(va) if gated else 0,
                "val_recall": mean(x["recall"] for x in inc_val) if gated else None}]
    for r in range(1, rounds + 1):
        batch = order[(r - 1) * TRAIN_PER_ROUND: r * TRAIN_PER_ROUND]
        base = base_of(mech)
        if base in ("cce", "cce_shuf"):
            exp = collect_cce(rules, batch, seed=seed * 10 + r, shuffle=(base == "cce_shuf"))
        elif base == "traj":
            exp = collect_traj(rules, batch, seed=seed * 10 + r)
        elif base == "bo8":
            exp = collect_bo8(rules, batch, seed=seed * 10 + r)
        else:
            exp = collect_prior(rules, batch, seed=seed * 10 + r)
        obj, raw = reflect(mech, rules, exp, r, seed)
        cand, oplog = apply_ops(rules, obj.get("ops"), r)
        rec = {"round": r, "batch": [p["_id"] for p in batch], "analysis": obj.get("analysis"), "ops": oplog,
               "incumbent": rules}
        if cand is None or cand == rules:
            rec.update(accepted=False, reason="no valid change")
        elif gate == "none":
            rules = cand
            rec.update(candidate=cand, accepted=True, reason="ungated (judge-free mechanism)")
        elif gate == "pref":
            rates, rows = pref_gate(rules, cand, gate_puzzles, seed=seed * 10 + r)
            ok = mean(rates.values()) >= PREF_MARGIN and min(rates.values()) >= 0.5
            rec.update(candidate=cand, pref_win_rates=rates, accepted=ok)
            (out / f"pref_gate_r{r}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
            if ok:
                rules = cand
        else:
            cand_val = eval_playbook(cand, va, seed=seed)
            inc = {x["id"]: x["recall"] for x in inc_val}
            deltas = [x["recall"] - inc[x["id"]] for x in cand_val if x["id"] in inc]
            d = mean(deltas)
            rec.update(candidate=cand, val_delta=d, val_recall_candidate=mean(x["recall"] for x in cand_val),
                       accepted=d > MIN_GAIN)
            if d > MIN_GAIN:
                rules, inc_val = cand, cand_val
        rec["val_recall_incumbent"] = mean(x["recall"] for x in inc_val) if gated else None
        rec["spent_usd_ledger"] = spent_usd()
        history.append(rec)
        (out / f"experience_r{r}.json").write_text(json.dumps(exp, ensure_ascii=False, indent=1), encoding="utf-8")
        (out / f"reflector_r{r}.txt").write_text(raw or "", encoding="utf-8")
        (out / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[{mech} s{seed}] round {r}: accepted={rec['accepted']} "
              f"val_incumbent={rec['val_recall_incumbent']} delta={rec.get('val_delta')} "
              f"rules={len(rules)} spent=${spent_usd():.2f}", flush=True)
    final = {"mech": mech, "seed": seed, "rules": rules, "playbook": render(rules),
             "sha": hashlib.sha256(render(rules).encode()).hexdigest()[:12]}
    (out / "final_playbook.json").write_text(json.dumps(final, ensure_ascii=False, indent=1), encoding="utf-8")
    return final


if __name__ == "__main__":
    if sys.argv[1].endswith("_gem"):
        EVOLVE_JUDGE = "gemini"
    elif sys.argv[1].endswith("_cons"):
        EVOLVE_JUDGE = "consensus"
    run(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else ROUNDS)
