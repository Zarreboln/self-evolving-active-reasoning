"""Two named open-source context-evolution methods on SP, re-implemented in our harness (both were on
the team's reading list). Same frozen 7B, referee, 40 TRAIN puzzles per seed (the same
batches as evolve.py), 400-word cap, 3 seeds; final playbooks go to runs/evolve/<method>_s<seed>/final_playbook.json
so evaluate.py / rescore_test.py / consensus_recall.py treat them like every other arm.

ace  ACE (Zhang et al., arXiv 2510.04618), offline adaptation, simplified: an itemized playbook of bullets with
     helpful/harmful counters. Generator = the 7B playing an episode. Reflector (Opus 5.5) sees the trajectory, the
     hidden story, the key points and the JUDGE feedback (Haiku key-point recall + missed key points), tags the
     bullets and states insights. Curator (Opus 5.5) adds <= 3 new bullets per batch (delta updates, no rewriting);
     grow-and-refine drops near-duplicates and, over budget, the bullets with the worst helpful-harmful balance.
     No VAL gate (ACE accepts deltas). Judge-DRIVEN.
dc   Dynamic Cheatsheet, cumulative variant (Suzgun et al., arXiv 2504.07952): after each batch of 10 episodes a
     curator (Opus 5.5) rewrites the cheatsheet from the trajectories alone - no hidden story, no key points, no
     judge score (test-time learning without labels). Judge-FREE.
The pair is a built-in test of the judge-exploitation hypothesis: if evolution learns the in-loop judge, ace should
show a Haiku-vs-Gemini gap and dc should not.
Usage: python -I code/named_methods.py ace|dc <seed>
"""
import difflib
import hashlib
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sp  # noqa: E402
import evolve  # noqa: E402
from llm import chat, spent_usd  # noqa: E402

OUT = evolve.OUT
REFLECTOR = evolve.REFLECTOR
MAX_WORDS = evolve.MAX_WORDS
INTRO = evolve.INTRO


def _json(txt):
    m = re.search(r"\{.*\}", txt or "", re.S)
    try:
        return json.loads(m.group(0)) if m else {}
    except Exception:
        return {}


def render_bullets(bullets):
    return "\n".join(f"- [{b['id']}] {b['text']}" for b in bullets)


def episodes(pb, batch, seed):
    def one(p):
        ep = sp.run_episode(p, playbook=pb, seed=seed)
        return p, ep
    return evolve.pmap(one, batch)


# ----------------------------------------------------------------------------- ACE

ACE_REFLECT = """You are the REFLECTOR of an agentic-context-engineering loop. A small frozen model played a situation
puzzle (asks 25 yes/no questions, then writes the hidden story) using the playbook below. Quality is the share of the
puzzle's hidden key points that its questions uncovered (judge feedback below).

Playbook:
{playbook}

Puzzle surface: {surface}
Hidden story (ground truth): {bottom}
Key points: {kps}
Judge feedback: key-point recall {recall:.2f}; key points missed: {missed}
Questions and referee answers:
{qa}

Return ONLY JSON: {{"error_analysis": "<=80 words", "key_insight": "<=40 words, general (no puzzle-specific content)",
"bullet_tags": [{{"id": "B1", "tag": "helpful|harmful|neutral"}}]}}"""

ACE_CURATE = """You are the CURATOR of an agentic-context-engineering loop. Current playbook (itemized bullets):
{playbook}

Reflections from the latest episodes:
{reflections}

Propose at most 3 NEW bullets (delta update: never rewrite or delete existing ones) that capture general, reusable
questioning strategies for a small model writing ONE yes/no question per turn. No puzzle-specific names, objects or
events; no compound or 'or' questions; never ask for the answer. Skip anything already covered.
Return ONLY JSON: {{"new_bullets": ["...", "..."]}}"""


def run_ace(seed, rounds=evolve.ROUNDS):
    out = OUT / f"ace_s{seed}"
    out.mkdir(parents=True, exist_ok=True)
    tr, _ = evolve.splits()
    order = tr[:]
    random.Random(1000 + seed).shuffle(order)
    bullets, nxt, log = [], 1, []
    for r in range(1, rounds + 1):
        batch = order[(r - 1) * evolve.TRAIN_PER_ROUND: r * evolve.TRAIN_PER_ROUND]
        pb = render_bullets(bullets)
        eps = episodes(pb, batch, seed * 10 + r)

        def reflect(pe):
            p, ep = pe
            rec, seen = evolve.recall_of(p, ep["history"])
            kps = sp.keypoints_of(p)
            prompt = ACE_REFLECT.format(playbook=pb or "(empty)", surface=p["surface"], bottom=p["bottom"],
                                        kps="; ".join(kps), recall=rec,
                                        missed="; ".join(k for i, k in enumerate(kps, 1) if i not in seen) or "none",
                                        qa="\n".join(f"Q: {sp.clean_q(q)} -> {a}" for q, a in ep["history"]))
            txt = chat(REFLECTOR, [{"role": "user", "content": prompt}], role="reflector", temperature=0.7,
                       max_tokens=6000, reasoning=True, seed=seed * 10 + r, tag=f"ace-s{seed}-r{r}-{p['_id']}")
            return _json(txt) | {"recall": rec}
        refl = evolve.pmap(reflect, eps)
        for x in refl:
            for t in x.get("bullet_tags") or []:
                for b in bullets:
                    if b["id"] == t.get("id") and t.get("tag") in ("helpful", "harmful"):
                        b[t["tag"]] += 1
        txt = chat(REFLECTOR, [{"role": "user", "content": ACE_CURATE.format(
            playbook=pb or "(empty)",
            reflections=json.dumps([{k: x.get(k) for k in ("error_analysis", "key_insight", "recall")} for x in refl],
                                   ensure_ascii=False))}],
            role="reflector", temperature=0.7, max_tokens=6000, reasoning=True, seed=seed * 10 + r,
            tag=f"ace-s{seed}-r{r}-curate")
        added = []
        for text in (_json(txt).get("new_bullets") or [])[:3]:
            text = str(text).strip()
            if not text or evolve.specific_ngram(text):
                continue
            if any(difflib.SequenceMatcher(None, text.lower(), b["text"].lower()).ratio() > 0.8 for b in bullets):
                continue          # grow-and-refine: drop near-duplicates
            bullets.append({"id": f"B{nxt}", "text": text, "helpful": 0, "harmful": 0, "round": r})
            added.append(text)
            nxt += 1
        while len(render_bullets(bullets).split()) > MAX_WORDS and bullets:   # refine: worst balance goes first
            worst = min(bullets, key=lambda b: (b["helpful"] - b["harmful"], -b["round"]))
            bullets.remove(worst)
        log.append({"round": r, "batch": [p["_id"] for p, _ in eps], "mean_recall_haiku": sum(x["recall"] for x in refl)
                    / len(refl), "added": added, "bullets": [dict(b) for b in bullets], "spent": spent_usd()})
        (out / "history.json").write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[ace s{seed}] round {r}: +{len(added)} bullets, {len(bullets)} total, spent ${spent_usd():.2f}",
              flush=True)
    pb = render_bullets(bullets)
    (out / "final_playbook.json").write_text(json.dumps(
        {"mech": "ace", "seed": seed, "bullets": bullets, "playbook": pb,
         "sha": hashlib.sha256(pb.encode()).hexdigest()[:12]}, ensure_ascii=False, indent=1), encoding="utf-8")


# ----------------------------------------------------------------------------- Dynamic Cheatsheet (cumulative)

DC_CURATE = """You maintain a CHEATSHEET for a small frozen model that plays situation puzzles (lateral-thinking
riddles): it sees a short surface story, asks 25 yes/no questions to a referee who knows the hidden story, then writes
the hidden story. Nobody tells you the true stories or any score; learn only from how the games went.

Current cheatsheet:
{sheet}

Games just played with this cheatsheet (questions, referee answers, final story):
{games}

Rewrite the cheatsheet: keep what still looks useful, add general strategies and insights suggested by these games
(which kinds of questions got informative Yes/No answers, which wasted turns on Unknown), remove what looks useless.
No puzzle-specific names, objects or events. At most {max_words} words, as short bullet points.
Return ONLY the cheatsheet text."""


def run_dc(seed, rounds=evolve.ROUNDS):
    out = OUT / f"dc_s{seed}"
    out.mkdir(parents=True, exist_ok=True)
    tr, _ = evolve.splits()
    order = tr[:]
    random.Random(1000 + seed).shuffle(order)
    sheet, log = "", []
    for r in range(1, rounds + 1):
        batch = order[(r - 1) * evolve.TRAIN_PER_ROUND: r * evolve.TRAIN_PER_ROUND]
        eps = episodes(sheet, batch, seed * 10 + r)
        games = "\n\n".join(f"Game {i + 1}. Surface: {p['surface']}\n" +
                            "\n".join(f"Q: {sp.clean_q(q)} -> {a}" for q, a in ep["history"]) +
                            f"\nFinal story: {ep['pred'][:600]}" for i, (p, ep) in enumerate(eps))
        txt = chat(REFLECTOR, [{"role": "user", "content": DC_CURATE.format(sheet=sheet or "(empty)", games=games,
                                                                         max_words=MAX_WORDS)}],
                   role="reflector", temperature=0.7, max_tokens=8000, reasoning=True, seed=seed * 10 + r,
                   tag=f"dc-s{seed}-r{r}")
        new = (txt or "").strip()
        if new and len(new.split()) <= MAX_WORDS + 40 and not evolve.specific_ngram(new, n=8):   # 8-grams: a full rewrite trips the 5-gram filter on generic phrases
            sheet = new
            status = "updated"
        else:
            status = "rejected (empty, too long or puzzle-specific)"
        log.append({"round": r, "batch": [p["_id"] for p, _ in eps], "status": status, "sheet": sheet,
                    "spent": spent_usd()})
        (out / "history.json").write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[dc s{seed}] round {r}: {status}, {len(sheet.split())} words, spent ${spent_usd():.2f}", flush=True)
    (out / "final_playbook.json").write_text(json.dumps(
        {"mech": "dc", "seed": seed, "playbook": sheet, "sha": hashlib.sha256(sheet.encode()).hexdigest()[:12]},
        ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    {"ace": run_ace, "dc": run_dc}[sys.argv[1]](int(sys.argv[2]))
