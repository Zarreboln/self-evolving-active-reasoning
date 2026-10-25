"""Score Situation Puzzles the way BALAR does: bidirectional semantic equivalence.

AR-Bench scores SP with character-level F1 against the hidden bottom. Experiment 11 measured what
that costs -- F1 correlates 0.806 with answer length, and a prediction naming the wrong cause of
death scored 0.637 -- so BALAR does not use it. Its paper reports "non-strict semantic equivalence
[Kuhn et al., 2023]", implemented in `balar_official/utils.py` as two entailment judgments:

    equiv_strict    = both directions are entailment
    equiv_nonstrict = neither direction is contradiction, and not both are neutral   <- reported

This re-scores finished runs rather than re-running them: AR-Bench's SP records already carry the
prediction and the gold bottom, so the metric is a property of the output, not of the rollout. The
character F1 is kept in the same table, never replaced, so the two can be compared directly.

The judge. BALAR passes `agent_config.agent_model_config`, i.e. the agent grades its own answer.
`--judge policy` reproduces that. `--judge jev` instead routes to the referee shim, whose shape is
exactly this one -- a three-way typed decision with a calibrated confidence -- and whose agreement
with the 405B is measured in experiments/02. Which judge is used is recorded per episode, because a
self-graded number and an independently graded one are not the same measurement.

Two measured properties of the Qwen2.5-7B judge that the column has to carry. It is unstable at
BALAR's temperature 0.1 -- asked twelve times whether an explanation entails itself it answered
entailment six times and neutral six -- so this runs at temperature 0, where the same question is
answered entailment ten times out of ten. And it never returns `contradiction`: zero in 36 samples,
including a pair taken from two different puzzles. BALAR's non-strict rule is "no contradiction and
not both neutral", so with this judge it collapses to "at least one direction entails". The metric
still separates identical from unrelated, but that collapse belongs in any caption that uses it.

    python sp_semantic_equivalence.py results/sp_few_shot.json --judge policy
    python sp_semantic_equivalence.py results/sp_*.json --judge policy --out results/sp_equiv.json
"""

import argparse
import json
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

# Verbatim from balar_official/utils.py (MIT), so the judgment asked for is the one BALAR asks for.
EQUIVALENCE_SYSTEM_PROMPT = "You are a helpful evaluator"
EQUIVALENCE_USER_PROMPT = """We are evaluating answers to the question "{question}"
Here are two possible answers:
Possible Answer 1: {text1}
Possible Answer 2: {text2}
Does Possible Answer 1 semantically entail Possible Answer 2? Respond with entailment, contradiction, or neutral."""

SCHEMA = {"type": "object",
          "properties": {"thought": {"type": "string"},
                         "evaluation": {"type": "string",
                                        "enum": ["entailment", "contradiction", "neutral"]}},
          "required": ["thought", "evaluation"]}

EQUIV_MAP = {"entailment": 2, "neutral": 1, "contradiction": 0}


def ask_policy(base_url, api_key, model, user_prompt, temperature=0.0, retries=4):
    body = {"model": model, "temperature": temperature, "top_p": 1.0, "max_tokens": 512,
            "messages": [{"role": "system", "content": EQUIVALENCE_SYSTEM_PROMPT},
                         {"role": "user", "content": user_prompt}],
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "EvaluationScheme", "schema": SCHEMA}}}
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", method="POST",
                                 data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {api_key}",
                                          "Content-Type": "application/json"})
    last = None
    for _ in range(retries):
        try:
            d = json.load(urllib.request.urlopen(req, timeout=120))
            parsed = json.loads(d["choices"][0]["message"]["content"])
            if parsed.get("evaluation") in EQUIV_MAP:
                return parsed["evaluation"], parsed.get("thought", "")
        except Exception as e:
            last = e
    raise RuntimeError(f"judge failed: {last}")


def implication(judge, a, b, question):
    """Does a entail b, in the context of the puzzle? Returns one of the three labels."""
    prompt = EQUIVALENCE_USER_PROMPT.format(question=question, text1=a, text2=b)
    return judge(prompt)


def score_episode(rec, judge):
    pred = (rec.get("pred") or "").strip()
    gold = (rec.get("answer") or "").strip()
    # BALAR's context for SP is "Question: Explain the following puzzle: <surface>"
    surface = rec.get("question") or rec.get("surface") or ""
    question = f"Explain the following puzzle: {surface}"
    if not pred or not gold:
        return {"equiv_strict": False, "equiv_nonstrict": False, "skipped": "empty prediction"}
    l2r, t1 = implication(judge, pred, gold, question)
    r2l, t2 = implication(judge, gold, pred, question)
    a, b = EQUIV_MAP[l2r], EQUIV_MAP[r2l]
    return {"equiv_strict": (a == 2) and (b == 2),
            "equiv_nonstrict": (0 not in (a, b)) and ([a, b] != [1, 1]),
            "left2right": l2r, "right2left": r2l,
            "left2rightImplication": t1, "right2leftImplication": t2}


def judge_is_usable(judge, sample):
    """Refuse to score with a judge that cannot call a text equivalent to itself.

    Measured on Qwen2.5-7B-Instruct: given two identical explanations it reasons correctly --
    "Both possible answers describe the same scenario with identical details" -- and then labels
    the pair `neutral`, five times out of six across diagnostic cases. Under BALAR's rule
    (not both neutral) a neutral-biased judge marks nearly everything inequivalent, so the column
    would read as near-zero regardless of answer quality. That is a broken measurement, not a hard
    task, and it is cheap to detect: ask whether a text entails itself.
    """
    q = f"Explain the following puzzle: {sample.get('question', '')}"
    gold = (sample.get("answer") or "").strip()
    if not gold:
        return True, "no gold text to check with"
    labels = [implication(judge, gold, gold, q)[0] for _ in range(3)]
    ok = all(EQUIV_MAP[x] == 2 for x in labels)
    return ok, f"identical texts judged {' / '.join(labels)} over three tries"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--judge", choices=["policy", "jev"], default="policy",
                    help="policy reproduces BALAR (the agent grades itself); jev grades independently")
    ap.add_argument("--base-url", default=os.environ.get("POLICY_BASE_URL"))
    ap.add_argument("--api-key", default=os.environ.get("POLICY_API_KEY", "x"))
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0 by default, not BALAR's 0.1: measured on Qwen2.5-7B, identical texts "
                         "are judged entailment 10/10 at 0 but 6/12 and 8/10 on two runs at 0.1")
    ap.add_argument("--out", default=None, help="write the per-episode judgments here")
    ap.add_argument("--force", action="store_true",
                    help="score even if the judge fails its self-check; the result must then be "
                         "labelled as measured with a judge known to be biased")
    a = ap.parse_args()

    if a.judge == "jev":
        sys.exit("--judge jev is not wired yet: the shim answers AR-Bench referee turns, not "
                 "entailment pairs. Add an entailment question type to it first.")
    if not a.base_url:
        sys.exit("set POLICY_BASE_URL or pass --base-url")

    def judge(prompt):
        return ask_policy(a.base_url, a.api_key, a.model, prompt, a.temperature)

    first = None
    for path in a.files:
        try:
            rows = json.load(open(path))
        except Exception:
            continue
        if rows:
            first = rows[0]
            break
    if first is not None:
        ok, why = judge_is_usable(judge, first)
        print(f"judge self-check ({a.model}): {'PASS' if ok else 'FAIL'} -- {why}")
        if not ok and not a.force:
            sys.exit("refusing to score: this judge does not call a text equivalent to itself, so "
                     "the column would measure the judge's neutral bias rather than the answers. "
                     "Use a stronger judge, or --force to score anyway and label the result.")

    all_rows = {}
    print("%-30s %10s %14s %14s" % ("run", "episodes", "equiv_nonstrict", "equiv_strict"))
    for path in a.files:
        try:
            d = json.load(open(path))
        except Exception:
            print("%-30s unreadable" % os.path.basename(path))
            continue
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            rows = list(ex.map(lambda r: score_episode(r, judge), d))
        ns = sum(1 for r in rows if r["equiv_nonstrict"])
        st = sum(1 for r in rows if r["equiv_strict"])
        f1 = [r.get("f1_score_char") for r in d if r.get("f1_score_char") is not None]
        print("%-30s %10d %9d/%-4d %9d/%-4d   (char F1 %.3f)"
              % (os.path.basename(path), len(d), ns, len(d), st, len(d),
                 sum(f1) / len(f1) if f1 else float("nan")))
        all_rows[os.path.basename(path)] = [{**r, "idx": rec.get("idx")} for r, rec in zip(rows, d)]
    if a.out:
        json.dump({"judge": a.judge, "model": a.model, "runs": all_rows}, open(a.out, "w"), indent=2)
        print(f"[saved] {a.out}")


if __name__ == "__main__":
    main()
