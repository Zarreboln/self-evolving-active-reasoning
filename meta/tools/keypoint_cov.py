"""Both abilities' first column, re-measured: what the questions were worth, per task.

The proxy this replaces counted distinct suspects interrogated, out of five. On a 25-turn budget
that saturates: the seed harness is at 100 by turn 5 and the evolved one by turn 12, so the number
the report quotes at turn 25 is 100 against 100 and cannot move. It is also blind to content --
five pointless questions, one per suspect, score full marks.

AR-Bench ships the thing the proxy was standing in for. Every case carries six `key_question`
entries (100 of 100 cases: three on the murderer, one each on three others, none on the fifth),
and `dc_evaluator.py` judges each question against the keypoints of the suspect it was put to,
accumulating `matched_keyquestion_set` and `question_hit_cnt`. Both are reported here, because
they are different quantities and only one of them is what the report's IA_t defines.

COVERAGE, |matched_keyquestion_set| / 6, is the faithful analogue: IA_t = E[1/|H_t|] is a state --
how much is known -- and asking one good question twice does not raise it. Its risk is the risk
the suspect proxy already has: six keypoints in twenty-five turns may saturate, and then the
turn-25 number is a ceiling again rather than a measurement.

HIT RATE, question_hit_cnt / questions asked, is how good the questions were. It has no ceiling
in the same place: covering all six in ten questions and covering all six in twenty-five are 60
and 24. It is not what IA_t defines, so reporting it means saying so.

One judge call yields both.

The judging rule is theirs, verbatim from `dc/prompt.py`: a question hits a keypoint when it can
DIRECTLY LEAD to it. The judge is not: AR-Bench sends this to `response_model`, which in our setup
is the frozen 7B that also plays the suspects and is the policy under test. Measuring that model
with itself cannot separate "the harness did not improve" from "the 7B cannot tell". So the judge
is Jev, as it is for the SP referee, and this is a substitution -- comparable across harnesses run
under it, not to AR-Bench's published numbers.

Jev takes several typed questions against one state, so a turn costs one call carrying that
suspect's keypoints, not one call per keypoint.

Coverage is monotone inside an episode, so the whole curve comes from the 25-turn run by taking
prefixes. No per-budget re-runs, and both curves are read off the transcripts already recorded.

    python meta/tools/dc_keypoint_cov.py dc_template_zero_shot_b25.json dc_lean_verify_b25.json \
        --cases AR-Bench/data/dc/test.json --out dc_cov.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# TypeSafe's own endpoint. A resold key (jevgo.cn and the like) speaks the same protocol at its
# own host, so the address is configurable rather than written in: the relay sees every state we
# send, which for a published benchmark is a routing choice, not a disclosure.
JEV_API = os.environ.get("JEV_API", "https://api.typesafe.ai/v1/systemone")
TURNS = [1, 2, 3, 5, 8, 12, 16, 20, 25]

# dc/prompt.py: "If the player's questions can directly leads to the key points, you can say that
# the question hits the key points". One keypoint, one typed question.
INSTRUCTIONS_SP = (
    "You are the referee of a situation-puzzle game. The player is shown the <Surface>; you are "
    "given the <Bottom>. Understand the whole story from both, then judge whether the player's "
    "question hits the keypoint: it hits when the question can directly lead to the keypoint, and "
    "misses otherwise."
)
INSTRUCTIONS = (
    "You are the referee of a detective game. The player interrogates a suspect, knowing only the "
    "case's initial information; you are given that suspect's private story. Judge whether the "
    "player's question hits the keypoint: it hits when the question can directly lead to the "
    "keypoint, and misses otherwise."
)


def api_key():
    k = os.environ.get("TYPESAFE_API_KEY")
    if k:
        return k.strip()
    return subprocess.run(["security", "find-generic-password", "-s", "typesafe_token", "-w"],
                          capture_output=True, text=True, check=True).stdout.strip()


def judge(state, keypoints, key, model, retries=4):
    """-> (set of hit indices, input tokens). `keypoints` is {index: text}."""
    instr = INSTRUCTIONS_SP if "Surface" in state else INSTRUCTIONS
    qs = {f"k{i}": {"type": "choice", "instructions": instr,
                    "criteria": {"Hit": f"The question can directly lead to: {t}",
                                 "Miss": "The question cannot directly lead to that keypoint."}}
          for i, t in keypoints.items()}
    body = {"model": model, "state": state, "questions": qs}
    req = urllib.request.Request(JEV_API, method="POST", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {key}",
                                          "Content-Type": "application/json"})
    last = None
    for a in range(retries):
        try:
            d = json.load(urllib.request.urlopen(req, timeout=90))
            hits = {i for i in keypoints
                    if (d["answers"].get(f"k{i}") or {}).get("choice") == "Hit"}
            return hits, (d.get("usage") or {}).get("input_tokens", 0)
        except Exception as e:
            last = e
            time.sleep(1.5 * (a + 1))
    raise RuntimeError(f"Jev failed after {retries} attempts: {last}")


def keypoints_sp(case):
    """SP keeps its six `key_question` at the top of the instance, with no suspect to address, so
    one judge call per turn carries all six against the puzzle's own Surface and Bottom."""
    ks = case.get("key_question") or []
    return {None: {j + 1: t for j, t in enumerate(ks)}}, len(ks)


def keypoints_of(case):
    """{suspect name: {global index: keypoint text}} -- indices run 1..6 across the case, in the
    order dc_evaluator builds them, so the union size is the denominator."""
    out, n = {}, 0
    for s in case["suspects"]:
        ks = s.get("key_question") or []
        if ks:
            out[s["name"]] = {n + j + 1: t for j, t in enumerate(ks)}
            n += len(ks)
    return out, n


def curve_point(curves, t):
    """Mean over episodes of the value at turn t, holding the last value for a short episode:
    coverage cannot fall, the episode simply stopped asking."""
    return sum(c[min(t, len(c)) - 1] if c else 0.0 for c in curves) / len(curves)


def run_gn(path):
    """No judge: Guess Number's own feedback settles both columns from the transcript.

    COVERAGE is the posterior mass 1/|H_t| the report already uses -- exact, since the feedback is
    deterministic and the target never leaves H_t.

    HIT RATE is T3's (ICLR 2026) dead-turn criterion read the other way round: a guess is spent
    when d(H_{t-1}, H_t) = |H_{t-1}| - |H_t| <= 0, so the share of guesses that strictly cut the
    candidate set is the same quantity the keypoint judge estimates on the two text tasks -- how
    many of the questions asked were worth asking.
    """
    sys.path.insert(0, str(Path(__file__).parent))
    from ia_bt import ALL, feedback
    # An episode that asked nothing is a real episode with zero coverage, not a missing record:
    # at a one-turn budget the evolved SP harness declines to ask at all (30 of 30, round 0). The
    # two are told apart by whether the key is there, since a run written before transcripts were
    # recorded has no `history` key at all.
    rows = [r for r in json.load(open(path)) if "history" in r]
    curves = []
    for r in rows:
        H, cov, rate, useful, asked = ALL, [], [], 0, 0
        for g, a, b in (tuple(h) for h in r["history"]):
            prev = len(H)
            H = [x for x in H if feedback(x, g) == (a, b)]
            asked += 1
            useful += 1 if len(H) < prev else 0
            cov.append(100.0 / len(H) if H else 0.0)
            rate.append(100.0 * useful / asked)
        curves.append((cov or [0.0], rate or [0.0]))
    out = []
    for col in (0, 1):
        out.append([sum(c[col][min(t, len(c[col])) - 1] if c[col] else 0.0 for c in curves)
                    / len(curves) for t in TURNS])
    return out[0], out[1], len(curves), 0, 0


def run(path, cases, key, model, log, task):
    rows = [r for r in json.load(open(path)) if "history" in r]
    curves, spent, calls = [], 0, 0
    for ep, r in enumerate(rows):
        case = cases[ep]
        if task == "sp":
            kp, total = keypoints_sp(case)
            init, stories = case["surface"], {None: case["bottom"]}
        else:
            kp, total = keypoints_of(case)
            init = json.dumps(case["initial_information"])
            stories = {s["name"]: s["story"] for s in case["suspects"]}
        # Positional alignment is an assumption, so it is checked: every suspect a transcript
        # names must be one of this case's suspects, or the episode is matched to the wrong case.
        unknown = set() if task == "sp" else {h[0] for h in r["history"]} - set(stories)
        if unknown:
            raise SystemExit("episode %d does not belong to case %s: %s"
                             % (ep, case.get("index"), sorted(unknown)[:3]))
        hits, per_turn, per_turn_rate = set(), [], []
        asked = hit_q = 0
        for h in (tuple(x) for x in r["history"]):
            name, question = (None, h[0]) if task == "sp" else (h[0], h[1])
            if name in kp:
                state = ({"Surface": init, "Bottom": stories[None], "player_question": question}
                         if task == "sp" else
                         {"CaseInformation": init, "Suspect": name,
                          "SuspectStory": stories[name], "player_question": question})
                h, tin = judge(state, kp[name], key, model)
                hits |= h
                asked += 1
                hit_q += 1 if h else 0
                spent += tin
                calls += 1
                if log:
                    log.write(json.dumps({"file": Path(path).name, "episode": ep,
                                          "case": case.get("index"), "suspect": name, "q": question,
                                          "hits": sorted(h), "input_tokens": tin}) + "\n")
                    log.flush()
            per_turn.append(100.0 * len(hits) / total)
            # Questions put to the fifth suspect carry no keypoints, so they can never hit and are
            # not counted either way: the rate is over the questions the rule can actually judge.
            per_turn_rate.append(100.0 * hit_q / asked if asked else 0.0)
        curves.append((per_turn or [0.0], per_turn_rate or [0.0]))
    # mean over episodes at each requested turn; an episode shorter than t holds its last value,
    # because coverage cannot fall and the episode simply stopped asking.
    cov, rate = [], []
    for t in TURNS:
        for col, dest in ((0, cov), (1, rate)):
            vals = [c[col][min(t, len(c[col])) - 1] if c[col] else 0.0 for c in curves]
            dest.append(sum(vals) / len(vals))
    return cov, rate, len(curves), calls, spent


def sweep(prefix, root, cases, key, model, log, task):
    """One point per budget, each read off the run whose budget was that turn.

    This is how belief tracking is read -- the episode has to end at t for the system to be asked
    for its answer -- so reading information acquisition off prefixes of the 25-turn run would put
    the two rows of a panel on different runs. On Guess Number the two readings differ by up to
    7.5 points, which is large enough that the choice has to be the same for both rows.
    """
    cov, rate, ns, calls, spent = [], [], [], 0, 0
    for t in TURNS:
        path = os.path.join(root, "%s_b%d.json" % (prefix, t))
        if task == "gn":
            c, r, n, k, tin = run_gn(path)
        else:
            c, r, n, k, tin = run(path, cases, key, model, log, task)
        # the run's own last turn is this budget's point
        cov.append(c[-1])
        rate.append(r[-1])
        ns.append(n)
        calls += k
        spent += tin
    return cov, rate, min(ns), calls, spent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--cases", help="the split the runs were driven from; not needed for gn")
    ap.add_argument("--task", choices=("gn", "dc", "sp"), default="dc")
    ap.add_argument("--model", default="jev-1.14.0")
    ap.add_argument("--out")
    ap.add_argument("--log")
    ap.add_argument("--limit", type=int, help="first N episodes, for a smoke test")
    ap.add_argument("--sweep", metavar="DIR",
                    help="read each turn off <DIR>/<prefix>_b<turn>.json instead of prefixes of "
                         "one run; `runs` are then prefixes, not paths")
    a = ap.parse_args()

    raw = json.load(open(a.cases)) if a.cases else {}
    # The split file the runs were driven from; episodes are its instances in order.
    cases = raw["instances"] if isinstance(raw, dict) and "instances" in raw else raw
    key = api_key() if a.task != "gn" else None
    log = open(a.log, "a") if a.log else None
    res = {}
    for p in a.runs:
        if a.limit:
            rows = [r for r in json.load(open(p)) if r.get("history")][:a.limit]
            tmp = Path(p).with_suffix(".limit.json")
            json.dump(rows, open(tmp, "w"))
            p = str(tmp)
        if a.sweep:
            cov, rate, n, calls, tin = sweep(p, a.sweep, cases, key, a.model, log, a.task)
        elif a.task == "gn":
            cov, rate, n, calls, tin = run_gn(p)
        else:
            cov, rate, n, calls, tin = run(p, cases, key, a.model, log, a.task)
        res[Path(p).stem if not a.sweep else p] = {"turns": TURNS, "n": n, "coverage": cov, "hit_rate": rate}
        print("%-30s n=%-3d calls=%-5d $%.4f  turn25: coverage=%.1f  hit_rate=%.1f"
              % (Path(p).stem if not a.sweep else p, n, calls, tin * 0.042 / 1e6, cov[-1], rate[-1]), file=sys.stderr)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    else:
        print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
