"""Verification of every adapter beyond "the oracle wins". CPU only; LLM roles are a fixed stub.

  random      R episodes per subtask of a random player whose actions are well-formed. Every score
              must be finite and in range, and the random player must do far worse than the oracle.
  partial     Hand-built wrong and partly-right endings per bench, each checked against the value
              the bench's own rule gives.
  leak        During oracle and random episodes, the secret must not appear in the instructions or
              in any observation before the final one (games reveal the answer when they end).
  concurrent  Every subtask's first search instance played K times at once, interleaved with all
              the other subtasks, with the same seeded random actions; all K transcripts must match.

    source mb/env.sh && python -m mb.verify [--benches ...] [--checks random,partial,leak,concurrent]
Prints one line per check and a summary; writes the details to --out.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import string
import traceback
from concurrent.futures import ThreadPoolExecutor

from mb.benches import BENCHES, get
from mb.smoke import oracle_episode, stub_responder

R_EPISODES = 5
K_COPIES = 3


# ------------------------------------------------------------------ random players (test-only)
def random_action(env, rng):
    b = env.bench
    adm = env.admissible()
    if adm:
        return rng.choice(adm)
    if b == "wordle":
        from mb.benches.wordle import _words
        return f"[{rng.choice(_words(env.subtask))}]"
    if b == "twentyq":
        from mb.benches.twentyq import SUBTASKS, _wordlist
        if rng.random() < 0.8:
            return rng.choice(["Is it alive?", "Is it bigger than a car?", "Is it in Europe?", "Is it famous?"])
        return f"[{rng.choice(_wordlist(*SUBTASKS[env.subtask]))}]"
    if b == "gn":
        return "".join(rng.sample("0123456789", 4))
    if b == "dc":
        if rng.random() < 0.85:
            return f"ask {rng.choice(env._env.suspects)}: Where were you that night?"
        return f"accuse {rng.choice('ABCDE')}"
    if b == "sp":
        return ("ask: Did someone die?" if rng.random() < 0.85
                else "answer: " + " ".join(rng.choice(["he", "was", "dead", "a", "letter", "twin"]) for _ in range(12)))
    if b == "mastermind":
        e = env._env
        return "[" + " ".join(map(str, rng.sample(range(1, e.num_numbers + 1), e.code_length))) + "]"
    if b == "hangman":
        return f"[{rng.choice(string.ascii_uppercase)}]"
    if b == "minesweeper":
        e = env._env
        return f"[{rng.randrange(e.rows)} {rng.randrange(e.cols)}]"
    if b == "guessnumber":
        e = env._env
        return f"[{rng.randint(e.min_number, e.max_number)}]"
    if b == "guesswho":
        if rng.random() < 0.7:
            return rng.choice(["Is the character male?", "Does the character wear glasses?", "Is the hair brown?"])
        return f"[{rng.choice(env._env.characters)['name']}]"
    if b == "circuit":
        if env._queries < 18:
            return f"<query>{rng.choice('ABC')}({rng.randint(0, 1)}, {rng.randint(0, 1)}, {rng.randint(0, 1)})</query>"
        return "<guess>" + "".join(rng.choice("01") for _ in range(24)) + "</guess>"
    if b == "movierec":
        if env._asked < 10:
            a, c = rng.sample(list(env._seen), 2)
            return f"<question>Would you prefer watching {a} over {c}?</question>"
        return f"<final_answer>{rng.choice(env._unseen_names)}</final_answer>"
    if b == "boxinggym":
        if env._experiments < 10:
            return env._oracle_next()          # a random valid design, drawn from the env's own sampler
        return f"<answer>{rng.choice([0, 1, 2.5, 7, 13.0])}</answer>"
    if b == "blicket":
        if rng.random() < 0.85:
            return "test: " + (", ".join(rng.sample(env._objs, rng.randint(1, env._n))))
        return (f"answer: blickets = {', '.join(rng.sample(env._objs, rng.randint(1, env._n)))}; "
                f"rule = {rng.choice(['conjunctive', 'disjunctive'])}")
    raise ValueError(f"no random player for {b}")


def play(env, rng, cap=2000):
    n = 0
    while not env.done and n < cap:
        env.act(random_action(env, rng))
        n += 1
    return env.result()


# ------------------------------------------------------------------ secrets for the leak check
def secrets(env):
    b = env.bench
    if b in ("wordle", "twentyq", "hangman"):
        s = env._instance if b != "hangman" else env._env.state.game_state["target_word"]
        return [s.lower()]
    if b == "mastermind":
        return [" ".join(map(str, env._env.state.game_state["secret_code"]))]
    if b == "circuit":
        return ["".join(map(str, env._truth))]
    if b == "movierec":
        return [str(list(env._w))]
    if b == "sp":
        return [env._env._bottom[:60].lower()]
    return []


def leaked(env):
    hits = []
    texts = [env.instructions] + [o for _, o in env.transcript[:-1]]
    for s in secrets(env):
        for t in texts:
            if s and s in (t or "").lower():
                hits.append(s)
                break
    return hits


# ------------------------------------------------------------------ partial-credit cases
def _close(a, b):
    return a is not None and b is not None and abs(a - b) < 1e-9


def partial_cases(name):
    """[(label, ok, detail)] for one bench."""
    out = []
    b = get(name)
    sub = b.subtasks[0]
    inst = b.instances(sub, "search", limit=1)[0]
    mk = lambda: b.make(sub, inst, stub_responder if b.responder else None)

    if name == "wordle":
        e = mk()
        from mb.benches.wordle import _words
        wrong = [w for w in _words(sub) if w != e._instance][:6]
        for w in wrong:
            if not e.done:
                e.act(f"[{w}]")
        fb = e._env.state.game_state["guess_history"][-1][1]
        exp = (fb.count("G") + 0.5 * fb.count("Y")) / len(fb)
        out.append(("six wrong guesses -> completion of the last", e.done and not e.success() and _close(e.score(), exp), (e.score(), exp)))
        e = mk(); e.act("no brackets"); e.act("still none")
        out.append(("two invalid moves -> game over", e.done and not e.success(), e.result()))
    elif name == "twentyq":
        e = mk()
        e.act("[zzzz wrong]")
        out.append(("wrong final guess -> 0", e.done and e.score() == 0.0, e.result()))
    elif name == "guesswho":
        e = mk()
        e.act("[Nobody]")
        first = e.done
        e.act("[Noone]")
        out.append(("a wrong name is an invalid move; the second ends it at raw -1, score 0",
                    not first and e.done and e.score() == 0.0 and e.raw_reward == -1.0, e.result()))
    elif name == "gn":
        e = mk()
        t = "".join(map(str, e._env._target))
        wrong = [g for g in ("0123", "4567", "8901", "2345", "6789") if g != t]
        while not e.done:
            e.act(wrong[e.steps % len(wrong)])
        out.append(("25 wrong guesses -> 0", e.score() == 0.0 and not e.success(), e.result()))
    elif name == "dc":
        e = mk()
        wrong = "ABCDE"[(e._env._label + 1) % 5]
        e.act(f"accuse {wrong}")
        out.append(("wrong accusation -> 0", e.done and e.score() == 0.0, e.result()))
        e = mk()
        while not e.done:
            e.act("ask nobody: hello")
        out.append(("never accusing -> 0", e.score() == 0.0, e.result()))
    elif name == "sp":
        e = mk()
        half = e._env._bottom[: len(e._env._bottom) // 2]
        e.act("answer: " + half)
        out.append(("half the bottom -> 0 < F1 < 1", 0 < e.score() < 1, e.score()))
        e = mk(); e.act("answer: xyz")
        out.append(("unrelated answer -> F1 near 0", e.score() < 0.3, e.score()))
    elif name == "mastermind":
        e = mk()
        code = e._env.state.game_state["secret_code"]
        rng = random.Random(0)
        while not e.done:
            g = rng.sample(range(1, e._env.num_numbers + 1), e._env.code_length)
            if g != code:
                e.act("[" + " ".join(map(str, g)) + "]")
        out.append(("never cracking -> 0 <= reward < 1", e.done and not e.success() and 0 <= e.score() < 1, e.score()))
    elif name == "hangman":
        e = mk()
        word = e._env.state.game_state["target_word"].upper()
        for ch in [c for c in string.ascii_uppercase if c not in word]:
            if e.done:
                break
            e.act(f"[{ch}]")
        out.append(("six wrong letters -> 0 <= reward < 1", e.done and not e.success() and 0 <= e.score() < 1, e.score()))
    elif name == "minesweeper":
        e = mk()
        e.act(e._oracle_next())                      # first click lays the mines
        g = e._env.grid
        mines = [(r, c) for r in range(e._env.rows) for c in range(e._env.cols) if g[r][c] == -1]
        for r, c in mines[:2]:
            if not e.done:
                e.act(f"[{r} {c}]")
        out.append(("two mines -> lost, reward < 1", e.done and not e.success() and e.score() < 1, e.result()))
    elif name == "guessnumber":
        e = mk()
        n = e._env.game_number
        for x in [v for v in range(e._env.min_number, e._env.max_number + 1) if v != n]:
            if e.done:
                break
            e.act(f"[{x}]")
        out.append(("never guessing it -> reward < 1", e.done and not e.success() and e.score() < 1, e.score()))
    elif name == "circuit":
        for k in (0, 1, 2, 3):
            e = mk()
            t = list(map(str, e._truth))
            for i in range(k, 3):                      # break circuits k..2
                t[i * 8] = "1" if t[i * 8] == "0" else "0"
            while e._queries < 18:
                e.act("<query>A(0, 0, 0)</query>")
            e.act("<guess>" + "".join(t) + "</guess>")
            out.append((f"{k}/3 circuits right -> {k}/3", _close(e.score(), k / 3) and e.success() == (k == 3), e.score()))
    elif name == "movierec":
        import numpy as np
        e0 = mk()
        u = [e0._utility(s) for s in e0._unseen_scores]
        order = list(np.argsort(u)[::-1])
        for k in (1, 2, 40):
            e = mk()
            while e._asked < 10:
                e.act("nothing")
            pick = e._unseen_names[order[k - 1]]
            e.act(f"<final_answer>{pick}</final_answer>")
            exp = (40 - e._rank()) / 39
            out.append((f"rank-{k} pick -> (40 - rank)/39", _close(e.score(), exp) and e._rank() <= k, (e.score(), e._rank())))
        e = mk()
        while not e.done:
            e.act("<final_answer>Not A Movie</final_answer>")
        out.append(("invalid pick -> 0", e.score() == 0.0, e.score()))
    elif name == "blicket":
        b5 = get("blicket")
        inst5 = b5.instances("n5", "search", limit=1)[0]
        e = b5.make("n5", inst5)
        bl = set(e._blickets)
        flip = "A" if "A" not in bl else "A"
        said = (bl ^ {flip}) or {"B"}
        e.act(f"answer: blickets = {', '.join(sorted(said))}; rule = {e._rule}")
        wrong_items = sum((o in said) != (o in bl) for o in e._objs)
        r = (5 - 2 * wrong_items) + 0.5
        exp = (r + 5.5) / 11
        out.append(("some objects wrong, rule right -> normalised quiz reward", _close(e.score(), exp) and not e.success(), (e.score(), exp)))
        e = b5.make("n5", inst5)
        other = "conjunctive" if e._rule == "disjunctive" else "disjunctive"
        e.act(f"answer: blickets = {', '.join(sorted(bl))}; rule = {other}")
        out.append(("objects right, rule wrong -> 10/11", _close(e.score(), 10 / 11) and not e.success(), e.score()))
    elif name == "boxinggym":
        bb = get("boxinggym")
        for sub2, bad, check in (("hyperbolic/direct", "flip", lambda raw: raw == 0.0),
                                 ("location/direct", "+100", lambda raw: raw > 1.0),
                                 ("location/source", "shape", lambda raw: raw is None)):
            e = bb.make(sub2, "3")
            while not e.done:
                a = e._oracle_next()
                if a.startswith("<answer>"):
                    gt = e._gts[-1]
                    if bad == "flip":
                        a = f"<answer>{1 - int(gt)}</answer>"
                    elif bad == "+100":
                        a = f"<answer>{float(gt) + 100}</answer>"
                    else:
                        a = "<answer>1.0</answer>"
                e.act(a)
            out.append((f"{sub2} {bad} -> {'NaN + eval_error' if bad == 'shape' else 'worst/large raw'}",
                        check(e.raw_score) and (bad != "shape" or (math.isnan(e.score()) and e.eval_error)),
                        (e.raw_score, e.eval_error)))
    elif name in ("babyai", "alfworld", "sciworld"):
        e = mk()
        while not e.done:
            e.act("do a backflip")
        out.append(("only invalid actions -> 0", e.score() == 0.0 and not e.success(), e.result()))
    return out


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benches", default=",".join(BENCHES))
    ap.add_argument("--checks", default="random,partial,leak,concurrent")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    checks = a.checks.split(",")
    report, fails = [], []

    def note(kind, bench, sub, ok, detail):
        report.append({"check": kind, "bench": bench, "subtask": sub, "ok": bool(ok), "detail": str(detail)[:300]})
        if not ok:
            fails.append((kind, bench, sub, str(detail)[:200]))
        print(f"{'ok  ' if ok else 'FAIL'} {kind:10s} {bench}/{sub}  {str(detail)[:120]}", flush=True)

    benches = a.benches.split(",")
    for name in benches:
        b = get(name)
        for sub in b.subtasks:
            insts = b.instances(sub, "search", limit=R_EPISODES)
            if "random" in checks or "leak" in checks:
                scores, wins, leaks = [], [], []
                try:
                    for i, inst in enumerate(insts):
                        e = b.make(sub, inst, stub_responder if b.responder else None)
                        r = play(e, random.Random(i))
                        scores.append(r["score"]); wins.append(r["success"])
                        leaks += leaked(e)
                        o = b.make(sub, inst, stub_responder if b.responder else None)
                        oracle_episode(o, garbage=False)
                        leaks += leaked(o)
                except Exception as ex:
                    note("random", name, sub, False, "crash: " + repr(ex)); traceback.print_exc(); continue
                if "random" in checks:
                    hib = getattr(e, "higher_is_better", True)
                    finite = all(isinstance(s, float) and not math.isinf(s) for s in scores)
                    in_range = name == "boxinggym" or all(0.0 <= s <= 1.0 for s in scores)
                    win_rate = sum(bool(w) for w in wins) / len(wins)
                    ok = finite and in_range and win_rate <= 0.4
                    note("random", name, sub, ok, {"scores": [round(s, 3) if not math.isnan(s) else "nan" for s in scores],
                                                   "win_rate": win_rate, "higher_is_better": hib})
                if "leak" in checks:
                    note("leak", name, sub, not leaks, leaks or "none")
        if "partial" in checks:
            try:
                for label, ok, det in partial_cases(name):
                    note("partial", name, label, ok, det)
            except Exception as ex:
                note("partial", name, "-", False, "crash: " + repr(ex)); traceback.print_exc()

    if "concurrent" in checks:
        jobs = []
        for name in benches:
            b = get(name)
            for sub in b.subtasks:
                inst = b.instances(sub, "search", limit=1)[0]
                for k in range(K_COPIES):
                    jobs.append((name, sub, inst, k))
        random.Random(0).shuffle(jobs)

        def run(job):
            name, sub, inst, k = job
            b = get(name)
            e = b.make(sub, inst, stub_responder if b.responder else None)
            play(e, random.Random(12345))
            return job, e.transcript, e.result()["score"]

        res = {}
        with ThreadPoolExecutor(a.workers) as ex:
            for job, tr, sc in ex.map(run, jobs):
                res.setdefault(job[:3], []).append((tr, sc))
        for (name, sub, inst), runs in res.items():
            same = all(r[0] == runs[0][0] for r in runs) and all(
                (r[1] == runs[0][1]) or (math.isnan(r[1]) and math.isnan(runs[0][1])) for r in runs)
            note("concurrent", name, sub, same, f"{len(runs)} copies, {len(runs[0][0])} steps")

    if a.out:
        json.dump(report, open(a.out, "w"), indent=1)
    print(f"\n{len(report)} checks, {len(fails)} failures")
    for f in fails:
        print("  FAIL", *f)


if __name__ == "__main__":
    main()
