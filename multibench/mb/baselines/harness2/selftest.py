"""End-to-end CPU test of the Harness² integration with fake environment, fake policy and fake session
agents (no network, no GPU, no paid call).

    cd multibench && python -m mb.baselines.harness2.selftest

Checks: H0 plays exactly like the seed it ports; the official prompts reach the sessions; the dev
stream (k=2 parallel, with selector) and a test stream (k=1, one episode per problem in the fixed
order) run end to end through the child sandbox; both resume after a simulated kill without
repeating finished sessions or episodes and end in the same state as an uninterrupted run; the
empty-library variant starts empty; names and answers never enter the library or any evidence tree,
and no score reaches a session.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from . import evidence as E
from . import guard, official
from . import surface as S
from .fakes import GAME, FakeAgent, FakePlay, Interrupted
from .method import Harness2
from .seams import CachedPlay, CachedSession, parse_stream

SEED = Path(__file__).resolve().parents[2] / "evolve" / "seeds" / "textarena_standard.py"
guard.NAMES.append(GAME.lower())                  # the fake game's name is one the store must not hold
DEV = [["fake:basic", "search", w] for w in ("brick", "grape", "lemon")]
DEV_SELECTION = ["house", "ivory", "joker", "knife"]
TEST = [["fake:hard", "test", w] for w in ("cloud", "eagle", "flame", "dream", "apple", "ivory", "knife",
                                           "house", "joker", "lemon")]


def answers():
    return guard.Answers(table={"fake:basic": {"search": {w: w for _, _, w in DEV},
                                               "selection": {w: w for w in DEV_SELECTION}},
                                "fake:hard": {"test": {w: w for _, _, w in TEST}}})


def tree_files(d: Path) -> dict:
    return {str(p.relative_to(d)): p.read_text() for p in sorted(d.rglob("*")) if p.is_file() and "__pycache__" not in p.parts}


def make(run: Path, agent, play, *, k, sequential_play=False, ans=None):
    return Harness2(run, CachedSession(agent, run / "sessions"), CachedPlay(play, run / "private" / "episodes"),
                    ans, k=k, mode="agg", sequential_play=sequential_play)


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print("  ok:", msg)


def test_h0_equals_seed(tmp):
    print("[1] H0 assembled from the component tree plays exactly like the seed")
    h0 = S.materialise(S.H0, tmp / "h0")
    seed = tmp / "seed" / "harness.py"
    seed.parent.mkdir()
    shutil.copy(SEED, seed)
    p = FakePlay()
    a, b = p.episode(seed, "fake:basic", "grape"), p.episode(h0, "fake:basic", "grape")
    check(a["calls"] == b["calls"] and a["steps_log"] == b["steps_log"] and a["end"] == b["end"] == "end",
          f"identical messages, replies and actions ({len(a['calls'])} calls)")
    check(S.validate(S.H0) == [], "H0 passes validation and mb/evolve/check.py")


def test_prompts():
    print("[2] the sessions get the official Harness² prompts")
    j = official.propose(official.PROMPTS, "INV", official.SURFACE_NAME)
    check("# YOUR DIMENSION: INV -- how the agent WORKS" in j and official.official_context.__name__ ==
          "harness2.core.context", "propose() is the official one, with our surface block")
    check("WHAT YOU DO NOT HAVE" in official.select(official.PROMPTS, 2, pool_kind="finals"), "select() official")
    log = Path(tempfile.mkdtemp()) / "s.jsonl"
    log.write_text('{"type":"system"}\n{"type":"result","subtype":"success","result":"<notes>x</notes>",'
                   '"num_turns":7,"usage":{"input_tokens":10,"cache_read_input_tokens":900,"output_tokens":55},'
                   '"total_cost_usd":0.1}\n')
    text, usage, err = parse_stream(log)
    check(text == "<notes>x</notes>" and usage["output_tokens"] == 55 and not err, "stream-json reply parsing")


def run_dev(run: Path, crash_sessions=None, crash_plays=None):
    agent, play = FakeAgent(crash_after=crash_sessions, answers_to_leak=[w for _, _, w in DEV]), FakePlay(crash_plays)
    h = make(run, agent, play, k=2, ans=answers())
    try:
        h.dev_stream(DEV, E.Library(run / "library"))
        done = True
    except Interrupted as exc:
        print("   interrupted:", exc)
        done = False
    return agent, play, done


def test_dev(tmp):
    print("[3] development stream, k=2 parallel (8 rollouts + 8 sessions per problem)")
    ref = tmp / "dev_ref"
    agent, play, done = run_dev(ref)
    check(done, "uninterrupted run completes")
    check(play.calls == 8 * len(DEV), f"{play.calls} episodes = 4k x {len(DEV)} problems")
    check(all(inst == DEV[i // 8][2] for i, (_, _, inst) in enumerate(sorted(play.log))),
          "every rollout of a step plays that step's problem")
    sums = [json.loads((ref / "steps" / f"dev-{i:04d}" / "summary.json").read_text()) for i in (1, 2, 3)]
    check(not sums[0]["selection"].get("trivial") and sums[0]["selection"]["ok"], "k=2: the blind selector ran")
    st = {s["label"]: s for s in sums[0]["stages"]}
    check(st["dom1"]["attempts"] == 2 and not st["dom1"]["fell_back"],
          "the DOM proposal that wrote the game's name and the answers was rejected, then fixed on retry")
    check(sums[0]["library_added"] and sums[2]["library_added"], "problems 1 and 3 added a library entry")
    check(not sums[1]["library_added"] and "library_refused" in sums[1],
          f"the reflector lesson naming the game was kept out: {sums[1].get('library_refused', [''])[0][:70]}")
    lib = ref / "library"
    check(sorted(E.Library(lib).entries()) == ["task-0001", "task-0003"], "library entries are anonymous ids")
    check(guard.audit_dir(lib, answers()) == [], "library audit: no names, no answers")
    texts = "\n".join(tree_files(lib).values()).lower()
    check(GAME.lower() not in texts and not any(w in texts for _, _, w in DEV), "no game name or answer string in the library")
    ctx = "\n".join(t for p, t in tree_files(ref / "steps").items() if "/context/" in p or "/selection/" in p
                    or "/reflection/" in p)
    check('"score"' not in ctx and "score " not in ctx.lower() and "fake:basic/" not in ctx,
          "no score, success flag or instance id in any evidence tree")
    print("  sessions:", agent.calls, "episodes:", play.calls)

    print("[4] dev stream killed twice (in a session, then before an episode) and resumed")
    run = tmp / "dev_resume"
    a1, p1, d1 = run_dev(run, crash_sessions=9)
    a2, p2, d2 = run_dev(run, crash_plays=6)
    a3, p3, d3 = run_dev(run)
    check(not d1 and not d2 and d3, "two interruptions, then completion")
    done_sessions = sum(a.calls - a.killed for a in (a1, a2, a3))
    check(done_sessions == agent.calls,
          f"no session run twice: {done_sessions} completed across the three attempts = {agent.calls} in the reference")
    done_eps = sum(p.calls - p.killed for p in (p1, p2, p3))
    check(done_eps == play.calls, f"no episode played twice: {done_eps} across the three attempts = {play.calls}")
    check(tree_files(run / "library") == tree_files(ref / "library"), "the library equals the uninterrupted run's")
    for i in (1, 2, 3):
        a = json.loads((run / "steps" / f"dev-{i:04d}" / "summary.json").read_text())
        b = json.loads((ref / "steps" / f"dev-{i:04d}" / "summary.json").read_text())
        assert a == b, i
    check(True, "every step summary equals the uninterrupted run's")
    return ref


def run_test(d: Path, start_lib: Path | None, crash=None):
    if not (d / "library").exists():
        if start_lib:
            shutil.copytree(start_lib, d / "library")
        else:
            (d / "library").mkdir(parents=True)
    agent, play = FakeAgent(crash_after=crash), FakePlay()
    ans = answers()
    for c, s, i in DEV:
        ans.mark_seen(c, s, i)
    h = make(d, agent, play, k=1, sequential_play=True, ans=ans)
    try:
        h.test_stream("fake_hard", TEST, E.Library(d / "library"))
        return agent, play, True
    except Interrupted as exc:
        print("   interrupted:", exc)
        return agent, play, False


def test_test(tmp, dev_run):
    print("[5] test stream, k=1: one episode per problem in the fixed order, 4 problems per step")
    d = tmp / "test"
    a1, p1, ok1 = run_test(d, dev_run / "library", crash=5)
    a2, p2, ok2 = run_test(d, dev_run / "library")
    check(not ok1 and ok2, "killed once, resumed to completion")
    played = [inst for _, _, inst in p1.log + p2.log]
    check(played == [w for _, _, w in TEST], f"each of the {len(TEST)} problems played exactly once, in order")
    sums = [json.loads((d / "steps" / f"test-fake_hard-{j:03d}" / "summary.json").read_text()) for j in (1, 2, 3)]
    check(sums[0]["assignment"] == {"base1": TEST[0], "inv1": TEST[1], "dom1": TEST[2], "agg1": TEST[3]},
          "step 1: H0 on p1, INV on p2, DOM on p3, composed harness on p4")
    check(sums[0]["selection"]["trivial"], "k=1: the single final is adopted without a selector session")
    check(sums[2]["assignment"] == {"base1": TEST[8], "inv1": TEST[9]} and not sums[2]["library_added"],
          "the last step (2 problems left) plays what it can and adds nothing")
    ents = E.Library(d / "library").entries()
    check(ents == ["task-0001", "task-0003", "task-t0001", "task-t0002"], f"library after the stream: {ents}")
    ctx1 = d / "steps" / "test-fake_hard-001" / "context" / "inv1"
    check((ctx1 / "library" / "INDEX.md").is_file(), "test-time sessions read the library the dev stream built")
    check("different instance" in (ctx1 / "task.md").read_text(), "task.md says each rollout is a different instance")
    priv = json.loads((d / "private" / "steps" / "test-fake_hard-001.json").read_text())
    print("  scores along the stream (private):", [priv[l]["score"] for l in ("base1", "inv1", "dom1", "agg1")])

    print("[6] empty-library variant")
    e = tmp / "test_empty"
    a, p, ok = run_test(e, None)
    ctx = e / "steps" / "test-fake_hard-001" / "context" / "inv1"
    check(ok and not (ctx / "library").exists(), "first step has no library; the run completes")
    check(guard.audit_dir(d / "library", answers()) == [] and guard.audit_dir(e / "library", answers()) == [],
          "both test libraries pass the names/answers audit")


def test_guard():
    print("[7] the shared-store check itself")
    a = answers()
    for c, s, i in DEV:
        a.mark_seen(c, s, i)
    check(guard.violations({"m.md": "use letter feedback"}, a) == [], "clean text passes")
    check(guard.violations({"m.md": "In Wordle, guess vowels"}, a) != [], "a game name is caught")
    check(guard.violations({"m.md": "seen: brick, grape, lemon"}, a) != [], "seen answers are caught (check.leaks)")
    check(guard.violations({"m.md": "e.g. house"}, a) == [], "a single unseen-split word is not a leak")


def main():
    tmp = Path(tempfile.mkdtemp(prefix="h2_selftest_"))
    test_h0_equals_seed(tmp)
    test_prompts()
    test_guard()
    ref = test_dev(tmp)
    test_test(tmp, ref)
    print(f"ALL OK ({tmp})")


if __name__ == "__main__":
    main()
