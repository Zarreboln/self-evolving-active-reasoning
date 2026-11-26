"""CPU-only end-to-end test of the ACE baseline with fakes (no GPU, no network, no credential).

    cd multibench && python3 -m mb.baselines.ace.test_fake

Checks:
  1. every official seed turns into a harness that passes mb/evolve/check.py;
  2. offline phase: runs end to end, keeps to the episode budget, retries failed problems with the
     reflection, and gives the same playbook, episode count and role calls when cut off midway and
     resumed as when run straight through;
  3. test phase (both variants): one episode per problem, in the given order, test traces carry no
     score, and an interruption midway resumes without replaying anything;
  4. no benchmark name and no answer of a played problem ever reaches a playbook (the fake Curator
     tries to write both), and the design's answer-copying test passes.
"""
from __future__ import annotations

import json
import re
import shutil
import tempfile
from pathlib import Path

from mb.baselines.ace import core, guard, harness, official
from mb.baselines.ace.fakes import FakeBackend, FakeLLM, Preempted
from mb.evolve.check import problems

WORDS = ["crane", "slate", "pious", "zesty", "brick", "flame", "ghost", "lunar", "mirth", "nudge",
         "otter", "plaid", "quirk", "rover", "shard", "tempo", "unzip", "vigor", "woken", "yield"]
CELLS = ["wordle:basic5", "hangman:basic", "dc:all"]
SEARCH = {c: WORDS[k * 4:(k + 1) * 4] for k, c in enumerate(CELLS)}
SELECTION = {c: WORDS[12:] for c in CELLS}


def offline_problems():
    per = [[(c, i) for i in SEARCH[c]] for c in CELLS]
    return [p for k in range(4) for p in (q[k] for q in per)]


def new_run(root, llm, backend, prefix="t/offline", init=None):
    return core.ACERun(root, llm, backend, prefix, token_budget=600, seed_override="textarena_standard",
                       initial_playbook=init)


def check_store(playbook, played):
    assert not guard.has_names(playbook), "benchmark name in playbook"
    hits = guard.answer_hits(playbook, played)
    assert not hits, f"answers in playbook: {hits}"


def main():
    tmp = Path(tempfile.mkdtemp(prefix="ace_test_"))
    try:
        # 1. seeds
        for seed in ("textarena_standard", "balrog_naive", "swiftsage_react", "arbench_zero_shot"):
            src = harness.build(seed, official.EMPTY_PLAYBOOK + "\n[sai-00001] helpful=0 harmful=0 :: x", "why")
            assert not problems(src), (seed, problems(src))
        print("1. harness from every official seed passes mb/evolve/check.py")

        # 2. offline, straight through
        budget, epochs = 30, 2
        llm_a, be_a = FakeLLM(), FakeBackend()
        ra = new_run(tmp / "a", llm_a, be_a)
        pa = core.offline(ra, offline_problems(), budget, epochs=epochs)
        ev = [json.loads(l) for l in (tmp / "a" / "events.jsonl").read_text().splitlines()]
        assert ra.state["episodes"] <= budget and ra.state["episodes"] == len(be_a.calls)
        retries = [c for c in be_a.calls if "post_reflect" in c["key"]]
        assert retries and all(c["split"] == "search" and not c["visible_only"] for c in be_a.calls)
        n_bullets = len([l for l in pa.splitlines() if official.pb.parse_playbook_line(l)])
        print(f"2a. offline straight: {len(ev)} problems, {ra.state['episodes']} episodes (budget {budget}), "
              f"{len(retries)} retries, {len(llm_a.calls)} role calls, playbook {n_bullets} bullets, "
              f"~{core.approx_tokens(pa)} tokens")
        later = json.loads((tmp / "a" / "plays" / "e2_s001_gen_initial.json").read_text())["trace"]
        assert harness.INTRO in later and re.search(r"\[(err|sai)-\d{5}\] helpful=", later), "playbook not in prompt"
        pruned = sum(len(e.get("pruned", [])) for e in ev)
        print(f"    the agent's prompt carries the playbook; {pruned} bullets pruned at the size cap")
        guards = [g for e in ev for g in e.get("guard", [])]
        assert any("redacted" in g for g in guards) and any("dropped_for_answers" in g for g in guards)
        assert any("curator_error" in e for e in ev)
        check_store(pa, ra.state["played_answers"])
        flagged = {c: guard.copying(pa, SEARCH[c], SELECTION[c]) for c in CELLS}
        assert not any(flagged.values()), flagged
        print(f"2b. guards fired: {sum('redacted' in g for g in guards)} name redactions, "
              f"{sum('dropped_for_answers' in g for g in guards)} ops dropped for answers; "
              f"{sum('curator_error' in e for e in ev)} malformed curator replies skipped; copying test clean")

        # 2c. the same, cut off twice and resumed
        llm_b = FakeLLM()
        for cut in (7, 19, None):
            be = FakeBackend(fail_after=cut)
            rb = new_run(tmp / "b", llm_b, be)
            try:
                pbk = core.offline(rb, offline_problems(), budget, epochs=epochs)
            except Preempted as e:
                print("    interrupted:", e)
        assert pbk == pa and rb.state["episodes"] == ra.state["episodes"]
        assert len(llm_b.calls) == len(llm_a.calls), (len(llm_b.calls), len(llm_a.calls))
        plays_b = sorted(p.name for p in (tmp / "b" / "plays").iterdir())
        plays_a = sorted(p.name for p in (tmp / "a" / "plays").iterdir())
        assert plays_a == plays_b
        print("2c. interrupted twice + resumed: same playbook, same episodes, same role calls, no episode replayed")

        # 3. test phase, both variants, with an interruption
        order = {"task_x": [("wordle:hard5", w) for w in WORDS[12:20]],
                 "task_y": [("hangman:hard", w + "y") for w in WORDS[:6]]}
        for variant, init in (("ace", pa), ("ace_online_only", None)):
            llm = FakeLLM()
            for task, probs in order.items():
                calls = []
                for cut in (3, None):
                    be = FakeBackend(fail_after=cut)
                    r = new_run(tmp / variant / task, llm, be, f"t/{variant}/{task}", init)
                    try:
                        final = core.online(r, probs)
                    except Preempted:
                        pass
                    calls += be.calls
                assert [c["instances"][0] for c in calls] == [i for _, i in probs], "order / one episode each"
                assert all(c["visible_only"] and c["split"] == "test" for c in calls)
                check_store(final, [i for _, i in probs])
                if init is not None:
                    assert final.startswith(init.split("\n")[0])
            refl = [c["prompt"] for c in llm.calls if c["role"] == "reflector"]
            assert refl and not any(re.search(r"^score ", p, re.M) or "Episode score" in p for p in refl)
            assert all(core.TEST_FEEDBACK in p for p in refl)
            assert not any(harness.INTRO in c["prompt"] for c in llm.calls), "inserted block leaked into a role prompt"
            print(f"3. test '{variant}': {sum(len(v) for v in order.values())} problems, one episode each in order, "
                  f"no score in any role prompt, resumed after interruption, store clean")
        print("ALL OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
