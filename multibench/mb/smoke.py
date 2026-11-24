"""Smoke test for every adapter. Two modes, both cheap:

  oracle (default, no GPU)  For each bench and subtask: list the instances of every split, build
                            one search instance, play a known-good action sequence (the bench's
                            own expert where it has one), and check that the episode ends and
                            scores what the bench says a solve scores. LLM-played roles are a stub.
                            One garbage action is also sent first on each, to exercise the
                            invalid-action path.
  llm                       The plain react harness on N search instances per bench, against
                            POLICY_BASE_URL. Checks the loop end to end, not the score.

    source mb/env.sh
    python -m mb.smoke                       # oracle, every bench, first two subtasks each
    python -m mb.smoke --all-subtasks
    python -m mb.smoke --mode llm --n 2 --benches wordle,babyai
"""
from __future__ import annotations

import argparse
import json
import time
import traceback

from mb.benches import BENCHES, get
from mb.core import SPLITS


def stub_responder(messages, temperature=0.7, max_tokens=None):
    return "No"


def oracle_episode(env, garbage=True):
    if garbage and not env.done:
        env.act("xyzzy garble")
    while not env.done:
        a = env._oracle_next()
        if a is None:
            break
        env.act(a)
    return env.result()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["oracle", "llm"], default="oracle")
    ap.add_argument("--benches", default=",".join(BENCHES))
    ap.add_argument("--all-subtasks", action="store_true")
    ap.add_argument("--subtasks", default=None, help="comma list; overrides the default first two")
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.mode == "llm":
        from mb.harness.react import run_episode
        from mb.llm import Counter, policy_chat, responder_chat
    rows, failures = [], []
    for name in a.benches.split(","):
        try:
            b = get(name)
        except Exception as e:
            failures.append((name, "-", repr(e)))
            print(f"FAIL {name}: {e!r}", flush=True)
            continue
        subs = b.subtasks if a.all_subtasks else b.subtasks[:2]
        if a.subtasks:
            subs = [s for s in b.subtasks if s in a.subtasks.split(",")]
        for sub in subs:
            t0 = time.time()
            try:
                sizes = {s: len(b.instances(sub, s)) for s in SPLITS}
                for inst in b.instances(sub, "search", limit=a.n):
                    if a.mode == "oracle":
                        env = b.make(sub, inst, stub_responder if b.responder else None)
                        r = oracle_episode(env)
                    else:
                        pc, rc = Counter(), Counter()
                        env = b.make(sub, inst, responder_chat(rc) if b.responder else None)
                        r = run_episode(env, policy_chat(pc))
                        r["policy_tokens"], r["responder_tokens"] = pc.as_dict(), rc.as_dict()
                    r.update(sizes=sizes, seconds=round(time.time() - t0, 1),
                             instructions_head=env.instructions[:160].replace("\n", " | "))
                    rows.append(r)
                    print(json.dumps({k: r[k] for k in ("bench", "subtask", "steps", "done", "score",
                                                        "success", "sizes", "seconds")}), flush=True)
            except Exception as e:
                failures.append((name, sub, repr(e)))
                print(f"FAIL {name}/{sub}: {e!r}", flush=True)
                traceback.print_exc()
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)
    print(f"\n{len(rows)} episodes, {len(failures)} failures")
    for f in failures:
        print("  FAIL", *f)


if __name__ == "__main__":
    main()
