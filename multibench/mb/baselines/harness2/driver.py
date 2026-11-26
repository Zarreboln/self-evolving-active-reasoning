"""Harness² in experiment B: the Mac-side entry point (control loop + Claude Code sessions here,
episodes on ORCD through `play`). Every phase resumes from disk after an interruption: rerun the same
command.

    # before deployment: the stream over the development problems (no scores; k=2 parallel by default,
    # 8 rollouts per problem x 140 problems = 1,120 episodes, the equal budget of 8 rounds x 140)
    python -m mb.baselines.harness2.driver dev  --run h2 --problems P/problems.json --answers P/answers.json \
        --play mailbox:~/mailbox_clone
    # test time, each new task from the library the dev stream left (k=1: 4 problems per step)
    python -m mb.baselines.harness2.driver test --run h2 --problems ... --answers ... --play ...
    # the same with an empty library (no development stream)
    python -m mb.baselines.harness2.driver test --run h2 --empty-library ...
    python -m mb.baselines.harness2.driver report --run h2

Run directory: ~/evo_b_runs/harness2/<run>/ (outside every repository). `private/` holds the episode
records with scores; nothing under it is ever copied into a session's context.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import statistics
from pathlib import Path

from . import evidence as E
from . import guard
from .method import Harness2, slug
from .seams import CachedPlay, CachedSession, LocalBrokerPlay, MailboxPlay, SandboxClaudeCode

RUNS = Path.home() / "evo_b_runs" / "harness2"


def make_session(kind: str, run: Path):
    if kind == "sandbox":
        return CachedSession(SandboxClaudeCode(run / "private" / "session_logs"), run / "sessions")
    if kind == "fake":
        from .fakes import FakeAgent
        return CachedSession(FakeAgent(), run / "sessions")
    raise ValueError(kind)


def make_play(spec: str, run: Path):
    kind, _, arg = spec.partition(":")
    if kind == "mailbox":
        raw = MailboxPlay(arg)
    elif kind == "local":
        raw = LocalBrokerPlay(Path(arg or run / "private" / "local_play"))
    elif kind == "fake":
        from .fakes import FakePlay
        raw = FakePlay()
    else:
        raise ValueError(spec)
    return CachedPlay(raw, run / "private" / "episodes")


def check_meta(d: Path, meta: dict):
    p = d / "run.json"
    if p.is_file():
        old = json.loads(p.read_text())
        diff = {k: (old.get(k), v) for k, v in meta.items() if old.get(k) != v}
        if diff:
            raise SystemExit(f"{d} was started with other settings {diff}; use another --run")
    else:
        E.dump(p, meta)


def phase(a, run: Path):
    probs = json.loads(Path(a.problems).expanduser().read_text())
    answers = guard.Answers(Path(a.answers).expanduser() if a.answers else None)
    if not answers.active:
        print("WARNING: no --answers file: the shared-store check covers names only", flush=True)
    if a.cmd == "dev":
        d = run / "dev"
        check_meta(d, {"phase": "dev", "k": a.k, "mode": a.mode, "n": len(probs["dev"]), "answers_check": answers.active})
        h = Harness2(d, make_session(a.session, d), make_play(a.play, d), answers, k=a.k, mode=a.mode)
        h.dev_stream(probs["dev"], E.Library(d / "library"))
        print("library audit:", guard.audit_dir(d / "library", answers) or "clean", flush=True)
        (d / "DONE").write_text("")
        return
    variant = "test_empty" if a.empty_library else "test"
    if not a.empty_library and not (run / "dev" / "DONE").exists():
        raise SystemExit("the development stream has not finished (run `dev` first, or pass --empty-library)")
    tasks = a.tasks.split(",") if a.tasks else list(probs["test"])
    for name in tasks:
        d = run / variant / slug(name)
        check_meta(d, {"phase": variant, "k": a.k, "mode": a.mode, "task": name, "n": len(probs["test"][name]),
                       "answers_check": answers.active})
        lib_dir = d / "library"
        if not (d / "library_start").exists():                # the library this task starts from, frozen
            if a.empty_library:
                (d / "library_start").mkdir(parents=True)
            else:
                shutil.copytree(run / "dev" / "library", d / "library_start")
            shutil.rmtree(lib_dir, ignore_errors=True)
            shutil.copytree(d / "library_start", lib_dir)
        if a.empty_library is False:
            for c, s, i in probs["dev"]:                       # the dev traces were seen
                answers.mark_seen(c, s, i)
        h = Harness2(d, make_session(a.session, d), make_play(a.play, d), answers, k=a.k, mode=a.mode,
                     sequential_play=True)
        h.test_stream(name, probs["test"][name], E.Library(lib_dir))
        print(f"{name}: library audit:", guard.audit_dir(lib_dir, answers) or "clean", flush=True)
        (d / "DONE").write_text("")


def report(run: Path) -> dict:
    """Test scores by task and along the stream; session counts and tokens per phase."""
    out = {"sessions": {}, "test": {}}
    for ph in sorted(p for p in run.iterdir() if p.is_dir() and (p.name == "dev" or p.name.startswith("test"))):
        metas = [json.loads(m.read_text()) for m in ph.rglob("sessions/*.meta.json")]
        tok = {}
        for m in metas:
            for kk, v in (m.get("usage") or {}).items():
                tok[kk] = tok.get(kk, 0) + v
        out["sessions"][ph.name] = {"sessions": len(metas), **tok}
        if ph.name.startswith("test"):
            for td in sorted(p for p in ph.iterdir() if p.is_dir()):
                probs_order, scores = [], []
                for f in sorted((td / "private" / "steps").glob("*.json")):
                    summ = json.loads((td / "steps" / f.stem / "summary.json").read_text())
                    priv = json.loads(f.read_text())
                    for lbl, p in summ["assignment"].items():
                        if lbl in priv:
                            probs_order.append((summ["step"], lbl, priv[lbl]["score"]))
                scores = [s for _, _, s in probs_order if isinstance(s, (int, float)) and not math.isnan(s)]
                out["test"].setdefault(ph.name, {})[td.name] = {
                    "episodes": len(probs_order), "mean_score": round(statistics.mean(scores), 4) if scores else None,
                    "stream": [s for _, _, s in probs_order]}
    E.dump(run / "report.json", out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["dev", "test", "report"])
    ap.add_argument("--run", required=True)
    ap.add_argument("--root", default=str(RUNS))
    ap.add_argument("--problems")
    ap.add_argument("--answers")
    ap.add_argument("--tasks", default="", help="test tasks to run (default: all in the problem file)")
    ap.add_argument("--empty-library", action="store_true")
    ap.add_argument("--k", type=int, default=None, help="recursion budget (default: dev 2, test 1)")
    ap.add_argument("--mode", default="agg", choices=["agg", "seq"], help="agg = parallel, seq = sequential")
    ap.add_argument("--play", default="mailbox:~/mailbox_clone", help="mailbox:<clone> | local[:workdir] | fake")
    ap.add_argument("--session", default="sandbox", choices=["sandbox", "fake"])
    a = ap.parse_args()
    run = Path(a.root).expanduser() / a.run
    if a.cmd == "report":
        print(json.dumps(report(run), indent=1))
        return
    if a.k is None:
        a.k = 2 if a.cmd == "dev" else 1
    phase(a, run)


if __name__ == "__main__":
    main()
