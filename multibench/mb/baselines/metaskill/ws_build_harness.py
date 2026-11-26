"""Assemble and check your harness: python build_harness.py

Writes harness/harness.py = harness/components.py + runner.py, applies the interface check and the
static rules, and plays the mock game (mock_game.py) in two variants with a fake Target that calls
every listed tool once. Prints `ok` or the problems to fix.
"""
import contextlib
import importlib.util
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from check_harness import problems as static_problems       # noqa: E402
from interface_check import component_problems              # noqa: E402
import mock_game                                            # noqa: E402


def main():
    comp = open(os.path.join(HERE, "harness", "components.py")).read()
    bad = component_problems(comp)
    if not bad:
        src = comp.rstrip() + "\n\n\n" + open(os.path.join(HERE, "runner.py")).read()
        bad = static_problems(src)
        open(os.path.join(HERE, "harness", "harness.py"), "w").write(src)
    if not bad:
        spec = importlib.util.spec_from_file_location("candidate_harness", os.path.join(HERE, "harness", "harness.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for i, sc in enumerate(mock_game.SCENARIOS):
            env = mock_game.MockEnv(**sc)
            out = io.StringIO()
            try:
                with contextlib.redirect_stdout(out):
                    mod.run_episode(env, mock_game.MockChat())
            except Exception as e:
                bad.append("mock %d: run_episode raised %r" % (i, e))
                continue
            text = out.getvalue()
            if "runner stop" in text:
                bad.append("mock %d: %s" % (i, text.strip().splitlines()[-1]))
            elif not env.done:
                bad.append("mock %d: run_episode returned before the episode ended" % i)
    print("\n".join(bad) if bad else "ok")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
