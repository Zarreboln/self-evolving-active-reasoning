"""Harness² (arXiv 2610.14224; github.com/google-research/harness2 @ 6868817, Apache-2.0, cloned in
ar_harness/third_party_b/harness2) as a baseline of experiment B (notes/exp_b_design.md).

    official.py   the official prompts and mission wrapper, imported from the clone; our surface block
    surface.py    the eight-component tree <-> our one-file harness (mapping table in its docstring)
    runtime.py    the fixed agent loop the components plug into (H0 = the TextArena standard seed)
    evidence.py   evidence trees and the library (ports of core/context.py, core/library.py)
    method.py     Propose-Probe-Compose, selection, reflection, streams (ports of core/recursion.py,
                  core/selection.py, core/schedule.py)
    guard.py      no game/benchmark names and no answers in the library
    seams.py      agent_session (Claude Code in the Docker sandbox) and play (ORCD) behind one call each
    driver.py     the Mac-side entry point (dev / test / test --empty-library / report), resumable
    problems.py   builds problems.json and answers.json where the environments are installed
    fakes.py, selftest.py   CPU end-to-end test: python -m mb.baselines.harness2.selftest
"""
