# Signal: metrics and scoring

* `runner/cd_score.py` — episode metrics (uniq@end, exact, circuit/bit accuracy, parsed%) and
  turn metrics (ex-ante efficiency, zero-info%, oracle-agreement%, repeat%, illegal%) with
  bootstrap CIs and paired deltas; errored episodes count as failures (`err%` column).
* `runner/cd_score_evoharness.py` — the same metrics replayed from EvoHarness-RL trajectories.
* `runner/cd_mock.py` — scripted task models (sequential / random / greedy oracle / follow /
  flaky) for CPU self-tests of any harness.
* `runner/cd_ladder_plot.py` — the actionability-ladder figure.
* `runner/turtle_judge_eval.py` — judge calibration on a fixed TurtleBench sample.
* `runner/calibrate_t3_proxy.py` — precision/recall of the T3-style trap proxy.
