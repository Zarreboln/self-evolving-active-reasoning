# Environments

The task code lives where the scripts expect it:

* `stage0/cd_oracle.py` — CircuitDecoding hypothesis space (all joint truth tables buildable
  from the gate inventory), exact feasible-set bookkeeping, max-entropy oracle and the
  ex-ante efficiency measure. Verified on all 300 released instances.
* `stage0/gn_oracle.py`, `stage0/dc_oracle.py`, `stage0/generic.py` — the same for AR-Bench
  GuessNumbers / DetectiveCases and the family survey.
* `mt_puzzles/` — CircuitDecoding data and the official instruction / last-turn prompts.
* `splits/cd_split.json` (search 50 / held-out 250), `splits/cd_split_v2.json` (100 / 200).
* `runner/cd_harness_run.py` — the episode loop with the paper's environment wording and the
  pluggable harness interface (`init_state / build_feedback / final_hint / filter_action /
  compose_messages`); the truth table never enters a harness.
* `runner/sp_runner_async.py`, `runner/run_arbench_sp.py` — AR-Bench Situation Puzzles with a
  separate judge/respondent endpoint.
