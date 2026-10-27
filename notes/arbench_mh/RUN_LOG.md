# Run log — Meta-Harness on AR-Bench

What was built, what broke, and what each number came from. Dated entries, newest last. This is a
record of the process, not a curated results document; numbers here are what was actually measured,
including the ones that were later found to be wrong and why.

---

## 2026-10-26 / 27 — build and first launch

### Infrastructure

| Piece | Where | Note |
|---|---|---|
| Harness interfaces | `gn/`, `dc/`, `sp/harness_base.py` | `solve()` plus a frozen-model `llm()`; hidden state lives in the env |
| Environments | `gn/env.py`, `dc/env.py`, `sp/env.py` | AR-Bench's own prompts and scorers imported, not restated |
| Episode runner | `benchmark.py` (gn/dc), `sp/benchmark.py` | intercepts `arbench.utils.inference` so AR-Bench's own helpers are costed too |
| Loop CLI | `mh_benchmark.py` | `--memory / --frontier / --results`, frontier maths copied from upstream |
| Splits | cluster `~/arbench_mh_splits/<task>/` | search 50 + selection 100 from **train**; `test.json` sealed |
| Run dir | `runs/dc/` | upstream loop with two lines changed, plus `config.yaml` and the skill |

The loop and the proposer run on the Mac (only it has `uv` and the `claude` CLI); episodes run on
the cluster (only it has AR-Bench, the splits and the served model). `evaluate()` ships the
candidate over, runs and scores it there, and copies back the one `val.json` the loop reads.

### Seeds and anchors, selection split, 100 instances, frozen Qwen2.5-7B

Seeds were measured with **AR-Bench's own evaluator** rather than rewrapped, so no wrapper fidelity
question arises. Fidelity check on SP zero_shot: our wrapper 0.665–0.675 mean char-F1 against the
official evaluator's 0.677–0.703 over 8 puzzles — equal within the measurable noise (see below).

| task | system | accuracy | total tokens/ep |
|---|---|---|---|
| gn | baseline_tot_fixed | 6.0% | 21,177 |
| gn | baseline_proactive_cot | 2.0% | 46,161 |
| gn | baseline_zero_shot | 0.0% | 31,094 |
| gn | **baseline_no_questions** (anchor) | 0.0% | 226 |
| dc | **baseline_no_questions** (anchor) | **43.0%** | 851 |
| dc | baseline_proactive_stop | 35.0% | 114,648 |
| dc | baseline_few_shot_inst | 34.0% | 286,863 |
| dc | baseline_tot | 31.0% | 95,518 |
| sp | baseline_tot_eval | 12.5% | 8,624 |
| sp | **baseline_no_questions** (anchor) | 10.5% | 216 |
| sp | baseline_zero_shot | 4.5% | 37,413 |
| sp | baseline_few_shot_inst | 3.5% | 54,746 |

UoT is **held out of every seed set** rather than measured here: it is the known best on gn
(100/100) and seeding it lets a proposer copy it instead of finding the mechanism. Its position
comes from the test split — gn 100.0% / 801,262 tokens, dc 30.0% / 50,144, sp 5.0% / 21,872.

### Findings that stand on their own

**DC: interrogating is net negative.** The anchor that asks nothing scores 37 / 43 / 43 / 38 / 41 /
42 over six runs (mean 40.7) against 31–35% for the three published methods, at 1/100th the tokens.
A hand-written 3-turn probe scored 30.0%. The three published numbers are single measurements, so
the comparison is not yet statistically settled; the anchor's own spread is ±2.8.

**SP: the official metrics overstate by an order of magnitude.** Same 8 methods, same BALAR prompt,
only the judge changed:

| method | char F1 | self-judged | jev-1.14.0 |
|---|---|---|---|
| tot_eval | 0.280 (lowest) | 57 | **18/100** (highest) |
| zero_shot | 0.638 (highest) | 63 | 14 |
| few_shot_inst | 0.627 | 68 (highest) | 11 |
| uot | 0.565 | 65 | 5 |
| proactive_cot | 0.586 | 13 | 0 |

Strict equivalence is **0/100 for all eight**, 800 episodes. The self-judge called "poisoning"
equivalent to "remotely hacking a pacemaker"; asked whether a text entails itself it answers
`neutral` on 4 of 10 selection-split golds, deterministically, where Jev answers entailment 10/10.

**SP: the bottleneck is asking, not writing.** Handing the model every gold key question lifts
partial correctness from 18% to 60/20, while full equivalence stays at 0/20.

**AMIGO: Skip rate is largely a sampling-diversity artifact.** τ=0.8, 50 episodes, Qwen3.5-9B:
temperature 0.0 → 72.0% Skip, 0.3 → 66.7%, 0.7 → 48.9%, 1.0 → 37.5%. At low temperature the model
re-asks the same question and the duplicate and P5 checks catch it.

### Measurement properties that constrain everything above

**The benchmark is not reproducible per instance.** AR-Bench's own evaluator, run twice on the same
3 puzzles at temperature 0, gave 0/3 identical predictions and one puzzle scored 0.7742 then
0.4937. A single call at temperature 0 *is* deterministic (5/5 identical), so the divergence comes
from the 25-turn trajectory amplifying prefix-cache and batching nondeterminism. Per-instance
comparisons are therefore impossible; only distributions compare.

**Non-strict equivalence is the stable signal.** The same harness re-run on the same 50 puzzles
scored 35/50 both times on non-strict while every prediction differed; strict moved 12 → 17 and
char F1 0.646 → 0.672 over that same pair.

### Bugs found and fixed

| Bug | Symptom | Cause | Fix |
|---|---|---|---|
| Token backfill | every DC episode reported 102 tokens | `apply_chat_template(tokenize=True)` returns a BatchEncoding in transformers 5.x; `len()` counted its two keys, 51 assistant turns × 2 | read `len(out["input_ids"])` |
| Judge self-check | refused to score a finished run | one text asked three times, all three had to pass; a 5% per-call slip trips it ~14% of the time | ten distinct texts, one try each, 90% must pass |
| `dc_tot_eval` | `KeyError` on every case | `keypoint_eval_agents` is built from `suspects[i]["key_question"]`, a presence test, so emptying the list did not skip the oracle block | remove the key entirely; also saves 75 calls an episode |
| AMIGO P5 | 33 of 39 Skips, every episode pinned at 19 turns / 13 Skips | P5 implemented as "one question per attribute type, ever" against a 7-type taxonomy, capping informative turns at 7 | ask Jev whether the question tries another *value* of a covered attribute, as the rule says |
| AMIGO repeats | the identical question passed three turns running | P5 asks about another *value*; an identical question is not another value | exact/near-duplicate check in code |
| Per-episode tokens | an episode using 682 tokens reported 1517 | one shared counter with a delta around each episode; sixteen threads, overlapping intervals | one counter per episode, chained to a run total |
| **Package not importable** | **two proposer sessions, 90 min, zero candidates** | **`__init__.py` existed on the cluster but not locally, and the loop sets no PYTHONPATH; the import check it runs was never tested the way the loop runs it** | **declare `arbench-mh` as an editable path dependency in `runs/` and `runs/dc/`** |

The last one is the one worth remembering: the import check was verified once, by hand, with
`PYTHONPATH` exported. That is not the path the loop takes. The proposer found the breakage
immediately and spent its whole budget investigating it — correctly, and twice.

### Environment verification (after the fix)

Run from a shell with `PYTHONPATH` explicitly unset:

1. `cd runs && uv run python -c "from arbench_mh.dc.agents.baseline_no_questions import *"` — the
   loop's own command, from the loop's own cwd. OK
2. `benchmark.py --results` OK
3. `benchmark.py --frontier` OK
4. A **newly written** candidate (`_probe_three_turns`) through import check → ship → evaluate →
   score → `val.json` back locally, 19 s for 100 episodes, 7 calls an episode as expected. OK
5. The frontier correctly excluded it (30.0% / 4,980 is dominated by 43.0% / 851). OK
6. `--fresh` keeps `baseline_no_questions.py` and `template_zero_shot.py`; the template is the only
   file showing `env.interrogate` in use, so it was added to `BASELINE_FILES`. OK

### Pinned versions and conventions

- Referee shim and equivalence judge both **`jev-1.14.0`**, pinned. The shim's default was
  `jev-latest`; the eight SP baselines in the main table were refereed by `jev-latest`, measured
  identical to 1.14.0 on 2026-10-26. 1.13 vs 1.14 as equivalence judge: 96–98% per-episode
  agreement, 1.13 slightly more permissive (20 vs 18, 14 vs 10 partial).
- Cost axis is **input + output**. On gn, uot spends 181k read and 620k generated an episode: by
  input it looks 8.6× the cheapest seed, by total it is 38×, and the generated side is the nine
  hours.
- Proposer: `model="opus"`, `effort="max"`, subscription auth (the loop strips
  `ANTHROPIC_API_KEY`).

### Not yet verified

- A full iteration: proposer writes candidates → all three validate → all three evaluate → frontier
  updates → next iteration reads it. Only the pieces have run, never the sequence.
- `--fresh` and resume have not been exercised end to end with our `BASELINE_FILES`.
- `--fresh` and resume have not been exercised end to end with our `BASELINE_FILES` (see the
  `BASELINE_FILES` bug found on 2026-10-27, below -- `--fresh` would have deleted two SP seeds).


## 2026-10-27 — what the proposer can reach

The loop gives the proposer `--dangerously-skip-permissions` with Read/Glob/Grep/Write/Edit/Bash
and no sandbox. That is upstream's own configuration: Meta-Harness keeps its held-out
`data/<task>/test.jsonl` in the same tree the proposer works in and relies on the skill saying not
to look. Isolation there is layout and convention, not enforcement. We match that for the splits
and go further for one thing, below.

### What an audit of the first sessions found

Three proposer sessions, read back from their transcripts:

| run | command | what it reached |
|---|---|---|
| dc | `python3 -I -` over `search.json`, `selection.json`, `test.json` | **the held-out split** |
| gn | `cat ~/code/AR-Bench/arbench/reasoner/gn/prompt.py` | the UoT prompts |
| sp | `sed -n 49,360p sp_evaluator.py` | ToT and the traditional runner (above the UoT span) |

The gn read propagated: the candidate it then wrote, `survivor_menu.py`, maintains the full
consistent answer set and says so in its docstring -- *"UoT fixes this by showing the model the
whole answer set, which is why it costs ~800k tokens an episode"*. That is the mechanism we hold
out, arrived at by reading it rather than by search.

### Splits

`test.json` and `test_flat.json` moved to `~/sealed/<task>/`. The proposer's working directory now
holds only `search.json` and `selection.json` (plus their flat forms). `benchmark.py` and
`sp/benchmark.py` pick the root by split, so finalization still reads the held-out data.

### UoT

The weaker claim ("it is somewhere else in $HOME") is not worth making, so the text is gone from
the machine instead. The pristine tree is `tmlr-group/AR-Bench` upstream and a verified copy sits
on the Mac at `~/Desktop/ar_harness_sealed/`, which the cluster cannot reach.

Removed from `~/code/AR-Bench` by `meta/tools/prune_uot.py` (spans established by `ast`
reachability from the UoT entry point, not by eye):

| file | removed |
|---|---|
| `gn_evaluator.py` | 252 lines: the runner and 7 helpers only it reaches |
| `dc_evaluator.py` | 297 lines: the runner and 5 `_uot_*` helpers |
| `sp_evaluator.py` | 462 lines: the runner and 9 helpers |
| three `prompt.py` | 12 templates, each checked unreferenced elsewhere in the repo |
| all three | the `elif method == "uot"` branch and `main`'s `simulate_depth` |
| `README.md`, `script/evaluation/*.sh` | the method name and the depth flag |

`respond_template` in dc is shared with other methods and stayed. Verified after: everything
compiles, all three evaluators import, `_run_tot_`/`_run_traditional_`/`_run_proactive_cot_` are
intact, every symbol our own code imports resolves, and 6 selection smoke runs plus one sealed-split
run reproduce their earlier numbers.

Also pulled to the Mac and deleted from the cluster: 21 of my own analysis scripts that named the
method, 108 result files and 7 logs from its baseline runs, and the three proposer transcripts that
had quoted it. 364 files, every one checksum-matched before deletion. A full `$HOME` sweep for
`\buot\b`, `uncertainty of thought` and `simulate_depth` now returns nothing outside binaries.

This repo still discusses the method, by design: the constraint is on what the proposer's
environment contains, and the paper has to say what was held out and why.

### Candidates discarded

Every candidate written before the two fixes above is gone -- gn's 2 (one of them the UoT
restatement) and dc's 3, which were written while the test split was readable. All three tasks
restarted from seeds only. Cost: one iteration each, none of which had completed anyway.

### A bug the reset found

SP's `BASELINE_FILES` listed `{__init__.py, baseline_no_questions.py}` but SP seeds three published
anchors. `baseline_zero_shot.py` and `baseline_few_shot_inst.py` were therefore classed as proposer
candidates -- `--fresh` would have deleted two seeds, and the reset nearly did. Both are now
registered. (Timestamps settled it: 10-26 12:10 and 12:30, eighteen hours before run1.)

`sp/benchmark.py` also printed `<dirname>/episodes.json` regardless of `--out`, while writing to
`--out` correctly. The file was always right and the message always wrong. Fixed.

### The deletion broke the scoring step

`mh_benchmark.py` scores every candidate by shelling out to `~/code/score_val.py`, and that file
named the method in one docstring about the cost axis, so the sweep deleted it. The loops would
have failed at evaluate, roughly an hour after launch, with the proposer's work already spent.
Restored at the same path with that paragraph rewritten to state the 8.6x / 38x spread without
naming which baseline it belongs to, then checked by scoring one real output per task:
gn 0.000 / 21,146 tok, dc 0.333 / 38,107, sp 0.167 / 36,518 with the judge self-check passing.

`score_val.py` is the loop's only dependency outside the package (`grep` over all of `~/meta`
confirms it), and `sp_semantic_equivalence.py`, which it imports, needs nothing further. Worth
vendoring into `arbench_mh/` so a cleanup of a scratch directory cannot take the scorer with it;
not done now, with runs in flight.

### run1, finished

All three relaunched 07:03 UTC and recorded five iterations each by 09:43, policy
`ml-16-node-002:9445`, referee shim `jev-1.14.0` on 8099. Selection split, 100 episodes a point,
frozen Qwen2.5-7B-Instruct. Figures: `figures/evolution_frontier.*`, `figures/evolution_trajectory.*`;
the numbers behind them in `evolution_run1_snapshot.json`, pulled by `meta/tools/pull_snapshot.py`.

**No published AR-Bench method survives on the frontier of any of the three tasks.**

| | frontier, cheapest first | published arms |
|---|---|---|
| gn | `packed_group_echo` 100.0% @ 7 | tot_fixed 6.0 @ 21,176 · zero_shot 0.0 @ 31,094 · proactive_cot 2.0 @ 46,161 |
| dc | anchor 37.0 @ 863 → `prior_verify_stop` 57.0 @ 20,854 → `lean_verify` 80.0 @ 25,344 | tot 31.0 @ 95,518 · proactive_stop 35.0 @ 114,648 · few_shot_inst 34.0 @ 286,863 |
| sp | anchor 12.0 @ 216 → `family_answer` 36.5 @ 8,448 → `coarse_first` 39.0 @ 12,261 → `atomic_audit` 41.0 @ 12,368 | tot_eval 12.5 @ 8,624 · zero_shot 4.5 @ 37,413 · few_shot_inst 3.5 @ 54,746 |

Detective Cases moved every iteration: 37.0 → 60 → 64 → 71 → 76 → 80. All three published arms
score below the anchor that asks no questions at all and answers for 863 tokens. Situation Puzzles
went 12.0 → 10.5 (iteration 1 lost ground) → 36.5 → 39.0 → 41.0 → 41.0, and `family_answer` finally
dominated `tot_eval`: more accurate and cheaper.

### Guess Number is not a result, it is a measurement artifact

`packed_group_echo` reaches 100.0% at **7 tokens an episode**, and the mechanism is worth stating
plainly because it is not reasoning:

- **0.08 model calls an episode. 92 of 100 episodes call the model zero times.**
- The harness solves the game itself -- exact elimination over the 5040 candidates -- and the model
  exists only to say the answer out loud.
- The 8 calls that do happen each serve a *group of sixteen concurrently running episodes*: the
  harness merges those sixteen answers into one digit string by overlapping their suffixes and
  prefixes, asks "Repeat: <string>", and each episode reads its own four digits back out.
- Its docstring explains the packing choice by Qwen2.5's tokenizer: every digit is one token, so
  separators were a fifth of the cost, and sixteen answers pack into ~52 digits instead of 64.

Two things follow. The search found that the cost axis is **per episode** while the evaluation
runs episodes **concurrently**, so cost can be amortised across episodes that merely happen to be
in flight together -- a property of our benchmark runner, not of the task. And the earlier
convention that "every guess played, and the final answer, is a number the model wrote", which the
iteration-1 candidates stated in their own docstrings, no longer holds.

The honest reading: gn saturated at iteration 1, where three different candidates all hit 100.0%,
and iterations 2-5 only drove cost from 6,283 tokens down to 7. Report gn as saturated, report the
cost collapse as a finding about the metric, and do not use gn as evidence that the search improves
reasoning. The two tasks that still carry a signal are dc and sp.

### The anti-stall change worked

Shipped after iteration 4 (dc) / 3 (gn) / 2 (sp). Before it, two of the first nine proposer
sessions hit the 40-minute cutoff and recorded nothing. After it, none did:

| | per-iteration wall clock |
|---|---|
| gn | 31.9 · **40.1 lost** · 7.3 · 25.2 · 21.2 · 18.5 |
| dc | 17.6 · 25.2 · 33.2 · 19.0 · 18.5 |
| sp | 18.9 · 37.5 · **40.1 lost** · 22.6 · 16.9 · 19.8 |

### Not yet done

- **Finalization has not been run.** `supervise.sh` exits when five iterations are recorded; it
  does not call `meta_harness.py --run-name run1 --test`. The held-out split in `~/sealed/` is
  still untouched, and that run happens once.

