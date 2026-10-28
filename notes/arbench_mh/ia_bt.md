# Information acquisition and belief tracking, before and after evolution

Selection split, 100 episodes a point, frozen Qwen2.5-7B-Instruct, all on one machine in one
session (MIT Engaging, L40S). SP's referee is Jev pinned to `jev-1.14.0`, as in run1; the whole
SP column cost $0.15.

The abilities are AREW's (ICML 2026) decomposition of active reasoning, with the first renamed:
**information acquisition** rather than action selection, because what we measure is how much the
questions a harness actually asked were worth, not what its action-selection kernel would do under
a correct belief. **Belief tracking** keeps its name.

## 1. Feedback ablation: does the interaction matter?

Every answer from the oracle is removed and nothing else changes -- the question is still asked,
the turn is still spent. On DC and SP the answer becomes the word "unknown" (AREW's own
intervention; both the record and the history carry that string). On GN that does not work:
feedback reaches a harness through the record text *and* the structured history, and
`packed_group_echo` reads the second one, so it scored 100% "ablated" on the first attempt.
Blanking the structured channel instead makes filtering harnesses crash on an empty candidate set,
which scores zero for the wrong reason. So GN replaces the **oracle**: for the whole episode,
feedback is computed against a decoy secret, drawn once and seeded by the real target. Every
channel is blinded at once, the feedback stays well formed, the harness finishes its machinery and
answers confidently -- about the wrong number.

|      | harness                 |        | real  | ablated | delta  | z    |
|------|-------------------------|--------|-------|---------|--------|------|
| GN   | `template_zero_shot`    | before |   2.0 |     0.0 |   +2.0 | 1.4 n.s. |
| GN   | `packed_group_echo`     | after  | 100.0 |     0.0 | +100.0 | exact |
| DC   | `template_zero_shot`    | before |  31.0 |    24.0 |   +7.0 | 1.1 n.s. |
| DC   | `prior_verify_stop`     | after  |  57.0 |    43.0 |  +14.0 | 2.0  |
| DC   | `lean_verify`           | after  |  77.0 |    18.0 |  +59.0 | 10.4 |
| SP   | `baseline_zero_shot`    | before |   7.0 |     5.5 |   +1.5 | 0.4 n.s. |
| SP   | `family_answer`         | after  |  38.0 |     5.0 |  +33.0 | 6.2  |
| SP   | `atomic_audit`          | after  |  37.0 |     7.5 |  +29.5 | 5.4  |
| SP   | `coarse_first`          | after  |  34.5 |     7.5 |  +27.0 | 5.0  |
| SP   | `baseline_no_questions` | anchor |  15.5 |    12.5 |   +3.0 | 0.6 n.s. |

**On all three tasks, the interaction contributed nothing measurable before evolution.** GN +2.0,
DC +7.0, SP +1.5, none of them distinguishable from zero at n=100. After evolution the same
ablation costs 27 to 100 points. The accuracy gain is not better answer formatting: DC's
no-information baseline *fell* 24.0 to 18.0 while accuracy rose 31.0 to 77.0, and SP's fell 5.5 to
5.0 while accuracy rose 7.0 to 38.0.

Two controls. `baseline_no_questions` asks nothing, so its ablation must change nothing: +3.0,
n.s. And in all three tasks the evolved harnesses, deprived of feedback, score *below* the
ask-nothing anchor -- a system built to synthesise from evidence does worse with none than one
designed never to have any.

The SP numbers here are a fresh paired run, not run1's (41.0 / 39.0 / 36.5). The gap between the
two is resampling at temperature 0.7, which is why both columns were re-run together.

## 2. IA and BT on Guess Number

Where the consistent set `H_t` is computable from the transcript, accuracy factors exactly:

    IA = mean 1/|H_T|   what a perfect reader of this harness's own transcript would score
    BT = accuracy / IA  the share of that the harness turned into a right answer
    accuracy = IA x BT

Named for the abilities, but they are our estimators: IA is the on-policy counterpart of AREW's
I_AS, conditioned on the questions actually asked rather than on actions re-drawn under an oracle
belief, and BT is a ratio off the final answer, not their C_BT, which needs a belief readout.

Only GN qualifies. DC's testimony is free text and its murderer lies; SP's hypothesis space is
unbounded. (Both still get §1 and §3.)

| harness | acc | IA | BT | dead turns | longest dead run | pinned | pinned & wrong |
|---|---|---|---|---|---|---|---|
| `template_zero_shot` (before) |   2.0 |  97.5 | **0.02** | 74.4% | 17.2 |  95.0% | **93.0%** |
| `packed_group_echo` (after)   | 100.0 | 100.0 | **1.00** | 16.7% |  1.0 | 100.0% |    0.0% |

Before evolution the questions were already nearly perfect -- a Bayesian reading that transcript
scores 97.5 -- and the harness scored 2.0. 95% of episodes end with the answer pinned to a single
candidate and 93% of episodes are pinned and still answered wrong. **What evolution moved on GN is
BT, 0.02 to 1.00, not IA.** That matches the delta analysis, where `surviving_set_state`
appears in all fifteen record-setting steps.

The 74.4% dead turns are not in tension with IA 97.5: `template_zero_shot` keeps nothing between
turns and repeats guesses, but GN's feedback is informative enough that the few live turns still
collapse 5040 to one.

Caveat for the second row: `packed_group_echo` makes 0.08 model calls an episode and merges
sixteen concurrent episodes into one "Repeat: <digits>" call. IA 100 / BT 1.00 is a program
solving the puzzle, not a model doing it. The decomposition makes that visible rather than hiding
it -- which is the point -- but the row is about a harness-as-solver.

Under ablation, `IA` is computed against the decoy's feedback, so it measures consistency with
the decoy, not information about the target. Ignore that column on ablated rows.

## 3. Belief trajectory

The belief that matters is the system's, harness state included. Instructing a harness to maintain
and report a belief vector -- AREW's method -- installs the scaffold the search is meant to invent,
and probing the frozen model with the raw transcript reads the bare model with the scaffold
stripped off. So the belief at turn t is read by **ending the episode at turn t and taking the
system's answer**: whatever state the harness keeps is in that answer by construction.

Two costs, run as controls rather than assumed away. The harness sees a shorter budget, so the
full budget must reproduce the ordinary run. And the runs are independent samples at temperature
0.7, so the full budget is run twice and that disagreement rate is the floor under any drift
number. n = 30.

Guess Number, oracle `1/|H_t|` against the system's answer:

| turn | before: oracle / system | after: oracle / system |
|---|---|---|
|  3 |  9.0 /  3.3 |  14.2 /   6.7 |
|  5 | 42.8 /  3.3 |  83.7 /  33.3 |
|  8 | 79.4 /  3.3 | 100.0 / 100.0 |
| 16 | 95.0 /  6.7 | 100.0 / 100.0 |
| 25 | 95.0 /  6.7 | 100.0 / 100.0 |

Before, the gap opens to -88.3 and never closes: the information arrives turn after turn and the
belief does not move. After, it is zero from turn 8.

Detective Cases has no oracle column, so only the system curve:

| turn | 1 | 2 | 3 | 5 | 8 | 12 | 16 | 20 | 25 |
|---|---|---|---|---|---|---|---|---|---|
| `template_zero_shot` (before) | 33.3 | 43.3 | 26.7 | 43.3 | 26.7 | 20.0 | 46.7 | 30.0 | 40.0 |
| `lean_verify` (after)         | 20.0 | 30.0 | 33.3 | 26.7 | 43.3 | 43.3 | 60.0 | 70.0 | 83.3 |

Flat against monotone. Episodes right at some turn and wrong at the end: 43.3% before against a
36.7% noise floor, 10.0% after against 16.7% -- the evolved harness is *below* its own floor, so
there is no measurable self-destructive drift in it.

## What is missing

- SP has no IA/BT columns and DC has none either; §1 and §3 cover all three, §2 covers GN only.
- DC's "before" is `template_zero_shot`, not a published method: the three published DC arms were
  scored through AR-Bench's own evaluator and were never wrapped on the harness interface, so they
  cannot take the ablation switch. `template_zero_shot` scores 31.0, inside the published band of
  31.0 / 34.0 / 35.0, which is why it stands in. SP's `baseline_zero_shot` *is* AR-Bench's own.
- The belief trajectory is n = 30 and the budget points are coarse.
