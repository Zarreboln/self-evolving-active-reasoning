Now the edits: docstring, the ranking key, menu wording, and the ask prompt.On query count, no improvement: 13.0 queries per episode vs the parent's 12.9. There were fewer re-asks, though. Next I'll compare which menu ranks the model actually picked.Audit passes. Writing the proposal card.The new harness `agents/cand_pilot_strict_b10g3r2_b2r0g1.py` runs cleanly: 10/10 episodes finished, every one produced a final guess, and `audit.py` passed. My hypothesis held only in part. The model now reads the query menu correctly, but it doesn't use fewer queries.

**What I tested.** In every parent episode, all three circuits ended "SOLVED" before the budget ran out. So the place to look was how queries get chosen. The model took the best menu option in only 44 of 129 picks, and 21 replies were off the menu or had to be re-asked. Its stated reasons showed it reading "n1 say 1, n0 say 0" backwards (one reply picked a query for its "highest imbalance"). The other branches had already tried re-ranking and shortening the menu, so I tested whether it's the *wording* that confuses the model. Each option now carries one number where bigger is better ("rules out at least M of the N candidates"), sorted biggest first. The menu still shows 8 options, and the checks on the model's replies and the final step are unchanged.

**Results (re-run on the 10 instances):**

| | Parent | New |
|---|---|---|
| Picks of the best option | 44/129 | **80/130** |
| Off-menu or malformed replies | 21 | **0** |
| Model calls | 156 | 140 |
| Queries per episode (mean / max) | 12.9 / 17 | 13.0 / 16 |

- All 10 episodes end with every circuit pinned to a single table, and no answer contradicted the harness's reading of the rules.
- **Why query count didn't drop:** the best possible here is about 12 queries for most of these instances and 8 for the 3 AND / 3 OR one. Episodes still use 13–16. That 1–4 query gap comes from how the menu is ranked: candidates are scored one circuit at a time, ignoring how the shared gate inventory links the circuits. A better-read menu can't fix that.
- **Natural next step:** combine this wording with a ranking that scores all three circuits together, which is what the b1r1 sibling branches did.

The proposal card is at `logs/pilot_strict_b10g3r2/proposal_cards/cand_pilot_strict_b10g3r2_b2r0g1.json`. It states the miss on query count plainly.

You should know that the parent trace `.json` files and `branch_lineage.jsonl` contain answer-key fields (`gold`, `exact`). I saw them in passing while checking the file format but didn't use them for any judgement. The analysis above comes only from the `.md` traces, the queries and answers, and the task rules. Those fields are visible to any branch, despite the setup saying no evaluation is given.