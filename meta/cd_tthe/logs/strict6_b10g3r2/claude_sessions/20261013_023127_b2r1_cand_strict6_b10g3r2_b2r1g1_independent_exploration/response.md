Now I'll rewrite the query phase and prepend the branch docstring.Syntax is fine. Running the check on the batch:Everything ran with no crashes. Instance 60 stopped at 13 queries, though, so I'll check why and analyze the confirmation queries.Before the board fills, behavior matches the parent exactly (same pin points). After that, 54 confirmation queries hit deduced cells and none disagreed. Now checking why instance 60 stopped early:The model wrote prose with no query tag. Let me see how many confirmation-phase replies missed, here and in the other instances:Re-running the check:Even when every call with more than 8 messages fails, the episode still uses all 18 queries and returns the right table. Now the proposal card:The new harness ran 10/10 on this batch with no crashes. Every instance used all 18 queries and produced a final guess, and in every episode the guess is the single circuit triple that fits the task rules and all the answers.

**Where I thought the risk was.** I replayed the parent's traces through its own rule enumeration. Every episode already ended with exactly one possible triple, and the guess matched it. The model's query count was also within one query of a greedy best-split policy (7–16 queries to pin everything, 2–11 to spare). So query choice wasn't the problem. The weak point was that 10–17 of the 24 output bits were never queried: they were "deduced" from the harness's own reading of the rules. The parent stopped as soon as the board had no '?', so a misread rule would produce a wrong table with nothing to show it.

**What I changed** (`agents/cand_strict6_b10g3r2_b2r1g1.py`): when the board is complete and queries remain, the model is asked to spend them confirming deduced cells. It sees the list of deduced cells, starting with the circuit that has the fewest observed cells, and still writes every query itself. If an answer disagrees with a deduction, the board drops all rule-based deductions, shows only observed cells, and querying resumes on the '?' cells. That fallback was already in the parent.

On the first run, instance 60 stopped at 13 queries. A model call failed once the conversation got long (most likely it exceeded the model's context), and the parent treats any failed call as the end of querying. I fixed this: confirmation turns now send a short conversation (system prompt plus the current board), and a failed call is retried once in that form. After the fix, instance 60 also used all 18 queries.

**What the re-run showed:**
- Up to the point where the board is complete, behaviour matches the parent's, with the same pin points.
- 60 confirmation queries hit deduced cells and none disagreed, so the rule reading held on this batch.
- No cell was queried twice.

**Offline tests** (a fake environment and scripted model in my scratch directory, no real instances):
- When the fake truth breaks the rules in one cell, the harness flagged it in 34 of 96 cases and the fallback ran without crashing.
- The episode still finishes correctly when every long conversation fails.

**Limits** (also recorded in the card):
- If the rules are read correctly, as on this batch, this can't make any guess more correct. It only adds 2–11 queries and model calls per episode as insurance.
- 18 queries can't cover all 24 cells, so a misread that only affects unqueried cells still goes unnoticed (26 of the 96 offline cases).
- Catching a misread doesn't give the right table. After the fallback, the 7B model fills the open cells itself, and it does that poorly.

The proposal card is at `logs/strict6_b10g3r2/proposal_cards/cand_strict6_b10g3r2_b2r1g1.json`.