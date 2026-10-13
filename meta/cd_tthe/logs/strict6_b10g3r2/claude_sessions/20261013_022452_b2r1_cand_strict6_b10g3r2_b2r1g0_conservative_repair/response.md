Copied the parent; now I'm making the edit.Edit is in place. Running the check script on the 10 instances:No crashes, every instance produced a guess, and instance 60 went from 10 queries to 9. Next I'm reading the new traces to confirm that the board ended complete and that no query was wasted.Verified across all 10 episodes: the board finishes with no '?' cells, exactly one triple fits the rules, the guess matches it, and no query was wasted. Checking the instance 60 exchange directly:I made one small change to the harness: when the model asks about a cell that's already known, the message rejecting that query now lists the cells that are still unknown. On the re-run, all 10 instances finished without crashing and returned a final guess.

**How I judged success:** there's no correct-answer signal, so I checked each episode against the task rules (how the gate inventory can be split across the circuits, no constant circuits). I replayed every query history and counted the circuit triples still consistent with all answers. The parent already reached exactly one consistent triple in all 10 episodes, and its guess matched that triple every time.

**Failure I found in the parent:** in instance 60, circuits B and C were fully worked out and only two cells of A were unknown. The model kept asking about B anyway. The parent blocks a query on a known cell twice in a row, but its message only said "choose a cell marked '?'" without naming any. After two blocks the model's third known-cell query went through to the environment, wasting one of the 18 queries.

**Fix:** in `agents/cand_strict6_b10g3r2_b2r1g0.py`, the rejection message now names the unknown cells, the same way the parent's early-guess message already does. Nothing else changed: the limit of two blocks is the same, and so are the knowledge board and the final phase.

**Re-run results (check script plus my replay of the new traces):**
- Every episode still ends with exactly one consistent triple, and the guess matches it.
- No query was wasted on an already-known cell (the parent had one).
- Instance 60: the rejection listed "A(1,0,1), A(1,1,1)", the model asked for A(1,0,1) next, and it finished in 9 queries instead of 10.
- Rejected queries across the batch fell from 12 to 7. Average queries per episode went from 13.2 to 13.1.

**Remaining risk:** instance 266 still uses 17 of 18 queries, because the last circuit can only be narrowed down one option at a time. A harder instance could run out of queries; when that happens the final phase lists the tables still allowed and asks the model to pick, which may not be right. Also, "correct" here only means consistent with my reading of the rules.

The proposal card is at `logs/strict6_b10g3r2/proposal_cards/cand_strict6_b10g3r2_b2r1g0.json`.