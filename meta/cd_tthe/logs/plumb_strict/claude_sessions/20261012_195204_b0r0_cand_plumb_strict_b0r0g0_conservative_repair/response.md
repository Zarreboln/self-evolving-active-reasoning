So far: the solver in this run is a scripted mock (`sequential`), so I'm checking which inputs it actually reacts to before I change anything.All 5 ran. Now comparing the final guesses with the bare harness's, and running a unit test of the repair path (re-asking for the final guess), which this mock never triggers.The candidate harness runs cleanly on this batch, but this run can't show whether it helps the model deduce the circuits. The batch's "frozen 7B" is a scripted `sequential` mock (`runner/cd_mock.py`), not a model. It asks for rows in order and ignores prompt text, so the candidate's final guesses are bit-for-bit the same as `bare`'s on all 5 instances.

**What goes wrong in the `bare` traces:** none crashed, every guess parsed and no query was repeated. The real failure is how the 18 queries are spent. The model reads A's 8 rows, then B's 8, then gets to only 2 of C's. So 6 of C's 8 cells in every final guess are unobserved 0s. Asking for rows one by one can't work, because 18 queries can't cover 24 rows; the gate inventory has to fill the gap.

**Changes in `agents/cand_plumb_strict_b0r0g0.py`:**
1. **Each turn's prompt** now adds the number of queries each circuit has had, a note that the remaining budget can't read every row (so the inventory must be used to infer the rest), and which circuits haven't been queried yet.
2. **The final-guess prompt** lists every answer received, grouped by circuit.
3. **The final guess is checked against the answers.** If it's missing or contradicts any answer, the model is asked once more, with the contradictions listed. Cells the environment actually answered are then filled in from those answers; every other cell is the model's.

Everything else is the same as `bare`: the benchmark wording, the model writing every query, a malformed turn still using up its turn, and the budget handling.

**What I checked:**
- **The check script:** all 5 instances ran (`status=ok`), each produced a parsed final guess, used 18 queries and made 19 model calls.
- **The trace for instance 11:** the new per-turn note and the final list of answers appear as intended.
- **A unit test with a fake environment:** this mock never exercises the re-ask path, so I tested it separately. A contradicting guess is re-asked and then corrected from the answers, an unparsable one is re-asked, and if both attempts fail the model's raw text is returned without crashing.

**Main risk:** the fix for C's missing rows is only a prompt change. A real 7B model may ignore it, or use the inventory to make wrong guesses about rows it should have queried instead. To find out whether it actually changes query allocation, run it with the real solver and compare how many episodes leave some circuit with 2 or fewer queries.

The proposal card is at `logs/plumb_strict/proposal_cards/cand_plumb_strict_b0r0g0.json`.