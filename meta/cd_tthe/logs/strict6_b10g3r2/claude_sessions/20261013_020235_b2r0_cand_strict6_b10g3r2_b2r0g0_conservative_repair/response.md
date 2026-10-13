I've made the edit. Next I'll unit-test the fallback with a mock model that always miscopies, then run the check script.The fallback test passes: even when the model always miscopies, the guess equals the board. Now I'm running the real check on the 10 instances.The re-run matches the parent exactly, as expected, since the solver is cached and deterministic and the new fallback never had to fire. Now I'll write the proposal card.I made one small fix, and the new harness gives the same guesses as the parent on this batch. On this batch the fix doesn't change any output; it's a safety net for a failure the traces show nearly happening.

**What the parent's traces show.** I re-checked all 10 episodes against the harness's own reading of the rules. Every episode ends with exactly one possible (A, B, C) combination left, and the returned guess matches it. There were no crashes, episodes used 6–17 of the 18 queries, and only one query told the model nothing new (instance 60). By the rules alone, all 10 guesses are fully determined; I can't check them against the hidden tables.

**The weakness I fixed.** At the end, the model is asked to copy each circuit's row from the board. On the first try it miscopied 5 of 30 rows, for example writing `00011111` where the board said `00010011`, the same slip on instances 0, 230 and 266. A re-ask fixed each one. But if all 4 tries had failed, the parent would have returned the model's last answer, which contradicts known answers.

**The change** (`agents/cand_strict6_b10g3r2_b2r0g0.py`). That last fallback now never contradicts the board. It keeps every observed or deduced value and takes only the still-unknown cells from the model's last table. Everything else is unchanged from the parent.

**How I checked it**
- **Mock test:** with a fake model that always miscopies in the final step, the new harness still returns the correct board; the parent would have returned the wrong rows.
- **Check script:** 10/10 runs completed with a final guess. Query counts (13.2 per episode on average) and model calls match the parent. Since the model's replies are cached, the run followed the parent's path exactly, and the new fallback never fired.

**Risks**
- The benefit isn't measured on this batch. It only matters when the model miscopies 4 times in a row.
- Every "deduced" value depends on how the harness reads the building rules. If the environment reads them differently, the fallback would keep a wrong deduced value instead of the model's.
- After 2 rejected queries, a query on an already-known cell is still sent. That produced the one wasted query, but the budget never ran short, so I left it alone.

The proposal card is at `logs/strict6_b10g3r2/proposal_cards/cand_strict6_b10g3r2_b2r0g0.json`.