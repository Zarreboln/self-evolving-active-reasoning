The parent already scores 10/10 on this batch, so my next check is whether its rule enumeration actually matches the task rules. Reading the system prompt now.Copied the parent into the new candidate file; now adding the information-floor veto.Offline: a single low pick costs 0.3–0.6 queries on average, and on a 4-NOT inventory it raises the share of episodes that run out of budget from 9/150 to 15/150. Writing the proposal card.I've made the new harness and re-run it on this batch: all 10 episodes still end with the correct 24-bit table (3/3 circuits each), with no crashes. Because the parent was already 10/10 here, this batch can't show a gain in correctness. The improvement is in how the queries are chosen, and should only matter on harder batches.

**Where I think episodes go wrong.** The parent ranks queries by how much information each answer would give, and that ranking is already about as good as it gets. In an offline simulation, a different ranking (minimising the worst-case number of queries still needed) gave no better worst case on any inventory I tried. So I left the ranking alone. The weak point is the 7B model ignoring the suggestions. In all four 1-NOT episodes, its second query was A(0,0,0), worth only 0.29 of the best available query, taken from the per-circuit list instead of the suggested ones. In the offline simulation, one such pick costs about 0.3–0.6 queries per episode. On an inventory of 3 AND, 3 OR and 4 NOT gates, it raises the share of episodes that run out of queries from 9/150 to 15/150. The simulation assumes every consistent set of three circuits is equally likely, which the real generator may not do.

**The change.** It's about 20 lines in `agents/cand_pilot_labeled_b10g3r2_b1r0g1.py`. A query the model writes is only sent if its information is at least 0.75 of the best available. Otherwise the model is told the answer is nearly predictable and asked again. On the 4th ask in a turn the threshold is dropped, so the model always writes the query that gets sent and a turn never stalls. Everything else is the parent's behaviour.

**Re-run results:**
- **Query quality:** every accepted query is now at least 0.93 of the best (the parent had 0.29 and 0.69 picks). The model was re-asked 6 times across the batch and picked a top suggestion each time.
- **Queries used:** 10.2 per episode vs. the parent's 10.5. Individual episodes moved both ways (instance 228 went from 7 to 12, instance 209 from 16 to 14), because the count depends mostly on which answers come back. Treat these numbers as noise, not evidence.

**Risks:**
- The 0.75 threshold is a guess.
- Each re-ask costs one extra model call.

The proposal card is at `logs/pilot_labeled_b10g3r2/proposal_cards/cand_pilot_labeled_b10g3r2_b1r0g1.json`.