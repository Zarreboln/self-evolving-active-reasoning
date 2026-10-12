I'm reading the parent's traces to find where episodes end without every circuit pinned down. I'm skipping the gold fields in the JSON files.The parent finished all 10 episodes with every circuit narrowed to exactly one candidate table. Next I'm checking for smaller problems: re-asks, rejected picks, and how the final answer was parsed.Fixing the long line, then re-running the simulation and the check.Average is 11.2 queries, but i209 and i253 rose to 16. I'll check the top-pick rate and what happened in those two.I'll update the module docstring to match, then re-run the simulation and the check.The last edit only touched the docstring, but I'll re-run the check so the verified file is the final one.Results are identical to the previous run. Writing the proposal card.The new harness runs cleanly on all 10 instances. Every episode ends with all three circuits narrowed to one table, but on this batch it doesn't need fewer queries (mean 11.1, same as the parent). The improvement is in the worst case, and that only shows up in an offline simulation, not in these 10 runs.

**What I found in the parent:** it already worked on this batch. All 10 episodes ended with every circuit narrowed to one table. Those tables agree with all ~580 answers that the parent and its two sibling harnesses got on these instances, so the parent's reading of the rules was never contradicted. I didn't use the gold fields in the trace JSON files.

The one weakness was running out of queries:
- **Model picks:** the model took the top suggested query only about a third of the time. It often chose weak splits (around 75/25 when 50/50 options were listed) or queries not on the list at all. Two parent episodes used 15 of 18 queries.
- **Simulation:** I replayed the parent's selection rule against every circuit set the rules allow. With two or more NOT gates, it needs 19–20 queries in the worst case, including for i81's inventory (1 AND, 5 OR, 2 NOT), even if the model always takes the top suggestion.

**Change** (`agents/cand_pilot_strict_b10g3r2_b1r1g0.py`):
1. Queries are ranked by how much they reveal about all three circuits together, since the circuits share one inventory. Before, each circuit's candidate tables were counted separately.
2. Only queries carrying at least 90% of the best query's information are listed and accepted. A weaker pick isn't sent; the model is re-asked and told its split, e.g. "only about 19% / 81%". The list shows the overall split next to the candidate counts.

Everything else is unchanged: the candidate tracking, re-asks for bad picks, stopping as soon as all circuits are solved, and the final-answer step. The model still writes every query.

**Verification:**
- **Simulation:** with the model picking at random from the accepted queries, worst-case query counts are 15, 17, 15, 17 and 18 for the five inventories tested, and no episode goes over 18.
- **Real check run** (repeated after the last edit, same output): 10/10 ok, a final guess parsed every time, queries used 5–16, at most one re-ask per episode, no rounds lost to repeated refusals. Final tables match the parent's on every instance and agree with every answer seen.

**Risks:**
- i253 still used 16 of 18 queries even with every pick near-best; late in an episode the best available splits are inherently uneven.
- Stricter acceptance could cause more re-ask rounds with a less cooperative model. This didn't happen here.
- I didn't simulate inventories with three NOT gates under the new rule.

The proposal card is at `logs/pilot_strict_b10g3r2/proposal_cards/cand_pilot_strict_b10g3r2_b1r1g0.json`.