Copied the parent; now editing the docstring, candidate weighting, and query ranking.Final step: tell the model candidates are listed most likely first. Then run the check.The prompts read correctly. Writing the proposal card.The new harness ran on all 10 instances without crashing. Every episode ended with exactly one candidate table per circuit, using 8–15 queries (mean 11.4). The parent used up to 18, and one episode ran its whole budget.

**Comparing the three branches on the same instances (without using the truth tables):**
- **bare, g0 and g1** each sent 3–11 queries per episode whose answer was already determined. bare's and g0's final guesses often contradicted answers they had received themselves. g1's guesses matched its answers but left 4–9 possible tables for some circuits, so the final table was partly a guess.
- **g2 (the parent)** narrowed every circuit to one table on all 10 episodes. The best evidence that it's right: its 10 final tables agree with all 524 answers the other three harnesses got on the same instances.
- **g2's weak spot** was running out of queries. It needed 18/18 on i250 and 17/18 on i247. It also sometimes spent a query on the model's useless fallback pick (e.g. A(1,1,1) on i156).

**Changes in `agents/cand_pilot_strict_b10g3r2_b0r1g2.py`:**
1. **Ranking:** queries are now ranked by how evenly they split the possibilities for A, B and C together, since the circuits share one gate inventory. Finding the NOT gate in A, for example, rules it out for B and C. The parent ranked each circuit separately.
2. **Shorter menu:** the model now sees only the top 3 queries instead of 8. It picked a listed query on the first ask in every episode.
3. **No wasted queries:** if the model never names a useful query, nothing is sent to the environment and it is asked again.
4. **Final pick:** the remaining candidates are listed most likely first.

**Evidence:**
- **Simulation:** I generated 60 random circuits per inventory from the task rules. With the parent's ranking, inventories with more NOT gates needed more than 18 queries in up to 8 of 60 episodes; with the new ranking, none did.
- **Re-run:** no queries were wasted. The final tables are the same as g2's on all 10 instances and contradict none of the 650 answers the other four harnesses received.

**Risks:**
- The whole approach depends on reading the rules as "exactly two 2-input gates per circuit, plus NOT gates". If the generator allowed something else, the right table could be missing from the candidates. The harness only notices if an answer outright contradicts every candidate, which never happened.
- With 3 or more NOT gates, simulated episodes still average about 15 queries, so the margin is thin.

The trace JSONs and `branch_lineage.jsonl` include a gold/exact correctness field. I didn't use it; I judged correctness only from consistency with the answers and agreement across branches. You may want to remove that field, since it breaks the "no evaluation" setup.

The proposal card is at `logs/pilot_strict_b10g3r2/proposal_cards/cand_pilot_strict_b10g3r2_b0r1g2.json` and is valid JSON.