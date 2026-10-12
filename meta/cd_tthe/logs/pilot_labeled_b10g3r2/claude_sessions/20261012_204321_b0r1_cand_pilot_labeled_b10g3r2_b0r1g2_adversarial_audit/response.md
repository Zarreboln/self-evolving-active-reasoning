Parent (g2) solved 10/10; now auditing whether that's robust (rule-derived hypothesis space, query efficiency) before changing anything.Running an offline simulation of the parent's query-suggestion policy across inventories; meanwhile cross-checking the other branches' observed answers against g2's final tables.Now the edits: docstring, joint weights in `_marginals`, ranking in `_state_text`, and the final fallback.A quick offline smoke test with a fake env and a scripted "model" that always takes the first suggestion, to catch crashes before the real run:Smoke test passes (40/40, even with a model that writes garbage final tables, since snapping recovers the determined tables). Now the real check against the 7B:10/10 correct, but mean queries went from 10.9 to 11.6. That contradicts the sim's prediction, so I'm checking where the model's picks went before claiming anything.The new harness solved all 10 instances (10/10 final tables correct, 3/3 circuits each, no crashes). It does not beat its parent on this batch, which also scored 10/10, and on the 2+ NOT inventories where it should help, the gain is small.

**Audit of the three branches.** g2's final tables agree with every one of the 530 answers that g0, g1 and the bare baseline got on the same instances. So g2's way of working out, from the rules alone, which circuits are still possible is independently supported. g0 and g1 failed because they never worked out which circuits were still possible, not because of any evidence that contradicts g2. Nothing from them was worth bringing over.

**What I changed in `agents/cand_pilot_labeled_b10g3r2_b0r1g2.py`:**
- **Query suggestions:** the four suggested queries are now ranked by how evenly the answer splits all the combinations still possible across the three circuits, rather than one circuit at a time. This also credits an answer on A for narrowing B and C, because all three share one gate inventory.
- **Leftover ambiguity:** if a circuit is still undecided at the end, its possible tables are listed most likely first. When the model's reply doesn't settle it, the harness falls back to the most likely table rather than the first in sort order.
- **Everything else from the parent is unchanged.** The model still writes every query, wasted queries are still rejected and re-asked, and the final answer is still built from what the model wrote, corrected to the nearest table that fits the answers.

**What the check run showed:** 10/10 correct, but 11.6 queries per episode against the parent's 10.9. The traces show why one run can't rank the two versions: the model usually takes the first suggestion but sometimes picks the second to fourth, so query counts mostly reflect that random pick.

**Matched replay:** I replayed both rankings 200 times on each of these 10 instances' tables (known now from the outcome). The new ranking needs slightly fewer queries on all 10 (11.42 vs 11.74 on average), and neither version ever ran out of queries. The parent's 10.9 was a lucky draw.

**Where it should matter:** inventories with 2–3 NOT gates, which this batch barely covers. In an offline simulation:

| Inventory | Episodes unresolved at 18 queries | Mean queries |
|---|---|---|
| 3 AND / 3 OR / 2 NOT | 1/60 → 0/60 | 15.4 → 15.0 |
| 2 AND / 4 OR / 3 NOT | 3/60 → 1/60 | 16.5 → 15.9 |
| 1 AND / 5 OR / 2 NOT | — | 13.3 → 12.1 |

This simulation assumes the model picks one of the top 4 suggestions at random, which matches the traces but hasn't been run with the real model.

The proposal card is at `logs/pilot_labeled_b10g3r2/proposal_cards/cand_pilot_labeled_b10g3r2_b0r1g2.json`. Its risks section notes that circuits with 3+ NOTs still come close to the 18-query limit. It also notes that if the rules were ever misread, the harness would stop early and confidently give a wrong table. Nothing on this batch suggests that, but other batches haven't been checked.