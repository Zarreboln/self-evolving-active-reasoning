Auditing: all three siblings score 10/10, so I'm comparing query counts on instances where they diverge (129, 195, 236, 253) to find what reduces the tail.Edits are in. Next I'll unit-test the joint snap offline on an ambiguous state, then run the check script.10/10 correct with the same queries as the parent (mean 9.9). On i228, calls dropped from 15 to 14 because the fenced tag now parses. Running the audit and confirming in the trace:I've written `agents/cand_pilot_labeled_b10g3r2_b1r1g2.py`. It solved all 10 instances (30/30 circuits right), passed `audit.py`, and used the same queries as its parent on every instance (mean 9.9). One caveat: the main new safeguard never fired on this batch, because every episode was fully pinned down before the budget ran out.

**What the traces showed.** All three siblings solved all 10 instances. Query counts on the same instance still differ, e.g. i253 took 15, 15 and 12 queries. I scored every query against the most informative query available at that point. The parent (g2) sent the best available query every time except once, at 0.97 of the best. g0 and g1 let through some weaker ones (0.69–0.97 of the best). So the different counts come from how ties were broken and from which answers came back, not from one sibling choosing queries better.

**What I took or added:**
1. **From g0: a looser query-tag match.** The model often writes a tag with the opening `<` missing (```` ```query>C(0, 1, 1)</query> ````); this happens 13 times in the sibling traces. In g2 this caused an extra model call on i228. That instance now takes 14 calls instead of 15, with no format re-asks.
2. **New: the final guess must fit one gate split.** Previously, if a circuit was still ambiguous at the end, each circuit's table was matched to the model's answer separately. The three choices could then form a combination that no way of dividing the gates across A, B and C allows, which is certain to be wrong. Now the harness picks the closest combination that a single split explains. I tested this offline on 450 random ambiguous states: matching each circuit separately gave an impossible combination 277 times; the new method gave 0.
3. **Kept everything else from g2:** query ranking and suggestions, vetoing weak queries, the fallback, and the stopping rule.

**What I tried and dropped.** I simulated a version that looks two queries ahead to break ties. It saved at most 0.2 queries on average and did not shorten the longest runs, so I didn't adopt it.

**Remaining risks:**
- The simulation puts the tightest margin on inventories with two NOT gates: 3 AND / 3 OR / 2 NOT sometimes needed all 18 queries. That's where the joint-split guess would matter.
- On every 1-NOT episode, the model's second query (always A(0,0,0)) is still vetoed. This costs one extra model call but no queries.

The proposal card is at `logs/pilot_labeled_b10g3r2/proposal_cards/cand_pilot_labeled_b10g3r2_b1r1g2.json`.