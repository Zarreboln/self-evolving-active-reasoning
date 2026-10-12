The new harness solved all 10 instances, but it adds nothing new. On this batch it behaves exactly like sibling b2r0g0: same queries on every instance, 11.5 per episode on average. That's because the only improvements the traces actually support are the two b2r0g0 already made.

**What the audit found.** All three branches solved 10/10, so they only differ in how many queries they used and how often the model had to be re-asked.
- **b2r0g0's changes work.**
  - It lists only the inputs the harness would accept for each circuit. That removed the wasted pick of `A(0, 0, 0)` that cost the parent and b2r0g1 one re-ask each on i230 and i266.
  - It accepts a query written as a ```` ```query ```` code block. On i60 that saved a re-ask.
  - It has no re-asks anywhere.
- **b2r0g1's headline change isn't in its code.** It says it spends leftover queries checking rows the rules already determine, but the loop still stops once nothing is uncertain. Its queries match the parent's on all 10 instances, because the same prompt always gets the same cached reply. Its other change, a looser fallback if an answer contradicts the rules, never triggered in any trace. I left both out.
- **b2r0g0 doesn't use fewer queries on every instance.** It used more than the parent on i171 (14 vs 12) and i173 (7 vs 6), and fewer on four others. The differences look like noise from the model choosing between nearly equal queries, so I don't claim it saves queries.

**What I changed.** `agents/cand_pilot_labeled_b10g3r2_b2r1g2.py` is the parent plus b2r0g0's two changes. Its docstring records why each sibling change was or wasn't taken.

**Verification.**
- **Check script:** 10/10 ran without error and produced a final guess. The final table was correct on all 10, with 3/3 circuits each.
- **Query replay:** the query sequences match b2r0g0's traces exactly, with no re-asks. On every instance all three circuits were fully determined before the final prompt, so no guess depended on picking among leftover candidates.
- **Offline simulation:** I used synthetic circuits and a mock model that mostly picks the top suggestion. Inventories with three or fewer NOT gates were always solved. With four or more NOT gates, roughly 20–30% of episodes needed 17–18 queries and about 1 in 140 ended wrong. That limit is the same for the parent and this harness, and nothing here improves it. It's the main remaining risk.

The proposal card is at `logs/pilot_labeled_b10g3r2/proposal_cards/cand_pilot_labeled_b10g3r2_b2r1g2.json`. I also saved a memory note: because replies are cached per prompt, a candidate whose traces exactly match its parent's made no change that reached the model.