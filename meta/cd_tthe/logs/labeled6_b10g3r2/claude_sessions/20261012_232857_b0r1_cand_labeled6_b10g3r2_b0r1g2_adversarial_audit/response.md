Status: the parent (g2) solved all 10 instances, using at most 15 of 18 queries. I'm now comparing it with the g0/g1 traces and checking how much budget margin it has to spare.Now the edits: a "best rows" list in the prompt, a veto on clearly less-informative queries, and a better fallback.10/10 correct. Next I'm checking the new traces for how often the veto fired and whether any accepted query was below the best split.No truth tables in it, only outcome flags. Checking veto activity:The veto fired only twice, and both times the model's re-asked query was accepted. Every episode ended with exactly one table left. Writing the proposal card.I re-ran the new harness `cand_labeled6_b10g3r2_b0r1g2` on this batch: it got the joint truth table right on all 10 instances (3/3 circuits each), with no crashes. It used 7–15 queries, averaging 11.6 against the parent's 12.0. Because the parent (g2) was already at 10/10, this change mostly guards against failures on harder inventories; it does little for this batch.

**What the traces show when the branches disagree**
- **g0 and g1 (53.3% of circuits right, 0% joint):** they make the model guess the rows it never queried. The 7B can't work out the wiring from the answers it has, and that's where they fail. I didn't carry this over.
- **g2 (100%):** it works out from the task rules every A/B/C combination that could exist, and drops the ones each answer rules out. That's why it wins. But the model is not choosing its queries well: it ignored the top-suggested query in 52 of 121 turns. Often the row it picked instead was equally informative, but not always.

**Is that a real risk?** I simulated it offline using only the parent's own rule-based enumeration, with no instance data. If the model picks any still-open row at random, it runs out of the 18 queries with more than one candidate left in 1.7–17.7% of episodes on the larger rule spaces (more NOT gates). If it only picks among the most informative rows, that happened in 0 of 300 runs for each.

**Changes in b0r1g2:**
1. The prompt now lists every row tied for most informative and says the model can pick any of them.
2. A query on a clearly less informative row is rejected and the model is asked again, without spending a query. The model still chooses which of the best rows to ask.
3. If the model never complies, the harness falls back to its last query that still tells it something, rather than one on a row that's already known.
4. Everything else from the parent is unchanged: the enumeration, the board, early stopping, and the checks and repair on the final guess.

**Verification:** the rejection fired only twice in the run (instances 11 and 239). Both times the model's next query was one of the best rows. Every episode ended with exactly one candidate left. I also tried ranking queries by looking two steps ahead, but it made one inventory worse, so I left it out.

**Limits:** the robustness claim for larger rule spaces comes from the simulation, not from real episodes. With 3 or more NOT gates, even perfect play sometimes needs more than 18 queries. In that case the model picks the final answer from the remaining options.

The proposal card is written to `logs/labeled6_b10g3r2/proposal_cards/cand_labeled6_b10g3r2_b0r1g2.json`.