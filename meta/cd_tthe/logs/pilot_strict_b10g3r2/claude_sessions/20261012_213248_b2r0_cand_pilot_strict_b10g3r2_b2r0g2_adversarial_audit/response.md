I'm reading the parent harness's traces to see how its episodes end. I'm not looking at the gold labels in the trace JSON.I'm collecting each b2 instance's inventory. Next I'll simulate query policies under the rules.Now the docstring, then run the check.The two-step lookahead didn't help: same mean, worst case at most 1 query lower, and about 30× slower. I'm keeping the greedy ranking and writing the proposal card.The new harness is at `agents/cand_pilot_strict_b10g3r2_b2r0g2.py` and the proposal card is written. On the re-run all 10 instances ran without crashing, used all 18 queries, and produced a final guess. Judged without labels, all 10 look solved, though I can't confirm correctness without the truth tables.

**What I took to mean "going well":** every circuit is narrowed to one table the rules allow (pinned) with queries to spare, the final guess agrees with every answer, and independent query sequences arrive at the same tables.

**What the audit found**
- **The parent's rule model holds up.** Across its b2 traces, no answer ever contradicted the possibilities it derives from the rules.
- **The parent's weak point is the query budget.** It pinned instance 284 at 17 of 18 queries and 266 at 16. The 7B often picked weak items from its 8-item menu (one reply chose a query because it "splits most unevenly").
- **I simulated query choice under the rules.** With the model picking at random from the parent's menu, 2–9% of simulated episodes run past 18 queries when the inventory has several NOT gates. With the new menu, none of 1,200 did.
- **Where sibling branches disagreed:** some stopped once everything was pinned, others kept checking. Checking caught nothing, but it can't change the answer when the rules hold, so I kept it as insurance.

**What changed (mostly borrowed from sibling branches, plus a few fixes of my own)**
- Queries are ranked by how much they tell about all three circuits together. The model sees only 2–3 options, each worth at least 90% of the best one, and picks among them. This is my variant of the siblings' "show only the top few" idea.
- A query that can't tell anything new is never sent.
- Once every circuit is pinned, the leftover queries go to checks the model chooses.
- The parser now accepts `Circuit_C(...)` and a garbled `=query>...` tag. The garbled tag had cost a sibling several re-asks.
- Fixed a bug copied from a sibling: the "you're stuck" note showed a truncated example query.

**Re-run results**
- **Pinning:** every circuit was pinned on all 10 instances. Total queries to pin fell from 129 to 119, and the worst case from 17 to 16. Model calls dropped to 19–20 per episode (the parent used 14–23).
- **Checks:** none of the 61 check answers contradicted a pinned table.
- **Cross-check:** the parent and the new harness asked different queries but returned identical 24-bit guesses on 10 of 10. Each guess matches every answer the other harness received, and between them they directly observed 19–23 of the 24 rows per instance.
- **One regression:** instance 230 took 16 queries to pin, up from 11. The model chose the top option every turn, so this comes from the ranking itself, not a bad pick.
- **Rejected:** a two-step lookahead ranking. It didn't lower the average and was about 30× slower.

**Remaining risks**
- Inventories with 2–3 NOT gates still need up to 18 queries in the worst simulated case, so there's no slack there.
- The checks add about 5–7 model calls per episode and haven't caught anything yet.
- The pilot log files (`branch_lineage.jsonl` and the trace JSONs) include outcome fields. I didn't use any of them, but they were in files I read.