Parent harness is 100% correct on all 30 episodes; I'm now checking the traces for smaller inefficiencies such as weak picks, re-asks, and how close episodes come to the budget.Parent is 100% correct, but an offline replay of its query ranking shows some episodes using up to 15–17 queries. I'm tracking down why it sometimes picks lopsided queries.Now running the check on the 10 instances.All 10 correct with zero re-asks (calls = queries + 1). Mean queries is 10.7, about the same as the parent's 10.6, but instance 81 went from 11 to 14, so I'm checking whether my change caused that.I made one small change: the model now gets the right table on 10/10 instances with no re-asks. Mean queries are about the same as the parent (10.7 vs 10.6), so the saving is in calls to the model, not queries.

**What the traces showed.** The parent (b1r0g0) already got the table right on all 10 instances, and the other branches did too. The one failure that kept coming back was on every episode whose inventory includes 1 NOT gate (4 per branch). The model's second query was `A(0, 0, 0)`, the last item in A's "still uncertain inputs" list and nearly worthless: 0.28 bits against 0.95 for the best query. The weak-query rule rejected it every time, costing an extra call to the model. The parent had 4 of these re-asks, b1r0g1 had 7 and b1r0g2 had 6.

**The change** (`agents/cand_pilot_labeled_b10g3r2_b1r1g0.py`, in `_state_text`): each circuit's "still uncertain inputs" now lists only rows the weak-query rule would accept, meaning at least half the information of the best open query. When rows are dropped, a short note says the rest are nearly predictable. Everything else is the parent's code: the query ranking, the rejection rules (the weak-query one stays as a backstop), the final-answer step and the tag parsing.

**Results from the check script and `audit.py`:**
- **Correctness:** 10/10 ran without errors, produced a final guess, got the table right and had 3/3 circuits right. `audit.py` reports OK.
- **Re-asks:** none. Calls equal queries + 1 on every instance, and the model now goes straight to a suggested query at that turn.
- **Queries:** 6–14 per episode, mean 10.7 (parent 10.6, max 15).
- **Instance 81** went from 11 to 14 queries. Every query it asked was a top-ranked one, so the extra queries come from the answers it got, not from bad choices.
- **One new pattern:** on instances 209, 228 and 253 the model picked `C(0, 0, 0)` at the third query, worth 0.78 of the best. That's slightly suboptimal but passes the rule.

**How close episodes get to 18.** The query count is mostly set by the puzzle. With 3 AND and 3 OR gates, each circuit has six possible mixed tables, and every input rules out at most one of them. So up to 15 queries can be unavoidable.

I replayed the ranking offline on 300 sampled sets of tables for each inventory. The worst cases were 15 queries for 3 AND/3 OR, 16–17 with 1 NOT added, and 16 for 1 AND/5 OR/2 NOT. That's always under 18 but with little margin, and this change doesn't affect it.

The proposal card is at `logs/pilot_labeled_b10g3r2/proposal_cards/cand_pilot_labeled_b10g3r2_b1r1g0.json`.