Parent is 100% on both batches (b0 and b1), so I'm auditing for hidden weaknesses: query efficiency (how close the tail gets to the 18 budget) and whether the model follows the suggestions.Veto alone trims the tail slightly (NOT=2: max 19→18). I'll also test tightening the suggestion list to near-max queries, since the "top 4" sometimes includes clearly weaker ones.Now running the check on this batch's 10 instances.10/10 correct; mean queries 10.5 → 9.9. Checking in the new traces whether the vetoes actually fired and whether the model complied or the fallback kicked in.The new harness got the final table right on 10/10 instances, with every circuit right, no crashes and a parsed final guess every time. It used 9.9 queries per episode on average; the parent used 10.5. The parent was also 10/10, so this change is about margin against the 18-query limit, not accuracy.

**What I found in the parent.** Only the parent branch has traces on this batch, so there were no branches disagreeing on the same instance. Instead I replayed all 20 of the parent's episodes (from both batches) to check how informative each query was:
- Almost every query was as informative as the best one available.
- The one repeated weak spot: on 7 of the 8 instances with one NOT gate, the model's 2nd query was much weaker, about 0.29 of the best. For example, on i209 it asked `A(0,0,0)` while the harness was suggesting `B(0,0,1)`.
- Instances with NOT gates are also the ones that use the most queries (13–16 of 18).
- No episode ended with a circuit still undecided.
- The earlier branches without the parent's candidate tracking either solved 0–2 of the 3 circuits or timed out. That tracking is the part with independent support, so I kept it unchanged.

**What changed** (`agents/cand_pilot_labeled_b10g3r2_b1r0g2.py`):
- The harness only suggests queries close to the best (at least 0.97 of its information).
- If the model picks a query below 0.9 of the best, the harness rejects it and asks again, naming the suggested queries.
- If all 4 attempts in a turn are rejected, the harness sends the most informative query the model itself proposed. The model still writes every query, and it can't stall the episode.

**Evidence:**
- **Live run:** the rejection fired 5 times, only on one-NOT instances, and the model chose a suggested query on the retry every time. The last-resort path never ran.
- **Per instance:** query counts on the one-NOT instances went 16→14, 14→11, 15→12, and 7→12. That last one got worse because different answers sent it down a different path.
- **Offline simulation:** I simulated the model as usually taking a suggestion and 15% of the time any allowed query. Average queries dropped slightly on the gate mixes in this batch. On a harder mix this batch doesn't contain (2 AND, 4 OR, 3 NOT), episodes running past 18 queries dropped from 5/250 to 0/250.

**Caveats:**
- One 10-instance run is too noisy to prove the 0.6-query gain; the main support is the simulation.
- The last-resort path is untested live.
- Retries add 1–3 model calls on affected episodes.

The proposal card is at `logs/pilot_labeled_b10g3r2/proposal_cards/cand_pilot_labeled_b10g3r2_b1r0g2.json`.