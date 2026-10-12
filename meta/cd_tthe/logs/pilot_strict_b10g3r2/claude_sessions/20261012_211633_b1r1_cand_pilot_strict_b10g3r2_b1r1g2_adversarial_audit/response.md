Now the edits: docstring, the joint split counts on `Space`, the ranking, the menu, the re-ask loop, and the solve loop.Edits are in. Running the check script on the 10 instances (live Qwen-7B):Nothing crashed, but i236 is a real failure: only 10 queries sent and 86 model calls. The model got stuck re-asking until the wasted-round limit stopped the loop. Looking at why:Case sensitivity is still blocking `Circuit_C`, so I'm making the prefix match case-insensitive:Parser fixed. Before re-running, I'm checking how the solver cache keys its calls, to see whether a re-ask with an identical prompt can ever get a different reply:I've built `agents/cand_pilot_strict_b10g3r2_b1r1g2.py` and re-ran it on this batch. All 10 instances ran without crashing and produced a final guess using all 18 queries. Every guess agrees with all 18 answers in its episode, and with the guesses from the four earlier branches.

**Audit: the branches don't disagree on any answer.** b0r0g2 and the three b1r0 branches give the same 24-bit guess on all 10 instances. Across the roughly 70 extra check queries in each of g1, g2 and my run, no answer contradicted the rule-derived deduction. So they differ only in how efficiently they spend queries, and that's where the risk is. Instances with NOT gates took 13–15 of the 18 queries just to pin every circuit down, and in simulation some NOT-heavy inventories need the full 18.

**Strengths each branch's own traces support, which I kept:**
- **From g0:** never send a query known to carry no information; re-ask instead and list the picks already rejected. g0 wasted none, while g1 and g2 each wasted queries on 2–3 instances.
- **From the g2 parent:** spend the leftover budget checking the deduction. It never changed an outcome here, but it costs nothing.

**New in this harness:**
- **Joint ranking, short menu.** Queries are ranked by how evenly they split the space of all three circuits together, which are linked through how the inventory is split. The model now sees only the top 3. With the old 8-item menu it took the top entry about 40% of the time; with the new one it took it on 103 of 103 parseable first replies.
- **Parser fix.** My first run stalled on i236 at 10 queries and 86 model calls. The model kept writing `<query>Circuit_C(0,0,1)</query>`, copying the benchmark's own template, and the old parser rejected it. That form is now read as `C(0,0,1)`.
- **No identical re-asks.** If a round yields no usable query, the next prompt names the rejected replies. The model runs with a fixed seed, so an identical prompt tends to get the identical reply.

**What the numbers show:**

| | Queries needed to pin all 3 circuits (summed over 10 instances) | Worst instance | Wasted queries |
|---|---|---|---|
| Parent (b1r0g2) | 111 | 15 | 2 |
| This harness | 107 | 14 | 0 |

- **On this batch the gain is modest and mixed:** 6 instances needed fewer queries, but i130 went from 6 to 10 and i195 and i228 from 10 to 12, all still within budget. Because the model now almost always takes the top entry, the ranking itself is effectively choosing the query order.
- **In simulation** (sampled rule-allowed tables, model taking the top entry only 40% of the time), the new ranking lowers the average by about 1–1.5 queries on inventories with NOT gates. It also removed the 2 out of 40 failures to pin within 18 queries on the 3-NOT inventory.
- **Remaining limit:** some 3-AND/3-OR/2-NOT cases still need 17–18 queries, so there's little slack there. A 2-step lookahead ranking gained at most 0.2 queries for 10× the compute, so I left it out.
- **Offline smoke test:** with a deliberately unreliable fake model over 40 episodes, nothing crashed and every rule-following episode came out exact. The only misses were the 10 tables I made deliberately break the rules.

One disclosure: while listing the keys of a trace JSON, I accidentally saw the `gold` result for one instance (i107). I didn't read `gold` or `signals` after that, and they played no part in any decision. This is noted in the card.

The proposal card is at `logs/pilot_strict_b10g3r2/proposal_cards/cand_pilot_strict_b10g3r2_b1r1g2.json`.