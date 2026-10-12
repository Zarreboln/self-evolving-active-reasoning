# Iteration 0 (seeds), run so1: bare vs enum15, v2 search split, 100 instances x 2 seeds = 200 episodes each

**What changed.** `bare` appends nothing. `enum15` appends the count of remaining configurations, the determined outputs ("do not ask these again") and, at 15 or fewer configurations, the joint 24-bit tables.

**What moved (bare -> enum15).** uniq@end 19.5% -> 38.0% (CI 31-45), exact 0% -> 6.0%, eff 0.550 -> 0.604, zero% 37.3 -> 31.3, rep% 2.2 -> 4.4, ill% 0.4 -> 0.3, parsed 87% -> 97%. Tokens per episode: input 32k -> 63k, output 2.2k -> 2.9k. One enum15 episode crashed on context overflow (115/0: prompt went past 15,184 tokens at turn 17; the server has a 16,384-token window). By inventory |H| = 252 / 2886 / 3924, enum15 scored 53 / 35 / 26% and bare 18 / 27 / 14%.

**What the traces show.**
- Both open with A(0,0,0) in 200/200 episodes. After that the model asks the next entry in canonical order on 86% of legal turns under bare and 73% under enum15.
- The named negative list half-works. When the next entry of the scan is already determined, bare asks it 87% of the time and enum15 56%. But when enum15's model does skip, it lands on another determined entry 28% of the time and on an open one only 15%. For comparison, open1's `open_entries` (a positive list of open outputs, with the instruction to choose from it) left this at 87%, the same as bare. The model checks its planned query against named entries; it does not pick from lists.
- Zero-info share by feasible-set size under enum15: >100: 14%, 16-100: 30%, 6-15: 51%, 3-5: 38%, 2: 79%.
- Key 13/1: circuit A is fully listed as determined after turn 5. The model still asks A(1,0,1), A(1,1,0) and A(1,1,1) "to further confirm the pattern", then B(0,1,1) through B(1,1,1), all determined. That is 8 zero-info turns, and 4 configurations remain at the end.
- Key 2/0: 3 configurations are left, and only B(1,0,1), B(1,1,0) and B(1,1,1) are open. The model decides "B could be a simple NOT gate" and moves to C (all determined). It then asks C(0,0,0) again and A(0,0,0) five times, and every reply from turn 13 on ends in a runaway `<guess>1111...`. That is 8 zero-info turns in a row. 8.2% of enum15 replies contain a `<guess>`; these runaways are what push the context toward the limit.
- Final step: only 5 of the 76 enum15 uniq episodes are exact; the other 7 exact hits are lucky picks from a listed set. Only 25/194 enum15 guesses (7/174 bare) even agree with the model's own observations.

**Takeaway.** Feasible-set collapse is lost to determined entries that the scan walks into, and a list the model has to consult does not stop the scan. The information has to arrive at the moment the model commits: veto the determined proposal and re-ask (in state_only the critic explicitly allows vetoing repeats and determined entries). A CPU replay of the scan habit gives 92-96% uniq@end with a 2-re-ask veto, even when the model ignores 40% of the vetoes. The final answer has to be presented already assembled.
