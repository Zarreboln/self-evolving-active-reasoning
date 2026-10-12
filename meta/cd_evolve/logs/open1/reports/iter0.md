# Iteration 0 (seeds): bare vs enum15, search split, 100 episodes each

**What changed.** `bare` appends nothing. `enum15` appends the count of remaining configurations, the list of determined outputs ("do not ask these again"), and the joint 24-bit tables once there are 15 or fewer.

**What moved (bare -> enum15).** uniq@end 26% -> 53% (CI 43-63), exact 0% -> 5%, eff 0.558 -> 0.629, zero% 36.9 -> 29.1, rep% 1.8 -> 3.5, ill% 0.3 -> 0.2, parsed 86% -> 97%. 2 enum15 episodes crashed (context overflow, both counted as failures), 0 for bare. Mean cumulative input tokens went from 31k to 63k.

**What the traces show.**
- Both open with A(0,0,0) every time (the harness cannot speak before query 1). After that, bare goes to the next entry in canonical order 92% of the time; enum15 does so 73% of the time.
- enum15: the model copies the determined list into its reasoning ("We know B(0,0,0)=0, B(1,1,1)=1") and sometimes skips those entries (key 275/1). But it does not reliably obey the negative instruction: with exactly 2 configurations left, 76% of its queries are zero-info (key 1/0: from turn 11 on it scans C(0,0,0)..C(1,0,1), all listed as determined, while B(1,0,0) would have decided between the two). With 6-15 left, 47% are zero-info.
- The joint enumeration gets misread: in key 1/0 the model treats each 24-bit joint table as one circuit's truth table.
- Final step: only 4 of the 53 uniq episodes are exact. The model lists all 24 values correctly, then concatenates them in the wrong order (key 140/1). enum15 stops listing configurations once only one is left, so there is nothing to copy. Only 4/53 guesses even agree with the model's own observations.
- Both crashes (keys 12/0, 79/1): from turn 11-13 on, the model put a `<guess>` into every reply (~1,600-1,850 chars each), on top of ~600-char harness messages, so the prompt for query 18 went past 11,088 tokens (12,288 minus the 1,200 reserved for output).

**CPU check of what limits uniq@end (search split, first query fixed to A(0,0,0)).** Canonical scan: 20%. Canonical scan that skips determined entries: 96%. Random undetermined entry: 98.8%. Max-entropy greedy: 100% (mean 12.0 queries, max 16). Under a directive that is copied with probability p, greedy stays at 98-100% for p >= 0.7 when the misses are scan moves, but drops to 89% / 72% / 44% at p = 0.8 / 0.7 / 0.6 when the misses are wasted turns. A 1-step lookahead recommender gains <= 1 point over greedy at every p.

**Takeaway.** The 7B model's choice among informative entries is good enough; what costs uniq@end is asking entries that are already determined. A short positive list it can copy from (or a single copyable action) should fix that, while a negative list and joint tables it has to reason over do not. The final answer should be handed over as a ready `<guess>` string, because the model cannot assemble 24 bits.
