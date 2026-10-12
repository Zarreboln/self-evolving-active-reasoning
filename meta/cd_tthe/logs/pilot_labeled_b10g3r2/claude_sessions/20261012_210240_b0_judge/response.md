I chose **`cand_pilot_labeled_b10g3r2_b0r1g2`** and wrote that name to `logs/pilot_labeled_b10g3r2/choice_b0.txt`. I didn't re-run any harness; the choice is based on the source files and the existing traces.

- **Three candidates were perfect on this batch:** `b0r0g2`, `b0r1g0` and `b0r1g2` all got 100% of final tables and circuits right, against 0% / 6.7% for `bare`. They used about 11–12 queries per episode instead of 18.
- **None of them chooses queries itself.** In all three, the model writes every query. The harness only rejects repeats or queries whose answer is already settled, and asks again. At the end it snaps each circuit the model wrote to the nearest table that fits the answers. So none of them breaks the interface.
- **The logic is general, not tuned to this batch.** All three list every table each circuit could have. That list follows from the rules in the system prompt: each circuit has exactly two 2-input gates and any number of stacked NOTs, the gates are split exactly across the three circuits, and no circuit is constant. Nothing depends on these 10 instances.
- **Why `b0r1g2` among the three:** it improves on `b0r0g2` in two ways. It ranks suggested queries using all three circuits together, so an answer about A also rules out options for B and C. And if a circuit is still ambiguous at the end, it lists the most likely table first and uses it as the fallback. `b0r1g0` also works, but its chat history grows with every turn and it caps queries per circuit, which seems less robust on new instances. On this batch the three are tied, so the tie-break is my judgement about new instances, not a measured difference.

**Rejected:**
- `b0r1g1` crashed in 9 of 10 episodes.
- `b0r0g0` and `b0r0g1` got more circuits right than `bare` but almost never the full table.

`tokens_out` shows 0 for several candidates even though their traces contain model replies. That column looks like an accounting gap, so I ignored it.