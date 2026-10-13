I'm comparing final guesses across harnesses per instance and checking whether each ended with the board fully pinned.All six harnesses produced identical guesses on all 10 instances, and each guess matches every answered query. Next I'll check the truncated diffs (b2r0g2, b2r1g0) for interface compliance.I picked **`cand_strict6_b10g3r2_b2r0g1`** and wrote it to `choice_b2.txt`.

**What the traces show**
- **No difference in accuracy on this batch.** All six harnesses gave the same final guess on all 10 instances. Every guess matches every query answer it received.
- **The deductions held up.** `b2r1g1` spends its leftover queries checking cells the board had deduced from the gate rules. It checked them on every instance and no answer contradicted a deduction. That supports the shared rule enumeration behind all six harnesses.
- **The difference is how many queries each needs.** Most new instances should come down to whether the board is fully pinned before the 18-query budget runs out. Here `b2r0g1` used fewer queries than the incumbent on 6 of 10 instances:
  - instance 206: 10 queries vs 16
  - instance 238: 11 vs 15
  - instance 205: 11 vs 14
  - instance 266, the hardest: 16 vs 17
  
  It used 1–2 more on the four easier instances (60, 171, 173, 230), where the extra queries don't matter.

**Interface check.** Every candidate sends only the model's own queries to the environment.
- `b2r0g1` adds an information-gain "worth" to each unknown cell and lists the top cells. The model still picks and writes each query.
- `b2r0g2` refuses to send a query unless the cell is on its own shortlist, which comes close to the harness choosing queries. I didn't pick it.

**Why not the others**
- `b2r0g0` and `b2r1g0` only fix fallback paths. On this batch they used the same number of queries as the incumbent or one fewer.
- `b2r1g1` always uses all 18 queries and makes more model calls, with no accuracy gain on this batch.

I didn't re-run any harness; this choice rests on the existing traces and the source files.