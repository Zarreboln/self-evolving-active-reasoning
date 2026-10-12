I'll start by reading the base interface, the parent harness, and the traces.Status: I've copied the parent to the b1r0g2 file and am now editing it. The audit forbids the word "truth" and file access in the harness source, so I'm avoiding both.Next, the harness side: a verification menu, phase-aware state text, and the new loop.The audit passes, and the offline cases behave: when the deduction is right it is unchanged, and an off-rules circuit was caught and recovered. Now the real run on the batch.After the checks, every circuit in all 10 episodes is pinned by its own observations, even with the inventory coupling dropped. Let me look at one check-phase prompt and the re-ask rate.Verified. Writing the proposal card.The child harness is written and verified on the batch: 10/10 ran with no crashes, each produced a final guess, and each used all 18 queries. Its final answers are identical to the parent's on every instance. On this batch it adds confirmation of the parent's answers, not corrections.

**What the audit found.** The b1 batch has only one active branch, so there were no cross-branch disagreements to settle. Instead I checked the parent's "SOLVED" claims with a separately written enumerator. On all 10 instances, each final circuit is the only table that fits both the rules and the answers. The weak point was how those circuits got pinned: many relied on the rule that the gates are split across A, B and C, not on their own answers. Examples:
- i228 B: 2 inputs observed directly, 16 tables fit those answers.
- i81 C: 2 inputs observed, 16 tables fit.
- i209 B and i253 B: 3 inputs observed, 9 tables fit.

At the same time, about 7 of the 18 queries went unused per episode.

**What the child changes** (`agents/cand_pilot_strict_b10g3r2_b1r0g2.py`):
- **Check queries:** once nothing is left to tell apart, the model picks checks from a menu. The first checks target outputs that were inferred only from the gate split; after that, any output not yet observed. The model still writes every query, and off-list replies are re-asked.
- **Fallback if a check contradicts a deduction:** first drop the gate-split rule for that circuit, then allow any non-constant table, then go back to normal queries. The parent jumped straight to allowing any non-constant table.
- **Unchanged:** the parent's candidate enumeration, query menu, re-asking, and building the final answer from the model's picks. The queries before the first check were identical to the parent's on all 10 instances.

**Evidence from the run:**
- The checks added about 74 new answers across the 10 episodes, and none contradicted a deduced table.
- After the checks, every circuit is pinned by its own answers alone, without the gate-split rule. Each circuit now has 5–7 of its 8 outputs observed directly.
- The model handled the check menu well: 4 re-asks in 73 calls.

**Limits:**
- The contradiction-and-recover path has only been exercised in an offline test with a fake environment, where it recovered a deliberately off-rules circuit. No real instance triggered it.
- Model calls per episode rose from about 8–17 to 19–27, and every episode now spends the full budget.
- The `traces/*.json` files include a `gold` field. I saw it for i130 while listing the file's keys, then used only the query histories and system prompts. Strict-mode runs probably shouldn't write that field into traces the proposer can read.

The proposal card is at `logs/pilot_strict_b10g3r2/proposal_cards/cand_pilot_strict_b10g3r2_b1r0g2.json`.