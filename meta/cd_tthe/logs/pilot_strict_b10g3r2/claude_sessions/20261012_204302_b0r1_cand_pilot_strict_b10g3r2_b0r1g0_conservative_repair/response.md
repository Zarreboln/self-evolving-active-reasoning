All 10 instances ran without crashing. Next I'll read the traces it wrote.The model now states a value for nearly every unqueried row. Next I'll look at instance 164 (3 final calls) and one ordinary final reply.Both fixes are working: every instance split its queries 6/6/6 with no repeats, and every ? row was filled from the model's own statements. Now writing the proposal card.I wrote the repaired harness in `agents/cand_pilot_strict_b10g3r2_b0r1g0.py`. The check script ran all 10 instances with no crashes: each produced a final guess and used all 18 queries. I can't tell whether the tables are correct, because no outcome signal is available.

**What the parent's traces showed**
1. **Final guess unusable (10/10 episodes):** even after 3 conflict retries, the model's `<guess>` was the wrong length (0, 20, 32 or 33 bits) or contradicted answers it had received. In i105 the returned text was just the first 24 bits of a 32-bit string.
2. **Uneven queries (4/10 split 8/8/2):** the model swept A and B row by row and left C with 2 queries; three other runs ended 4/6/8, 5/6/7 and 6/6/6. i265 also asked the same query twice.

**The two changes**
- **Spreading the queries:** the harness rejects a query (without spending budget) if the row was already asked, or if that circuit already has 6 queries while another has fewer. The model gets a message listing the unasked rows and picks its own replacement. This is capped at 12 rejections per episode. The per-turn message also says "at most 6 per circuit."
- **Building the final guess:** the model now writes `X(a,b,c) = v` for each row it didn't query. The harness builds the 24 bits from the answers actually received plus those lines, falling back to the model's own `<guess>` if needed. It asks again, up to 3 times, if a row is missing or a circuit comes out constant (the rules forbid constant circuits). Everything else from the parent is unchanged.

**What the re-run showed**
- All 10 episodes split queries 6/6/6 with 18 distinct queries, and the model followed every rejection on its next reply.
- In 9/10 episodes the model stated all 6 unqueried rows itself; in i247 two came from its `<guess>`. Its raw `<guess>` was still malformed in 5/10 (lengths like 72, 22 and 16), so assembling it in the harness is what makes these episodes usable.
- In i164 the constant-circuit check fired: the model's first answer made B all zeros, and the third reply fixed it to a reasonable "a AND b".

**Main remaining risk:** because the model queries rows in order and stops at 6, A and B usually leave the same two rows unqueried, (1,1,0) and (1,1,1). Those rows tell many circuits apart, so the model's guesses there are the likely errors. Spreading which rows get asked would need more than a conservative repair. The other risks are in the card.

One disclosure: while checking the trace JSON format early on, I printed a parent result file that includes a `gold` field, so I saw that the parent's i105 guess was wrong. I didn't look at gold again, and the design is based only on the failures described above, which are visible in the `.md` traces.

The proposal card is at `logs/pilot_strict_b10g3r2/proposal_cards/cand_pilot_strict_b10g3r2_b0r1g0.json`.