I've read all 7 sources, and none of them breaks the interface. Next I'm comparing the traces' final answers and the referee's replies.I picked **`cand_sp_strict1_b10g3r2_b0r1g2`** and wrote that name to `choice_b0.txt`. I judged only from the traces and the referee's replies; I didn't re-run any harness.

**Interface check:** None of the seven break the rules. The model writes every question and every final answer. The candidates only build the prompts, filter questions and choose among drafts the model wrote. One edge case: r0g2 and r1g2 can send a placeholder "(no question)" if every retry fails. That never happened in these traces.

**Why r1g2:**
- **It finds the twist more often.** It checks the puzzle's assumptions first, then keeps one working story and tests the parts not yet confirmed. Its questions surfaced key Yes answers more often than the others:
  - i16: "John faked his death?" (Yes)
  - i82: "Was Sarah writing the letters herself?" (Yes)
  - i5: Emily was still alive after her supposed death and wrote the letter herself
  - i94: "Did Emily write the letter?" (Yes)
  - i20, i36: the other strong candidate (r0g2) got close, but r1g2's answers stay closer to the Yes facts.
- **Its final answers mostly keep what the referee confirmed.** Before answering, it checks for confirmed facts the draft left out and has the model rewrite once. Others often contradicted their own Yes answers:
  - r0g2 on i5 says Emily was not alive, although the referee said Yes to "Was Emily still alive two days after the funeral?"
  - r0g2 on i94 ends with "a postal worker" story that ignores its own Yes facts.
  - r0g0 and r1g0 on i20 say an impostor sent the postcard, although the referee said Yes to "Did Emily send the postcard herself?"
- **The others are weaker.**
  - **bare, r0g0, r1g0:** they spend many turns on feelings and minor details, and their answers are hedged or just restate their questions. r1g0 adds little over r0g0 (on several puzzles they asked identical questions).
  - **r0g1, r1g1:** they often invent names (Dr. Emily Chen, Mark, Alex) and gadgets that nothing the referee said supports.

**Caveats:**
- r1g2 uses far more tokens (about 5,000 output tokens per puzzle versus 673 for bare).
- Its answers can still add made-up details; on i52 its answer says the incident "was not staged by someone else" after a Yes to that question.
- With ten puzzles and no truth labels, this is a judgment call, not a measured win.
- The `guess_text` field in the trace JSON is often cut off at the start. The model's full reply in the trace is complete, so I judged from that. It affects every harness the same way.