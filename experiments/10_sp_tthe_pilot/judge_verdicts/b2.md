I'm comparing the candidate sources against the incumbent; next I'll read the traces.Reading instances 71–98, then I'll check the leading candidates' sources for interface violations.I chose **`cand_sp_strict1_b10g3r2_b2r1g0`** and wrote it to `choice_b2.txt`. With no hidden stories available, this is a judgment from the traces and source code, not a measured win.

**Why b2r1g0:**
- **It builds on the incumbent with general fixes.** It adds three checks:
  - Stories and final answers can't keep a person or thing the referee turned down in two or more questions.
  - A question can't say "the X" when X exists only in the model's own unconfirmed guess. The model must first ask whether any X was involved.
  - Invented names are caught before the other checks, and a name only counts as known once the puzzle or a Yes answer uses it.
- **It fixes a problem the incumbent actually showed.** On instance 4, the incumbent spent several questions on "the digital device" and "the secure storage location", which the referee never confirmed. That made-up setup carried into its final answer.
- **It had the best public signals.** Its Yes rate was the highest (37.2%) and its No rate the lowest. It also confirmed more facts on instances 19, 71, 85 and 98. These are signs of better questions, not proof of correct answers.
- **It doesn't break the interface.** The harness only checks the model's questions and asks again; the model writes every question and the final answer.

**Why not the others:**
- **b2r0g0** is a subset of b2r1g0 (it only has the turned-down-entity filter) and gave nearly the same results as the incumbent.
- **b2r0g1, b2r1g1 and b2r1g2** switch to a different search strategy that splits each open point into a few possibilities. Their results were uneven: b2r1g1 confirmed only 2 facts on instance 23, and b2r1g2 only 1 on instance 85.
- **b2r1g2's word cap rests on a wrong premise.** It caps answers at about 90 words because "stored answers lose text beyond 600 characters". In `solve_worker.py:107` that cut applies only to the logged copy, while scoring uses the full answer.

No candidate gave strong evidence of being better. I picked b2r1g0 over keeping the incumbent because its changes are low-risk, fix a failure the incumbent actually showed, and its signals don't point the wrong way. I didn't re-run any harness.