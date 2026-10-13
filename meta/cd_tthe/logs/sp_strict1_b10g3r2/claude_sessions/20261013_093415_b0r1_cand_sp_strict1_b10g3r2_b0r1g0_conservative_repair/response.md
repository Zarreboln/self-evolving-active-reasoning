Reading the parent's traces for all 10 puzzles to find concrete failures.Detector at ≥60% overlap and ≥3 shared words catches i20, i52, i94 copies with few false positives. Writing the candidate now.Candidate written (parent + mental-state re-ask + NO-claim revision). Running the check on the 10 puzzles.All 10 ran with no crashes. i90 shows 8% not-a-question after re-asks, so I'm checking what the re-asks produced.Patch is in. Re-running the full check.No crashes; not-a-question is 0% everywhere now. i52 dropped to 0 Yes, so I'm checking whether the re-ask caused it.Audit passes. Last step before the card: counting mental-state questions actually sent, compared with the parent.I built `agents_sp/cand_sp_strict1_b10g3r2_b0r1g0.py` from the parent with two small changes. On the final run all 10 puzzles ran with no crashes, each produced a final answer, and `audit.py --task sp` passed. There's no F1 score, so I can't say whether answers actually got closer to the hidden stories. What I can show is fewer wasted questions and fewer answers that contradict the referee.

**The two failures I targeted, both visible in the parent's traces:**
1. **Questions about feelings.** On the "who sends these letters" puzzles the model kept asking what the character feels, believes or suspects (8 such questions in i82, 9 in i93). The story doesn't record feelings, so the referee mostly answered Unknown (14 Unknowns in i93) or an empty Yes.
2. **Final answers claiming things the referee had said No to.** The parent's answers to i20, i52 and i94 restated No-answered questions as if they were true.

**Changes:**
- **Re-ask feelings questions.** If a question matches words like feel, believe, suspect, think or aware, the model is asked once to replace it with a yes/no question about what actually happened. If the replacement starts with "What/How…", the original question is sent instead. The first version sent two "What…?" questions on i90, and the referee rejected both as not a question.
- **Check the final answer against No replies.** If the answer reuses most of the distinctive words of a No-answered question, the model is shown those questions and asked to delete only that claim and add nothing. My first wording ("rewrite without the claim") made i94 swap one No-answered claim for another and made i82 and i93 tack on "they do not…" sentences, so I narrowed it.
- **Everything else is the parent's.** The stall nudge, the recap, the official prompts, sending questions unedited and using all 25 turns are unchanged. Where neither check fires (i16, i36, i5, i72), the episode is identical to the parent's.

**Results from the final run of the check script (I ran it three times while fixing the issues above):**

| | Parent | Candidate |
|---|---|---|
| Feelings questions sent, i82 / i93 / i90 | 8 / 9 / 2 | 0 / 0 / 0 |
| Unknown replies, i93 | 14 | 6 |
| Not-a-question replies | 0% | 0% |
| Final-answer length, i82 / i93 / i90 (words) | 143 / 131 / 160 | 94 / 82 / 116 |

- **Yes rate fell slightly overall (32.0% → 29.2%).** Almost all of that is i52, which got no Yes at all after its questions went a different way from Q8 on. That's one sampled run, so it could be chance as much as the change.
- **The final-answer check fired 4 times.** In i93 and i94 it removed one claim each, and in i82 it correctly left the answer unchanged. In i52 the model only reworded the contradicted claim and kept it.
- **It misses reworded contradictions** like i20's "had access to her identity" and i36's "surprise celebration".

The proposal card is at `logs/sp_strict1_b10g3r2/proposal_cards/cand_sp_strict1_b10g3r2_b0r1g0.json`.