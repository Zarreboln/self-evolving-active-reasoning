All five sources are mechanism-only (no hardcoded questions/answers). Now comparing traces per puzzle.I picked **`cand_sp_strict1_b10g3r2_b1r1g0`** and wrote that name to `choice_b1.txt`. It's a judgement call, not a clear win. Every harness has final answers that contradict the referee's replies somewhere. I didn't re-run any harness.

**Interface check:** all four candidates only add checks and re-prompts on top of the incumbent. The model still writes every question and the final story, and nothing is keyed to this batch's puzzles, so none had to be rejected.

**What's wrong with the incumbent in these traces:**
- **Repeated questions:** it asked "Did someone else sign the death certificate?" three times in i9. In i75 it asked "Did the tech genius install the mechanism without Sarah's knowledge?" three times and the tech-genius questions twice.
- **Chasing ideas the referee already denied:** in i75 it kept asking about the tech genius after the referee said No to one existing. In i37 it asked about an inventor and a mechanical arm through about 10 No answers.
- **Final answers that contradict the referee:** the i37 answer is built on that ruled-out inventor. The i27 answer denies the confirmed fact that Emily faked her death. The i75 answer says the forest is always moonlit, which was answered No.

**Why b1r1g1 (the second-closest) lost:**
- It also tackles chasing denied ideas, in a more targeted way.
- But it keeps the incumbent's fallback that can still send a repeated question.
- Its check on the final answer only covers things denied outright. In i13 its final answer still blames the business partner, after the referee said No to the partner being involved in faking John's death.

**Why b1r1g0:**
- It has the broadest set of general fixes:
  - A check that rejects a question or story built on someone or something the referee denied.
  - A repair step that sends any working story or final draft back when it states a ruled-out point as true. This fired often in the traces.
  - A fallback that asks a broad new question instead of sending a repeat.
- It had 0% repeats, the highest Yes rate (26.8%) and the lowest Unknown rate (7.2%) of the five.
- b1r0g0 is the same harness minus the denied-entity check, so b1r1g0 includes everything b1r0g0 does.
- It still has the problem in places: in i75 it kept asking about a levitation device, and that final answer still says the forest is always moonlit.