Reading the traces now. Next I'll measure, from the rules alone, how many candidate tables are still possible at the end of each episode.No crashes, and most episodes stopped early with the board fully resolved. Let me check with the rule analyzer whether each episode was pinned and whether the guess is consistent with that.Everything checks out. i250 resolved on its last query, so the margin there is thin. The model still sweeps rows in order and just skips deduced ones. Writing the proposal card.The new harness ran on all 10 instances with no crashes, and every episode produced a final guess. I can't see whether the guesses are actually correct. Judging only from the rules and the answers, all 10 episodes ended with exactly one possible set of three tables, and the guess matched it every time.

**Where the parent (g1) went wrong, from its traces:**
- **Querying:** about 6.4 of its 18 queries per episode asked rows whose answer was already forced by the rules plus earlier answers. The model sweeps one circuit row by row. A reply in the wrong format also cost a query (i98 used only 12).
- **Final answer:** its row parser didn't recognise LaTeX-formatted rows (`\text{C(0,0,0)} = 0`, `B(0,0,0) &= 0`). In i156 it used up every retry on that and kept a table the rules don't allow, even though the answers already pinned every row. In 4 episodes the answers fixed a single answer and it still guessed something the rules exclude.

**My hypothesis and mechanism:** the model fails because it can't see what the rules plus the answers already imply. It doesn't need someone choosing or blocking its queries. So the parent's continuous chat is kept, and every answer now comes with a 24-cell "knowledge board": each output is marked observed, deduced, or `?`. There are no query vetoes, candidate lists or suggested queries, unlike g2. Querying stops once no `?` is left. At the end, the model copies each circuit's board row as `<table>XXXXXXXX</table>`. A table that contradicts the board or breaks the rules is sent back with the reason. A reply without a valid query no longer uses up a query.

**Results on this batch** (the last two columns are my own check, from the rules and the answers only):

| | queries/episode | output tokens/episode | wasted queries/episode | one possible answer left | guess fits the rules |
|---|---|---|---|---|---|
| bare | 17.0 | ~3k | – | 5/10 | 0/10 |
| parent g1 | 16.8 | ~10k | 6.4 | 5/10 | 3/10 |
| **b0r1g1** | **12.9** | **~2k** | **1.3** | **10/10** | **10/10** |

Wasted queries fell from 6.4 to 1.3 per episode with no veto at all, which supports the hypothesis.

**Risks:**
- Everything above, including what the board marks as deduced, depends on my reading of the building rules. g1 and g2 use the same reading. If the real generator builds circuits differently, the board would show wrong values and the model would copy them.
- The margin is thin on some instances: i250 needed all 18 queries and i265 needed 17. The model still sweeps rows in order and only skips the deduced ones.
- The conversation gets longer (30–90k input tokens per episode).

The harness is at `agents/cand_strict6_b10g3r2_b0r1g1.py` and the proposal card at `logs/strict6_b10g3r2/proposal_cards/cand_strict6_b10g3r2_b0r1g1.json`.