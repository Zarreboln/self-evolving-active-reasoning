I wrote the new harness in `agents/cand_pilot_strict_b10g3r2_b0r0g2.py`. On this batch every episode now ends with each circuit narrowed to a single table, within the 18 queries, and the returned 24 bits match those tables. That's checked against the task rules and the answers only; I never saw the hidden tables, so it isn't a confirmed score.

**What was going wrong in `bare`.** I checked its final guesses against its own answers:
- The model reads each circuit row by row, 8 queries per circuit, so it reaches C with only 1–2 queries left.
- Only 4 of 10 final guesses were 24 bits and agreed with the answers it had received. 2 had no guess, 3 were the wrong length (20, 26 and 32 bits), and 4 contradicted 2–11 of its own answers.

Only `bare` ran on this batch, so there were no other branches to compare. The one real disagreement was on i98. Bare's guess matched its 18 answers, but it guessed A(1,1,1)=1 without ever asking. My run asked, and the environment answered 0.

**What the new harness does:**
- **Lists what's possible.** Each circuit starts with 3 inputs and must end with 1 output, so it has exactly two AND/OR gates plus any NOTs, and the inventory is split across A, B and C. The harness lists every circuit that fits and drops those that contradict an answer. All 10 of bare's query histories fit this reading of the rules.
- **Shows a short status each turn.** The model gets the answers so far, which circuits are solved, the remaining candidate tables, and up to 8 queries that would still tell candidates apart, with how each would split them.
- **Leaves the queries to the model.** It writes every query. A query that's malformed, repeated, on a solved circuit, or useless is sent back with the reason (up to 4 tries), and these retries don't use up queries. Querying stops once all three circuits are solved.
- **Builds the final answer from the model's picks.** The model picks a numbered candidate for each circuit, and the harness assembles the 24 bits, accepting only tables that fit the answers.

**Results.** I ran the check script twice, once before and once after a wording fix: 10/10 ran without crashing and produced a guess. The final run used 6–18 queries (mean 10.7). Replaying each history left exactly one candidate per circuit, and the guess equalled them on all 10. The model's final reply was always "A: 1, B: 1, C: 1" with a matching guess, so the backup path was never used. For the three instances I compared across the two runs, the tables were identical even though the queries differed.

**Risks:**
- **Thin margin when NOT gates are in play.** i250 needed all 18 queries and i247 needed 17, because the model doesn't always pick the query that splits the candidates most evenly.
- **Depends on my reading of the rules.** If the real circuit generator allows more than I enumerate, the harness could mark a circuit solved with the wrong table. If answers ever contradict the list outright, it falls back to "any table consistent with the answers."
- **The harness now does much of the reasoning.** The model still chooses every query and every final pick, but mostly from the list the harness gives it.

The proposal card is at `logs/pilot_strict_b10g3r2/proposal_cards/cand_pilot_strict_b10g3r2_b0r0g2.json`.