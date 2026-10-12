No crashes, but only 3 of 10 episodes produced a parseable guess. I'll check the final replies and how the queries were spread.The query phase now runs with almost no re-asks: 21 calls = 18 queries + 3 final attempts. But the final step fails in 9 of 10 episodes. Checking the final replies and query spread:The new harness runs on all 10 instances with no crashes, and every episode now produces a final guess (bare missed one). It fixes how the model records what it saw, but not how it spreads its queries, so circuit C is still mostly guessed. I can't tell how many tables are actually right, since I have no outcome data.

**What I tested:** whether the model fails because it loses track of its own answers in the long chat (36+ messages by the end), rather than because of how it spends queries. The sibling branch blocks queries to force a spread and overwrites the final bits with the observed answers; this one does neither.

**The change** (`agents/cand_pilot_strict_b10g3r2_b0r0g1.py`): there's no running conversation. Every model call is the benchmark's system prompt plus one fresh message containing:
- a line saying the game is already under way,
- every answer received so far,
- each circuit's table so far, with `?` for rows not yet asked,
- the list of rows still unknown.

The model still writes every query and every bit. For the final answer it writes one 8-bit line per circuit, and the harness joins its three lines into the 24-bit guess.

**How it got there (3 runs of the check script):**
- **Run 1 was broken.** The system prompt ends with "Please start with your first query", so with no chat history the model treated every call as turn 1. It asked A(0,0,0) up to 18 times. The solver also returns the same reply to the same prompt, so plain re-asks didn't help.
- **Run 2** said "you are in the middle of the game" and listed the unknown rows. All queries became unique, but 9 of 10 final answers didn't parse, because my prompt led the model to put its `A:` / `B:` / `C:` lines inside `<guess>`.
- **Run 3** joins those three lines into the guess.

**What I found in the run 3 traces:**

| | bare | this harness |
|---|---|---|
| Guess contradicts rows the model itself observed | 8 of 9 guesses (2–11 rows each) | 0 of 10 |
| Repeated queries | 0 | 0 (18 unique per episode) |
| Queries per circuit (A/B/C) | mostly 8/8/2 | 8 / 7–8 / 2–3 in all 10 |

- **The hypothesis held for recording answers.** Once the model sees a clean list of what it observed, its final tables match it.
- **It did not hold for query spread.** Even with "C (8 unknown)" on screen, the model works through A and then B. Its inferred rows can also break the task rules: instance 128 guesses C = 00000000, a constant output the rules forbid, and in a circuit built only from AND/OR it sets B(1,1,1)=0. Spreading queries across circuits is the next lever; I left it out to test one change at a time.

The proposal card is at `logs/pilot_strict_b10g3r2/proposal_cards/cand_pilot_strict_b10g3r2_b0r0g1.json`. The trace `.json` files include a `gold` field with the correct answers; I never read it, and every check above uses only the queries, the answers and the guesses.