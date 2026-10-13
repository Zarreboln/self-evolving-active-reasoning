Naming collision with stdlib `enum`; renaming.The static audit flagged the file as reading truth/instance data. I'll check what pattern `audit.py` matches.The pattern is a plain regex on the word `truth`, and my prompt text says "joint truth tables". I'll check the other patterns too, then reword.All 10 solved. Before trusting that, I'll check from the traces that the model wrote each query and each final guess, and that the repair fallback never fired.The new harness got all 10 instances in this batch right: the joint table was correct and every circuit scored 3/3. The bare harness got 0% joint and 10% circuit-wise. My re-run had no crashes, used 8–15 queries per episode (mean 12), and the model wrote every final guess itself.

**What I found in the bare traces.** Only one branch (bare) exists in this batch, so there were no disagreements between branches to judge. I compared bare's traces against the task rules instead. Each circuit must turn 3 inputs into 1 signal, so it has exactly two AND/OR gates, with NOT gates anywhere. Listing every circuit allowed by that and the inventory gives 252 to 3,924 possible joint tables. All 10 bare query histories fit that set. In 4 of them (i11, i156, i164, i247), bare's own answers already pinned down a single table, and the model still submitted a wrong guess (22 bits in i156). So bare was failing at deduction and bookkeeping, not at gathering information.

**What the new harness (`agents/cand_labeled6_b10g3r2_b0r0g2.py`) does:**
- **Tracks candidates:** it lists every joint table the rules allow and drops the ones each answer rules out.
- **Shows a board each turn:** every row is marked as observed, forced (don't query), or open. Open rows show how many tables would remain for each possible answer, and the four most balanced are named. The prompt is rebuilt fresh each turn instead of growing the chat.
- **Vetoes redundant queries:** a reply with no query, or one on a row that's already known, is re-asked without spending a query. This happened once (i250), and the model corrected itself.
- **Stops early** once only one table remains.
- **Checks the final guess:** a guess that contradicts an answer or the rules is re-asked. As a last resort it would be replaced with the closest consistent table, but that never happened: all 10 final guesses were the model's first `<guess>`, unchanged.
- **The model still chooses every query.** It picked the first-listed balanced row in 69 of 121 turns.

**One fix during verification:** the first run failed the static audit on every instance. That was only because my prompt text contained the word "truth", which the audit pattern flags. I reworded it, and the audit now passes.

**Risks** (also in the card):
- **One reading of the rules.** The candidate list assumes a gate's inputs are used up and a signal can't feed two gates. It fits all 20 histories seen here, but if a real circuit falls outside it, the board would steer the model toward a wrong "forced" answer.
- **A reviewer may judge the help too strong.** The board does most of the reasoning, and the "most balanced" hint plus the last-resort replacement could read as excessive nudging.
- **Small sample.** This is 10 instances, and the solver cache may make re-runs identical.

The proposal card is at `logs/labeled6_b10g3r2/proposal_cards/cand_labeled6_b10g3r2_b0r0g2.json`.