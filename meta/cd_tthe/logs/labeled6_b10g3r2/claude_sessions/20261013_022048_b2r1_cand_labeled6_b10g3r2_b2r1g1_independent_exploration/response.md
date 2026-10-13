Parent harness already scores 100% here; I'm scanning the traces for vetoes/retries to find a residual weak spot worth testing.Now I'll replace the two prompt builders with compact versions and update the docstring.All 10 correct, and calls = queries + 1 on every instance, so there were no vetoes. But average queries rose from 11.1 to 11.8, and i173 used only 5. I'll check both before trusting the result.The model actually copied the first menu line on 105 of 118 turns, not all of them. Correcting the card.The new harness got all 10 instances right on this batch, with no crashes, and it never had to re-ask the model. The catch is that it used more queries than the parent (11.8 per episode vs 11.1).

**What I tested.** The parent and both siblings already score 100% here, so the outcome can't tell harnesses apart. The only problem left in the traces was one re-ask: on instance 266 the model asked about row A(1,1,1) when the suggested row was A(1,1,0). My hypothesis is that this is a reading problem, not a strategy problem. The prompt lists all 24 rows on near-identical lines, and the 7B copies a neighbouring one by mistake.

**The change** (`agents/cand_labeled6_b10g3r2_b2r1g1.py`): one mechanism, a shorter prompt.
- **Each turn:** the 24-line row list is replaced by three strings like `A = 0??1????` (`?` = not yet known), the number of tables still possible, and a list of the best queries written out ready to copy as `<query>…</query>` lines.
- **Final turn:** the row list is dropped there too.
- **Kept from the parent:** the list of possible tables, the rejection and re-ask of bad queries, stopping early, the labelled A/B/C final answer, and the fallback that corrects a final answer to the nearest possible table.

**Results from re-running the check script:**
- **Outcome:** 10/10 fully correct, every final answer read successfully, no crashes.
- **Re-asks:** zero on every instance; the parent needed one, on instance 266.
- **Queries used:** 13, 13, 15, 13, 10, 13, 14, 12, 10, 5. Every episode still narrowed down to a single table well within the 18 allowed.

**Risks:**
- The model copied the first item of the query list on 105 of 118 turns. So in practice the list's order mostly decides which of several equally good queries gets asked. The model still writes every query, but that order cost queries here: instance 266 took 15 against the parent's 12, so the margin before 18 is smaller.
- The model no longer sees how much each row would narrow things down, so it has less to reason from if the list were ever unhelpful.

The proposal card is at `logs/labeled6_b10g3r2/proposal_cards/cand_labeled6_b10g3r2_b2r1g1.json` and is valid JSON.