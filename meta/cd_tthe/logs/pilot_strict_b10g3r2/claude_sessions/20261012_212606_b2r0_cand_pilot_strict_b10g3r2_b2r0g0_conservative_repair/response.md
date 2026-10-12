I'll stick to the `.md` traces (the `.json` files also carry outcome fields I'm not supposed to use).The model often accepts weak splits (e.g., 1-vs-3) while even splits are on offer. Let me quantify this across all 10 traces by replaying the logged queries through the parent's own candidate space.Writing the edits to the new candidate now: docstring, menu, and query-acceptance logic.Now the query-acceptance loop:`solve()` already handles `None` by re-asking without spending a query, so it stays unchanged. Running the batch check now.The check is running in the background; I'll read the traces once it finishes.The check is still running (first instance in progress). I'll pick up from here when it finishes.Everything checks out. I'll write the proposal card.The new harness, `agents/cand_pilot_strict_b10g3r2_b2r0g0.py`, re-ran on all 10 instances without a crash. It used 11.5 queries per episode on average, down from 12.9. Every episode ended with each circuit narrowed to one possible table, and the final answer matched those tables.

**How I judged "going well":** the system prompt's rules fix exactly which circuits are possible. I replayed each episode's queries and answers through that space. A good episode ends with one possible table per circuit and spends no queries that can't tell candidates apart.

**What the parent's traces showed:** all 10 parent episodes did end fully pinned down. But two problems were eating into the 18-query budget:
1. **A useless query was actually sent (i171).** The model kept alternating between `A(1,1,1)` and `B(1,1,1)`, which couldn't distinguish any remaining candidates. Each retry message showed only the latest rejection. After 4 tries the harness sent the model's first rejected pick anyway, wasting a query.
2. **Weak picks were accepted.** The parent accepted any query that told at least two candidates apart. The model often chose lopsided ones, splitting 1 against 3, when even 50/50 splits were listed. i284 finished at 17 of 18 queries and i266 at 16.

**Changes (everything else is kept from the parent):**
- The menu now lists only the queries that split the remaining candidates most evenly, and those are the ones accepted.
- A lopsided but still useful pick is rejected and kept only as a last resort, after 4 retries.
- A query known to be useless is never sent; the turn is asked again without spending budget.
- Retry messages list every query already rejected that turn.

**Re-run results:**
- Every accepted query was a most-even split, with no useless queries sent.
- Only one retry was needed per episode at most (6 episodes needed one, all on the first turn), and the last-resort fallback never triggered.

**What's still open:** i0 needed 17 of 18 queries even though every query was a most-even split. Its gate mix is 1 AND, 5 OR, 2 NOT. I simulated this harness's query rule on random circuits allowed by the rules, with no access to the real answers. For that gate mix, roughly 2% would need more than 18 queries. Picking the most even split one query at a time isn't always the fastest route overall. I left that alone because this was meant to be a minimal repair.

The proposal card is at `logs/pilot_strict_b10g3r2/proposal_cards/cand_pilot_strict_b10g3r2_b2r0g0.json`.

One more thing: the trace and result `.json` files contain `gold` outcome fields, which this setup is supposed to keep hidden. I worked only from the `.md` traces and the logged queries and answers. I saw one `exact` value by accident early on and didn't use it. I saved a memory note so later sessions avoid those fields.