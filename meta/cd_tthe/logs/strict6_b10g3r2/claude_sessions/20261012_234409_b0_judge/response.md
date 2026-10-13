Reading the harness sources is done. Now I'm checking the traces: for each final guess, does it agree with the answers and the gate rules, and is it the only possibility left?I chose **`cand_strict6_b10g3r2_b0r1g1`** and wrote it to `choice_b0.txt`.

**How I judged:** I can't see the truth tables, so I used a stand-in measure. For each trace, I listed every circuit triple the inventory and gate rules allow. I kept only the ones that fit all the query answers. Then I checked whether that left exactly one solution and whether the final guess was that solution.

| harness | answers leave one solution | guess fits rules + answers | guess agrees with its own answers |
|---|---|---|---|
| bare | 5/10 | 0/10 | 0/10 |
| r0g1 | 5/10 | 3/10 | 7/10 |
| r1g0 | 3/10 | 0/10 | 10/10 |
| r0g2 | 10/10 | 10/10 | 10/10 |
| **r1g1** | **10/10** | **10/10** | **10/10** |

- **bare:** none of its guesses matched its own query answers, and 3 guesses couldn't be parsed at all.
- **r0g1 and r1g0:** their final step is better, but their questions often don't narrow things down to one solution. r1g0 never produced a guess that fits the rules.
- **r0g2 vs r1g1:** both pinned down every instance and gave the matching guess, so on this evidence they tie.

**Why r1g1 over r0g2:** r0g2 is effectively choosing the queries itself. Each turn it shows a ranked list of the "most informative" queries. The model sent the top one in 113 of 126 query turns, and the harness rejected 5 queries and asked again. That is the query-choosing the interface forbids.

r1g1 keeps bare's conversation. After each answer it adds a board showing which outputs are already observed, which the rules already force, and which are still unknown. It doesn't rank, suggest or reject queries, so the frozen model still picks every one. It stops asking once nothing is unknown, then has the model copy the board one circuit at a time, sending back any table that contradicts the board or the rules. Everything it uses is general: the inventory, the rules and the answers, with nothing tied to this batch.

One caveat: r1g1 builds the full list of possible triples up front. With much larger gate inventories that list could get big, but on this batch each run took at most 0.12 s.