I picked **`cand_plumb_b0r0g0`** and wrote that name to `logs/plumb/choice_b0.txt`. I didn't re-run either harness; the decision rests on the numbers you gave and on comparing the two source files.

**It stays within the interface.** The model still writes every query and the final guess. What the candidate adds is bookkeeping that never looks at the truth tables:
- It lists every set of circuits that fits the gate inventory, then drops any set that contradicts an answer received so far.
- If the model asks for an output that earlier answers already fix, the harness tells it the value without using a query and asks again. If the model asks for the same output twice in a row, the query is sent, just as in `bare`.
- It stops querying once only one set of circuits is left.
- Before the final-guess prompt, it lists every output that is already determined.

**It does better than `bare` on every signal:**

| Signal | `bare` | candidate |
|---|---|---|
| uniq_end | 40% | 100% |
| alive_end | 4.2 | 1.0 |
| zero_info | 34.1% | 0% |
| eff | 0.580 | 0.904 |
| health | 100% | 100% |
| crashes | 0 | 0 |

**It isn't tuned to this batch.** The circuit list is computed from each instance's own gate inventory, so the same logic applies to new instances. There's also a cap of 40 model calls, so the re-ask step can't loop forever.