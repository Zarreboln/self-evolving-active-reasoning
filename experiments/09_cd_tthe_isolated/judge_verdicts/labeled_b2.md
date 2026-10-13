I kept the incumbent, `cand_labeled6_b10g3r2_b1r0g2`, and wrote that name to `choice_b2.txt`.

I compared each candidate's source file line by line against the incumbent's. I did not open the per-episode trace files or re-run any harness. The run summary was enough.

- **No candidate scored better.** All six got every joint truth table right on all 10 instances.
- **Four candidates are the incumbent's code.** `b2r0g0`, `b2r0g2`, `b2r1g0` and `b2r1g2` differ from it only in the docstring, or not at all. There's nothing to prefer them for.
- **`b2r1g1` is the only real change, and it's worse.** It replaces the deduction board with a short list of the best next queries, already written out, and tells the model to copy one exactly. That gave the same accuracy but used more queries per episode (11.8 against 11.1). It also takes away most of the model's own reasoning, which moves it closer to the harness choosing the query itself. The incumbent names the best rows but leaves the model to work from the board and pick one, which is a safer fit with the interface rule.