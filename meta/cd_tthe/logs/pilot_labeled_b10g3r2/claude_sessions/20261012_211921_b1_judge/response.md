Every harness got 100% correct, so I'm comparing the source code and the number of queries each one used. Next I'm checking the newest candidate's final-guess code.I chose **`cand_pilot_labeled_b10g3r2_b1r1g2`** and wrote that name to `choice_b1.txt`.

- **Outcome doesn't separate them:** all 7 harnesses got every final table and every circuit right on this batch.
- **No candidate breaks the interface:** each one sends only queries the model wrote. In `b1r0g2` and `b1r1g2`, if the harness rejects every query the model proposes in a turn, it sends the best one of those the model itself proposed. The harness never makes up a query, so I didn't reject them.
- **Why `b1r1g2`:**
  - **Fewest queries:** it tied for the lowest average, 9.9 per episode against the incumbent's 10.5.
  - **It refuses weak queries:** if the model picks a query that tells much less than the best one available, it's sent back to choose again. In the incumbent's traces, the model kept making exactly that kind of second query.
  - **It reads a common formatting slip:** it accepts the query tag when the model drops the opening `<`, which the model often did.
  - **Its final guess is always a possible combination:** the three tables must come from one way of sharing out the gates across the circuits. The other harnesses fit each circuit separately, which can produce a set of tables the gate counts don't allow. That only matters when the queries run out before the circuits are pinned down, and that never happened on this batch. On new instances with more NOT gates, where that happens more, it protects accuracy.
- **Nothing is tuned to this batch:** the changes apply to any instance.

The candidates show `tokens_out/ep=0` even though their code calls the model. That looks like a logging gap rather than a difference between harnesses, so I didn't count it for or against any of them.