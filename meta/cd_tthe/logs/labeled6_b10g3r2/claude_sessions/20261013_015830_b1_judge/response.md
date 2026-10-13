Checking how the candidates handled instance i228, the one where the incumbent's final answer came out the wrong length.I chose **`cand_labeled6_b10g3r2_b1r0g2`** and wrote that name to `choice_b1.txt`.

- **Interface:** I read all six sources, and none of them breaks the rules. In each one, only the model writes the query that actually gets sent. If its pick is rejected four times, the harness falls back to the model's own last query (`accepted = last_open or last_parsed`), never one the harness picks itself.
- **Outcomes:** All six got 100% on both joint and circuit-wise accuracy, with the same number of queries (11.4 per episode). The scores can't separate them, so I judged from the traces.
- **The incumbent's weak spot:** On instance i228 the incumbent wrote a 32-bit guess instead of 24 bits, and did it on all four tries. It only scored as correct because a single table was left, so the harness's last-resort fix ("nearest consistent table") had only one possible answer. With several tables left, a misaligned string like that could easily be fixed to the wrong table.
- **The candidates:** All five have the model write A, B and C as three separate 8-bit fields, which the harness joins. Every one got i228 right on the first try, and none of them needed a retry or veto anywhere in the batch. The incumbent needed three, all on i228.
- **Why b1r0g2:**
  - It changes only the final-answer step, so it adds little risk.
  - It still accepts a plain 24-bit guess.
  - If a field has the wrong length, the re-ask names that field.
  - If it does have to fall back to the nearest table, it pads or trims each field to 8 bits first. That keeps A, B and C in their own slots. b1r0g0 and b1r1g0 (which behaves the same as b1r0g0) don't do this.
  - b1r1g1 also rewrites the query loop to resend the model's earlier replies with each correction. None of that ran in this batch, so it's an untested extra risk.