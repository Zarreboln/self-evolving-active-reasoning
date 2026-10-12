The simulation finished, and b2r0g2 comes out ahead on NOT-heavy inventories. I'm writing the choice now.I picked **`cand_pilot_strict_b10g3r2_b2r0g2`** and wrote it to `choice_b2.txt`.

- **Interface check:** none of the five harnesses chooses queries itself; in each one the model writes every query from a menu, so none was rejected. b2r0g2 shortens the menu to the 2–3 strongest options, but the model still makes the pick.
- **Traces don't separate them:** on all 10 instances, every harness ended with all three circuits pinned to one candidate. Every final guess matched every answer received, and all five gave the same 24 bits on every instance. No harness repeated a query.
- **What decided it:** whether a harness pins all three circuits within the 18-query budget on harder, NOT-heavy inventories. I ran an offline simulation with a stand-in model that picks either the first menu item or a random one. It used 40 random circuits for each of 7 inventories (0–4 NOT gates). Counting runs that needed more than 18 queries:

| harness | runs over 18 (first pick / random pick) |
|---|---|
| b0r0g2 (incumbent) | 12 / 28 |
| b2r0g0 | 12 / 7 |
| b2r0g1 | 5 / 16 |
| b2r1g1 | 5 / 16 |
| **b2r0g2** | **1 / 4** |

  b2r0g2 is better because it ranks queries over all three circuits together, using how the gates are split between them, and its short menu keeps the model off weak picks. Live, it pinned all three circuits after about 12 queries on average (most 16). The incumbent averaged 12.9 and needed up to 17, close to the 18-query budget.
- **Its other change is harmless:** after pinning, b2r0g2 spends the leftover queries checking its deduced tables. None of those checks contradicted a deduction in this batch, so it never changed an answer.

One caution: `branch_lineage.jsonl` contains an `exact` score field, which is a label the strict mode is meant to withhold. I saw it while reading the lineage and didn't use it in the decision.