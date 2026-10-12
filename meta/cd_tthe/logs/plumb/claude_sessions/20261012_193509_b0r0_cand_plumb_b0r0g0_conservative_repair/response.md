The provided check script crashes on its own: it writes JSON `null` into Python source. I'll run it unmodified with `null = None` defined first, and report the bug.Every final guess matches the uniquely determined 24-bit table. Last, I'll run the invariant audit if it can be run on one file, then write the proposal card.The new harness runs cleanly on all 5 instances, and in every one the answers now pin the three circuits down. Caveat: the solver here is the scripted "sequential" mock, not the real Qwen-7B, so this shows the mechanics work, not how the real model will behave.

**Failures I found in the `bare` traces:**
- The model sweeps entries in a fixed order and asks for outputs its earlier answers already determine (34% of queries carried no information).
- It runs out of queries partway through circuit C.
- It fills every entry it never asked with 0, so instance 11's guess was wrong even though its answers had pinned the circuits down.

**What I changed** in `agents/cand_plumb_b0r0g0.py`:
- **Tracking:** the harness works out every circuit configuration the gate inventory allows, then drops the ones each answer rules out. I wrote this myself in the harness; it doesn't import the oracle library.
- **Veto:** if the model asks for an output that every remaining configuration agrees on, no query is spent. The harness tells the model the value and asks again. If the model asks for the same entry again straight away, the query goes through as in `bare`, so it can't loop.
- **Early stop:** once only one configuration is left, the harness stops asking for queries.
- **Final turn:** all the determined outputs are listed just before the benchmark's final-guess prompt. The model still writes every query and the final guess.
- Prompts and wording are otherwise the same as `bare`.

**First attempt, dropped:** I first listed the determined outputs after every answer. That made things worse: the model's fixed-order scan started repeating entries (55–87% repeats) and only 1 of 5 instances was pinned down. So the harness now only states a value when the model asks for it.

**Results:**

| instance | queries | uniq_end | zero_info | repeat | eff |
|---|---|---|---|---|---|
| 156 | 9 | ✓ | 0 | 0 | 0.95 |
| 11 | 12 | ✓ | 0 | 0 | 0.85 |
| 250 | 17 | ✓ | 0 | 0 | 0.83 |
| 128 | 12 | ✓ | 0 | 0 | 0.98 |
| 98 | 10 | ✓ | 0 | 0 | 0.91 |

The parent scored uniq_end 40%, zero_info 34% and eff 0.58. In all 5 episodes the model's final 24-bit guess matches the single table the answers allow (checked without looking at the labels), and `audit.py --arm scratch` passes.

**Bug in the check script:** `logs/plumb/check_b0r0_cand_plumb_b0r0g0.py` crashes before running anything, because line 4 contains JSON `null` instead of Python `None`. I didn't edit it; I ran it unchanged with `null = None` defined first. Whatever generates these scripts probably has the same bug for other candidates.

**Risks:** with the real model, it might ignore the veto message, or copy the listed outputs into the 24-bit string wrongly. If my enumeration of possible circuits were wrong and no configuration stayed consistent, the harness falls back to `bare`'s behaviour.

The proposal card is written to `logs/plumb/proposal_cards/cand_plumb_b0r0g0.json`.