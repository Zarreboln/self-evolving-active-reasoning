---
name: cd-tthe-cross-branch-answer-check
description: "cd_tthe label-free validation: check a candidate's final tables against every answer in OTHER branches' .md traces on the same instance"
metadata:
  type: project
---

In meta/cd_tthe strict runs (strict2_b10g3r2, 2026-10-12), a strong label-free check is to compare a candidate's returned 24-bit table against every `X(a, b, c) = bit` line in sibling/parent `.md` traces on the same instance. Those are real environment answers, mostly on rows the candidate never queried. For example, b0r0g2 vs bare had 174 answers and 0 disagreements. Also: the system prompt is not in the .md traces. To see it, put env.system_prompt into one llm user message in a probe harness and read `trace[0]` from the check's res json. strict2 check res files (reveal=none) have no `gold` key, but assert this before reading.

**Why:** "ran without error" proves nothing. The candidate-space replay alone can't detect a misread of the rules, and this check can.
**How to apply:** combine it with the replay in [[cd-tthe-strict-traces-label-leak]]. The rule space was already validated in [[cd-tthe-falsified-query-hypotheses]].
