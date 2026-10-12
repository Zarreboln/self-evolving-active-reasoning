---
name: cd-tthe-read-system-prompt-safely
description: "cd_tthe: get the filled system prompt by a 1-instance diagnostic harness, not by reading the mt_puzzles templates (off-limits)"
metadata:
  type: reference
---

The benchmark rule text lives in mt_puzzles/ (off-limits), and bridge.py only points to it. A safe way to see it is to copy the check script with idx=[one instance] and a separate out_dir. Then run a throwaway harness that makes one self.llm call with env.system_prompt as the user message (max_tokens=1). Afterwards, read only `trace[0]["messages"]` from the res json, never `gold` ([[cd-tthe-strict-traces-label-leak]]), and delete that diag dir.
Rules (2026-10-12): every circuit has exactly 2 binary gates, NOTs can go on any signal and stack, no circuit is constant, and the whole inventory is used.
**How to apply:** use this to verify the rules before encoding a hypothesis space.
