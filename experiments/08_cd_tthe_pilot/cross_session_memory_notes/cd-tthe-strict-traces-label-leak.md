---
name: cd-tthe-strict-traces-label-leak
description: "In cd_tthe strict-mode runs, trace/check .json files contain gold outcome fields; judge only from .md traces and history"
metadata:
  node_type: memory
  type: project
  originSessionId: 102606a2-b80e-47ea-820a-578dd4894dad
  modified: 2026-10-12T21:25:53.307Z
---

In meta/cd_tthe strict-signal runs, the trace `.json` files (logs/<run>/traces/...) and check `res_*.json` files carry a `gold` field (exact / ckt / guess) and an `oracle_agree` signal. The strict task says no evaluation beyond "did it run" is given.

**Why:** reading these leaks the label the strict arm is supposed to withhold.
**How to apply:** use the `.md` traces, plus `history`/`trace`/`guess_text` from the JSON. Never read `gold`. To judge progress label-free, replay the history through the harness's rules-derived candidate space (unique candidate per circuit at the end, and the entropy of each accepted query versus the best one available).
