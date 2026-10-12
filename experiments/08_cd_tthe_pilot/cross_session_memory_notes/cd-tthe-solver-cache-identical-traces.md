---
name: cd-tthe-solver-cache-identical-traces
description: "cd_tthe solver replies are cached per prompt, so a candidate whose traces match its parent's made no effective change"
metadata:
  node_type: memory
  type: project
  originSessionId: d797e323-95a9-49cd-b4e7-bc668b65ddb8
  modified: 2026-10-12T21:40:41.935Z
---

In meta/cd_tthe, the solver cache (`logs/<run>/solver_cache`) returns the same reply for the same prompt, and `tokens_out/ep=0` in a run summary means every reply came from the cache. So if a candidate's query sequence matches its parent's on every instance, its changes never reached the model or never ran. Example from the b2 audit (2026-10-12): b2r0g1's docstring claims "spend leftover budget checking implied rows", but the code still breaks when no row is open, and its traces are identical to b1r1g2's.

**Why:** if you go by docstrings and proposal cards, you can credit a sibling with a mechanism it doesn't have.
**How to apply:** when auditing branches, diff the code against the parent and compare the query histories per instance before you credit a claimed change. A re-run of an unchanged prompt path reproduces the old trace exactly, so it adds no new evidence. Judge from .md files and history replay only ([[cd-tthe-strict-traces-label-leak]]).
