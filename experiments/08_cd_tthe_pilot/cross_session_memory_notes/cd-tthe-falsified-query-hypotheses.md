---
name: cd-tthe-falsified-query-hypotheses
description: "cd_tthe offline tests (2026-10-12) — structure-count prior and 2-step lookahead both lose to the uniform-combo joint entropy; don't re-propose them"
metadata:
  node_type: memory
  type: project
  originSessionId: 77b1998e-1fcc-4d96-ae2a-b19c569f566a
  modified: 2026-10-12T21:47:17.907Z
---

In cd_tthe (pilot_labeled_b10g3r2, b2r1g1 work, 2026-10-12), two plausible harness ideas were tested offline and rejected:
- Prior: the uniform-over-(split, table) prior fits the 30 determined triples best (log-lik -216.6). Structure-count priors (-223.8) and a random-process prior (-244.8) are worse.
- 2-step lookahead ranking: slightly worse than one-step joint entropy (mean 11.44 vs 11.08 queries).
- The rule-derived joint space is complete. A contradiction can't arise because only open rows are queried, so fallback code is dead.
- Query count spread per truth is large (5-17) from tie choice, so n=10 query differences are noise.

**Why:** these took real effort to establish. Outcome on the observed inventories (NOT<=2) is saturated at 100%.
**How to apply:** for new cd_tthe branches, look elsewhere (reply protocol, robustness). Judge using the label-free method in [[cd-tthe-strict-traces-label-leak]].
