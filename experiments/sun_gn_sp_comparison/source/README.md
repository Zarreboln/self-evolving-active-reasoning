# Adapter source references

These are unchanged copies of the experiment's own boundary/orchestration files. They expose the controlled algorithm choices for inspection. They are **not** a self-contained execution package: private cases, provider/runtime configuration and upstream vendor trees are deliberately omitted. Running these files is not part of public arithmetic verification.

- `run_compare.py`: GN arm/checkpoint scheduling and memory use.
- `baseline-adapters/arbench_adapter.py`: hash-bound native GN algorithm interface and declared parser overlay.
- `method-adapters/method_adapters.py`: cumulative DC and supervised ACE task interfaces.
- `method-adapters/public_feedback_sp.py`: named public-feedback-only SP adaptation.
- `method-adapters/upstream_loader.py` / `vendor-manifest.json`: external-source hash checks and definition selection. The vendor files themselves are not included.
- `sp-exploratory-01/controller.py`: the SP 33-episode/6-update schedule, retained as a source reference; its episode/runtime dependencies are omitted.
- `analysis/extract_gn_mechanism_aggregates.py`: descriptive update extraction for someone separately possessing the original private evidence. It checks update hashes against that evidence's result manifest; it does not establish provenance or reimplement the merge algorithm.

Use the two standard-library verifiers in the package README to reproduce the published numbers. A fresh generative experiment would require separately licensed upstream material, new task data, a reviewed execution scope and costs; anonymous case aliases do not reconstruct original provider requests or seeds. See [third-party provenance](../THIRD_PARTY.md).
