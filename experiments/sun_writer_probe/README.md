# Offline fixed-dialogue writer probe

This standard-library-only component prepares a four-cell writer probe and analyzes returned
responses offline. This milestone uses explicitly fictional data, a non-operational mock model,
and hand-assigned synthetic scores. It makes no model calls and reports no empirical findings.
Independent code review remains pending.

## Design

Each stable case identity has four requests:

| Cell | Recorded dialogue | Writer instruction |
| --- | --- | --- |
| A | Bare | Plain |
| B | Bare | Evidence-constrained |
| C | First-selected | Plain |
| D | First-selected | Evidence-constrained |

Both writers say: "Answer the puzzle using the recorded dialogue. Return only the final
explanation." The evidence-constrained writer additionally says: "Use only supported facts;
do not present conjectures or denied/unknown claims as established."

The per-case contrasts are `B-A`, `D-C`, `C-A`, `D-B`, and `(D-C)-(B-A)`. The first two measure
instruction effects conditional on the supplied history. The next two measure history utility
conditional on a writer; they are not equal-compute acquisition effects. The last measures the
interaction for these two sources and instructions. These prompts are standardized probes,
not the original evolved multi-draft finalizer.

## Run the synthetic demonstration

From this directory:

```sh
python -B -S -m unittest discover -s tests -v
python -B -S probe.py prepare --manifest fixtures/manifest.json --protocol protocol.json --synthetic-only --out outputs/requests.jsonl
python -B -S probe.py analyze --requests outputs/requests.jsonl --responses fixtures/responses.jsonl --out outputs/contrasts.json
```

`prepare` requires `--synthetic-only` and rejects real or historical data even when that flag is
present. Every input and output declares `data_kind=synthetic`. The mock settings in the
protocol are test settings, not approvals for a real run. All four cells share model,
temperature, per-case seed, and a 400-output-token cap. Shared seeds do not guarantee
synchronized backend draws.

The accepted synthetic protocol fixes the cap at exactly 400. A different cap is an identity
change, but preparation rejects it in this milestone rather than silently changing the protocol.

The fixtures contain two fictional cases, `toy_clock` and `toy_kite`. Their scores are literal
test inputs. The second case includes negative effects so the demonstration does not assume
that either instruction or history improves the score. No score was produced by a model or
an automatic judge.

## Identities and schemas

Canonical serialization is sorted-key, compact UTF-8 JSON with `ensure_ascii=False` and
`allow_nan=False`. All identity digests are SHA-256. `payload_sha256` hashes the entire
payload, including every message and setting; `request_id` hashes the full request envelope
except its own `request_id` field. Human cell/source labels are separate from these identities.

| File or record | Identity and retained fields |
| --- | --- |
| Manifest | `c2_manifest_v1`, synthetic source-run identity, and unique `bare`/`first` entries with safe relative paths and exact file-byte hashes |
| Source row | Stable `case_id` and `record_id`, source/run IDs, puzzle `surface`, and recorded question/reply history |
| Request | `c2_request_v1`, cell/writer, full payload and its hash, builder version/code hash, full manifest/protocol documents with canonical and byte hashes, and source/member/record/line/surface/dialogue identities |
| Response | `c2_response_v1`, case/cell, matching request and payload identities, explicit scorer identity, status, full `text`, finish/truncation fields, usage, score, and error |
| Report | `c2_analysis_v1`, planned case/request counts, all case/cell rows, per-case contrasts, means with completeness counts, diagnostics, usage coverage, scorer, provenance, and limitations |

The source member is explicitly identified as `standalone_jsonl`. Request preparation checks
source file bytes against the manifest. Changing source content, messages, model/settings,
protocol, manifest, or builder version/code changes the bound identity. Editing `probe.py`
therefore invalidates existing response-fixture request IDs; reprepare and deliberately rebind
the synthetic response fixture before analyzing it. A free-text run label is not source proof.

Scorer identity contains `name`, `version`, `metric`, and `spec_sha256`, the canonical digest of
the protocol's `scoring_spec`. The fixture specification identifies hand assignment of
synthetic test values. There is no guessed F1 implementation or approved real-data target.

Response `status` is `completed`, `failed`, or `unscored`. A completed response requires a
bounded finite score and no error. Failed/unscored responses require a `null` score and a
nonempty diagnostic error. `finish_reason` is a string or `null`, and `truncated` is a boolean
or `null`. Usage has `input_tokens`, `output_tokens`, and `total_tokens`, each an integer or
`null`. The full returned response is retained in its report cell. An absent response receives
a missing diagnostic and `response=null`.

## Pairing, prompts, and completeness

Cases are paired by stable IDs, not input row order. Source files must have the same case set
and matching puzzle surfaces. The builder preserves recorded questions and replies verbatim,
including nonstandard replies. It does not clean, reclassify, strip, or truncate them. Prompt
fields are allowlisted; acquisition-role labels, gold, keypoints, earlier final answers, scores,
provenance, paths, and expected-success commentary stay out of model messages. The sentinel
tests verify exclusion for these fixtures; they do not prove that arbitrary historical observed
text could never carry privileged information.

A malformed request design is an error: missing or duplicate cells, inconsistent settings,
different case sets, or mismatching surfaces prevent a valid report. Duplicate, foreign,
identity-mismatched, and invalid-score responses are errors. Missing responses and explicit
failed/unscored responses remain diagnostic rows; they are never scored as zero.

Each mean keeps its planned matched-case denominator. If any required per-case contrast is
unavailable, that mean is `null`, with expected, available, and missing counts. Unaffected
contrasts may remain available. There are no complete-case means, imputations, penalties,
significance tests, bootstraps, or winner selection.

Unknown usage is `null`, with coverage counts, never silently zero. Known usage must be
nonnegative integers. Responses retain their full returned text, completion or failure state,
and finish and truncation information. Synthetic scores are finite values in `[0, 1]`;
booleans are invalid scores. These hand-assigned values test arithmetic only and do not
implement an F1 scorer.

## Future gates and publication

A real experiment still needs reviewed source/configuration proof, selected development cases,
historical reply policy, scorer and target hashes, approved model/settings/endpoint, request
budgets, retries, and cancellation policy. No real requests are prepared by this milestone.

`PUBLICATION_ALLOWLIST.json` gives the exact additive publication-file preview. Only explicitly
listed code, fictional fixtures, reproduction documentation, and synthetic demonstration files
are candidates. Inclusion in that preview is not publication, a commit, or review approval.
All other output filenames are ignored; only the two exact demo filenames are exceptions.
