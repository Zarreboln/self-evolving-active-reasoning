"""Verify the public GN UoT aggregate without models, network, or private data."""
import json
from pathlib import Path
import re
import sys


def require(condition, message):
    if not condition:
        raise SystemExit("FAIL: " + message)


path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("results.json")
data = json.loads(path.read_bytes())

require(data["status"] == "PASS_NATIVE_UOT_OFFICIAL100_CAP2048_MIT91_CLOUD9_COMBINED_INTEGRITY",
        "unexpected status")
require(data["task"] == "GN" and data["method"] == "native_uot_depth1", "scope differs")
require(data["cases"] == 100, "official case count differs")

segments = data["execution_segments"]
require(segments["mit_completed_cases"] == 91, "MIT completed count differs")
require(segments["cloud_suffix_cases"] == 9, "cloud suffix count differs")
require(segments["mit_completed_cases"] + segments["cloud_suffix_cases"] == data["cases"],
        "segments do not total 100")
require(segments["original_execution_failures_replaced"] == 9, "failure replacement differs")
require(segments["old_successful_cases_repeated"] == 0, "a successful MIT case was repeated")
require(segments["source_journals"] == 4 and len(data["source_digests"]) == 4,
        "source count differs")
require(sum(x["selected_completed_cases"] for x in data["source_digests"]) == 100,
        "source-selected counts do not total 100")

usage = data["usage"]
require(usage["exact_native_requests"] == 12803, "request count differs")
require(usage["actual_input_tokens"] == 6221091, "input-token count differs")
require(usage["actual_output_tokens"] == 16085099, "output-token count differs")
require(sum(usage["length_finishes"].values()) == 4727, "length-finish total differs")

metrics = data["metrics"]
require(metrics["official_successes"] == metrics["harness_final_successes"] == 100,
        "harness endpoint differs")
require(metrics["hit_at_any"] == 100, "hit-at-any differs")
require(metrics["actor_final_evaluated"] == metrics["actor_final_successes"] == 0,
        "actor-final must remain unavailable, not imputed")
require(metrics["selected_path_environment_queries_total"] == 774, "query total differs")
require(abs(metrics["selected_path_environment_queries_mean"] - 7.74) < 1e-12,
        "query mean differs")
require(metrics["selected_path_environment_queries_total"] / data["cases"]
        == metrics["selected_path_environment_queries_mean"], "query arithmetic differs")

audit = data["audit"]
require(audit["model_calls"] == audit["api_calls"] == audit["gpu_jobs"] == 0,
        "audit must be zero-model")
require(audit["source_journals_verified_independently"] is True, "journals not verified")
require(audit["journals_merged"] is False, "journals must not be re-chained")
require(audit["scientific_parameters_changed"] is False, "scientific parameters changed")

hex64 = re.compile(r"[0-9a-f]{64}").fullmatch
for key in ("assignment_sha256", "combined_outcomes_sha256",
            "private_audit_result_sha256", "auditor_sha256"):
    require(hex64(audit[key]) is not None, key + " is not SHA-256")
for source in data["source_digests"]:
    for key in ("plan_sha256", "result_sha256", "journal_sha256"):
        require(hex64(source[key]) is not None, source["segment"] + " " + key + " differs")

require(data["model_adaptation_disclosed"] is True, "model adaptation not disclosed")
require(data["private_material_included"] is False, "private-material flag differs")
require(len(data["claim_limits"]) >= 5, "claim limits missing")

text = json.dumps(data, sort_keys=True).lower()
for forbidden in ("private-cases.json", "course_kimi_api_key", "runtime_call_id",
                  "secret_answer", "interaction_transcript"):
    require(forbidden not in text, "private field leaked: " + forbidden)

print("PASS: GN UoT public aggregate checks")
