#!/usr/bin/env python3
import json
import pathlib
import sys


def require(condition, message):
    if not condition:
        raise AssertionError(message)


path = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "results.json")
data = json.loads(path.read_text(encoding="utf-8"))

require(data["schema"] == "gn_posttraining_public_v1", "unexpected schema")
training = data["training_data"]
require(training["eligible_states"] + training["excluded_states"] == training["total_states"], "state accounting mismatch")
require(training["solved_episodes"] + training["max_states_exhausted"] == training["episodes"], "episode accounting mismatch")
require(training["invalid_responses"] == 0, "invalid responses must remain disclosed")
require(training["token_inspection"]["rows_dropped"] == 0, "dropped rows must remain zero")
require(training["token_inspection"]["tokens_truncated"] == 0, "token truncation must remain zero")
require(training["token_inspection"]["maximum_tokens"] <= training["token_inspection"]["training_limit_tokens"], "token limit exceeded")

sft = data["sft"]
require(sft["optimizer_steps"] == sft["expected_optimizer_steps"] == 1371, "SFT step mismatch")
evaluation = sft["evaluation"]
require(evaluation["cases"] == 100, "SFT evaluation must cover 100 cases")
require(evaluation["official_successes"] == 1, "unexpected SFT success count")
require(evaluation["matched_base_official_successes"] == 0, "unexpected base success count")
require(evaluation["completed_model_calls"] == evaluation["audit_verified_model_responses"], "SFT response audit mismatch")

dpo = data["dpo"]
require(dpo["attempted_cases"] == 8 and dpo["complete_cases"] == 0, "DPO failure accounting changed")
require(dpo["effect_conclusion"] == "UNAVAILABLE", "DPO effect must not be inferred")

guards = data["claim_guards"]
require(not any(guards.values()), "all public claim guards must remain false")
print("PASS: GN post-training public aggregate checks")
