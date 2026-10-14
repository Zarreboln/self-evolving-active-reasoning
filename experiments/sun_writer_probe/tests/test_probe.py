"""Synthetic-only implementation tests; this is not independent code review.

Numerical expectations are hand-specified here, independently of probe's
contrast helper. No real data, model, network or scorer implementation is used.
"""

from __future__ import annotations

import ast
import copy
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load_probe():
    spec = importlib.util.spec_from_file_location("sun_writer_probe_under_test", ROOT / "probe.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SCORES = {
    "toy_clock": {"A": 0.2, "B": 0.3, "C": 0.4, "D": 0.6},
    "toy_kite": {"A": 0.8, "B": 0.6, "C": 0.5, "D": 0.4},
}
CONTRASTS = {
    "toy_clock": {"B-A": 0.1, "D-C": 0.2, "C-A": 0.2, "D-B": 0.3, "interaction": 0.1},
    "toy_kite": {"B-A": -0.2, "D-C": -0.1, "C-A": -0.3, "D-B": -0.2, "interaction": 0.1},
}
MEANS = {"B-A": -0.05, "D-C": 0.05, "C-A": -0.05, "D-B": 0.05, "interaction": 0.1}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def write_rows(path, rows):
    Path(path).write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows), encoding="utf-8")


class ProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.probe = load_probe()

    def setUp(self):
        self.requests = self.probe.prepare(ROOT / "fixtures/manifest.json", ROOT / "protocol.json", synthetic_only=True)
        self.responses = self.make_responses(self.requests)

    def make_responses(self, requests):
        rows = []
        for request in requests:
            spec = request["protocol"]["document"]["scoring_spec"]
            rows.append({
                "format": "c2_response_v1", "data_kind": "synthetic",
                "case_id": request["case_id"], "cell": request["cell"],
                "request_id": request["request_id"], "payload_sha256": request["payload_sha256"],
                "scorer": {"name": spec["name"], "version": spec["version"], "metric": spec["metric"],
                           "spec_sha256": self.probe.digest(spec)},
                "status": "completed", "text": "  Synthetic response \u2603\n\tkept verbatim.  ",
                "finish_reason": "stop", "truncated": False,
                "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                "score": SCORES[request["case_id"]][request["cell"]], "error": None,
            })
        return rows

    def report(self, requests=None, responses=None):
        return self.probe.analyze(self.requests if requests is None else requests,
                                  self.responses if responses is None else responses)

    def cell_response(self, case_id="toy_clock", cell="D"):
        return next(row for row in self.responses if (row["case_id"], row["cell"]) == (case_id, cell))

    def assert_invalid_requests(self, requests):
        with self.assertRaises(self.probe.ValidationError):
            self.probe.analyze(requests, [])

    def assert_invalid_responses(self):
        with self.assertRaises(self.probe.ValidationError):
            self.report()

    def rehash(self, request):
        request["payload_sha256"] = self.probe.digest(request["payload"])
        request["request_id"] = self.probe.request_identity(request)

    @contextmanager
    def copied_inputs(self):
        with tempfile.TemporaryDirectory(prefix="_test_", dir=ROOT) as directory:
            target = Path(directory)
            shutil.copytree(ROOT / "fixtures", target / "fixtures")
            shutil.copy2(ROOT / "protocol.json", target / "protocol.json")
            manifest_path = target / "fixtures/manifest.json"
            yield target, manifest_path, json.loads(manifest_path.read_text(encoding="utf-8"))

    def refresh_source_digest(self, manifest_path, manifest, role):
        source = next(entry for entry in manifest["sources"] if entry["role"] == role)
        source["sha256"] = hashlib.sha256((manifest_path.parent / source["path"]).read_bytes()).hexdigest()
        write_json(manifest_path, manifest)

    def mutate_source(self, mutation, role="first"):
        with self.copied_inputs() as (target, manifest_path, manifest):
            source = next(entry for entry in manifest["sources"] if entry["role"] == role)
            source_path = manifest_path.parent / source["path"]
            rows = read_rows(source_path)
            mutation(rows)
            write_rows(source_path, rows)
            self.refresh_source_digest(manifest_path, manifest, role)
            return self.probe.prepare(manifest_path, target / "protocol.json", synthetic_only=True)

    def test_four_cells_per_case_same_model_settings_and_seed(self):
        self.assertEqual(len(self.requests), 8)
        self.assertEqual(len({row["request_id"] for row in self.requests}), 8)
        for case_id in SCORES:
            rows = [row for row in self.requests if row["case_id"] == case_id]
            self.assertEqual({row["cell"] for row in rows}, {"A", "B", "C", "D"})
            settings = [{key: row["payload"][key] for key in ("model", "temperature", "seed", "max_output_tokens")} for row in rows]
            self.assertTrue(all(item == settings[0] for item in settings))
            self.assertEqual(settings[0]["max_output_tokens"], 400)
            self.assertTrue(settings[0]["model"].startswith("synthetic://"))

    def test_exact_required_writer_instructions_and_shared_system_text(self):
        base = "Answer the puzzle using the recorded dialogue. Return only the final explanation."
        constrained = base + "\nUse only supported facts; do not present conjectures or denied/unknown claims as established."
        systems = set()
        for request in self.requests:
            messages = request["payload"]["messages"]
            self.assertEqual([message["role"] for message in messages], ["system", "user", "user"])
            self.assertEqual(messages[2]["content"], base if request["cell"] in ("A", "C") else constrained)
            systems.add(messages[0]["content"])
        self.assertEqual(len(systems), 1)

    def test_independently_expected_arithmetic_and_planned_means(self):
        report = self.report()
        self.assertEqual(report["data_kind"], "synthetic")
        self.assertEqual(report["planned_cases"], 2)
        self.assertEqual(report["planned_requests"], 8)
        for case in report["case_results"]:
            for name, expected in CONTRASTS[case["case_id"]].items():
                self.assertAlmostEqual(case["contrasts"][name], expected, delta=1e-12)
        for name, expected in MEANS.items():
            self.assertAlmostEqual(report["means"][name]["mean"], expected, delta=1e-12)
            self.assertEqual({key: report["means"][name][key] for key in ("expected", "available", "missing")},
                             {"expected": 2, "available": 2, "missing": 0})

    def test_static_response_fixture_is_bound_to_actual_requests(self):
        report = self.report(responses=read_rows(ROOT / "fixtures/responses.jsonl"))
        for name, expected in MEANS.items():
            self.assertAlmostEqual(report["means"][name]["mean"], expected, delta=1e-12)

    def test_response_and_request_order_do_not_change_report(self):
        baseline = self.report()
        self.assertEqual(self.report(list(reversed(self.requests)), list(reversed(self.responses))), baseline)
        self.assertEqual(self.report(self.requests[::2] + self.requests[1::2], self.responses[3:] + self.responses[:3]), baseline)

    def test_source_row_order_preserves_pairing_and_arithmetic(self):
        changed = self.mutate_source(lambda rows: rows.reverse(), role="bare")
        report = self.report(changed, self.make_responses(changed))
        for case in report["case_results"]:
            for name, expected in CONTRASTS[case["case_id"]].items():
                self.assertAlmostEqual(case["contrasts"][name], expected, delta=1e-12)

    def test_observation_projects_only_surface_question_reply_verbatim(self):
        manifest = json.loads((ROOT / "fixtures/manifest.json").read_text(encoding="utf-8"))
        source_rows = {}
        for source in manifest["sources"]:
            for row in read_rows(ROOT / "fixtures" / source["path"]):
                source_rows[(source["role"], row["case_id"])] = row
        for request in self.requests:
            original = source_rows[(request["source_identity"]["role"], request["case_id"])]
            expected = {"surface": original["surface"], "history": [{"question": turn["question"], "reply": turn["reply"]} for turn in original["history"]]}
            self.assertEqual(request["observation"], expected)
            self.assertEqual(json.loads(request["payload"]["messages"][1]["content"]), expected)
            self.assertEqual(set(request["observation"]), {"surface", "history"})

    def test_hidden_sentinels_do_not_enter_model_messages(self):
        sentinel = "HIDDEN_TEST_SENTINEL_DO_NOT_PROMPT_4c310b"
        def inject(rows):
            for row in rows:
                for key in ("gold", "keypoints", "prior_final_answer", "score", "provenance", "commentary", "acquisition_role", "file_path"):
                    row[key] = sentinel
                for turn in row["history"]:
                    turn["gold"] = sentinel
                    turn["commentary"] = sentinel
        changed = self.mutate_source(inject)
        for request in changed:
            self.assertNotIn(sentinel, json.dumps(request["payload"]["messages"]))

    def test_nonstandard_reply_is_not_stripped_cleaned_or_reclassified(self):
        original = "  Maybe?! UNKNOWN / denied? \u4e0d\u786e\u5b9a\n\t(not yes/no)  "
        changed = self.mutate_source(lambda rows: rows[0]["history"][0].update(reply=original))
        first_requests = [row for row in changed if row["source_identity"]["role"] == "first"]
        self.assertTrue(any(row["observation"]["history"][0]["reply"] == original for row in first_requests))

    def test_canonical_serialization_independent_hash(self):
        value = {"z": ["\u96ea", 2], "a": {"q": "\n", "n": None}}
        expected = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.assertEqual(self.probe.canonical_bytes(value), expected)
        self.assertEqual(self.probe.digest(value), hashlib.sha256(expected).hexdigest())

    def test_payload_identity_hashes_entire_payload(self):
        for request in self.requests:
            independently_serialized = json.dumps(request["payload"], sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.assertEqual(request["payload_sha256"], hashlib.sha256(independently_serialized).hexdigest())
            envelope = {key: value for key, value in request.items() if key != "request_id"}
            expected_id = hashlib.sha256(json.dumps(envelope, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()
            self.assertEqual(request["request_id"], expected_id)

    def test_all_request_identity_determinants_change_identity(self):
        baseline = self.requests[0]
        mutations = {
            "system_text": lambda row: row["payload"]["messages"][0].update(content="different fictional system text"),
            "writer_instruction": lambda row: row["payload"]["messages"][2].update(content="different writer instruction"),
            "dialogue": lambda row: row["payload"]["messages"][1].update(content="different fictional dialogue"),
            "model": lambda row: row["payload"].update(model="synthetic://writer-stub/v2"),
            "temperature": lambda row: row["payload"].update(temperature=0.5),
            "seed": lambda row: row["payload"].update(seed=999),
            "token_cap": lambda row: row["payload"].update(max_output_tokens=399),
            "protocol": lambda row: row["protocol"].update(sha256="0" * 64),
            "builder_version": lambda row: row["builder"].update(version="synthetic-test-other-builder"),
            "builder_code": lambda row: row["builder"].update(code_sha256="0" * 64),
            "manifest": lambda row: row["manifest"].update(sha256="0" * 64),
            "source_record": lambda row: row["source_identity"].update(record_id="synthetic-other-record"),
            "source_member": lambda row: row["source_identity"]["member"].update(sha256="0" * 64),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                changed = copy.deepcopy(baseline)
                mutate(changed)
                self.assertNotEqual(self.probe.request_identity(changed), baseline["request_id"])

    def test_changed_source_history_changes_affected_request_identity(self):
        changed = self.mutate_source(lambda rows: rows[0]["history"][0].update(reply="Different fictional observation."))
        original_ids = {(row["case_id"], row["cell"]): row["request_id"] for row in self.requests}
        for request in changed:
            self.assertNotEqual(request["request_id"], original_ids[(request["case_id"], request["cell"])])

    def test_changed_protocol_content_propagates_to_prepared_identity(self):
        with self.copied_inputs() as (target, manifest_path, manifest):
            protocol_path = target / "protocol.json"
            baseline = self.probe.prepare(manifest_path, protocol_path, synthetic_only=True)
            protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
            protocol["version"] = "synthetic_protocol_identity_test"
            protocol["system_text"] += "\nA harmless synthetic identity-test suffix."
            write_json(protocol_path, protocol)
            changed = self.probe.prepare(manifest_path, protocol_path, synthetic_only=True)
            self.assertEqual(len(baseline), len(changed))
            for original, current in zip(baseline, changed):
                self.assertNotEqual(original["request_id"], current["request_id"])
                self.assertNotEqual(original["payload_sha256"], current["payload_sha256"])

    def test_prepare_without_synthetic_flag_is_rejected(self):
        with self.assertRaises(self.probe.ValidationError):
            self.probe.prepare(ROOT / "fixtures/manifest.json", ROOT / "protocol.json")

    def test_real_manifest_is_rejected_even_with_synthetic_flag(self):
        with self.copied_inputs() as (target, manifest_path, manifest):
            for kind in ("real", "historical"):
                with self.subTest(data_kind=kind):
                    manifest["data_kind"] = kind
                    write_json(manifest_path, manifest)
                    with self.assertRaises(self.probe.ValidationError):
                        self.probe.prepare(manifest_path, target / "protocol.json", synthetic_only=True)

    def test_real_source_record_is_rejected_even_with_synthetic_flag(self):
        with self.assertRaises(self.probe.ValidationError):
            self.mutate_source(lambda rows: rows[0].update(data_kind="historical"))

    def test_real_protocol_is_rejected_even_with_synthetic_flag(self):
        with self.copied_inputs() as (target, manifest_path, manifest):
            protocol = json.loads((target / "protocol.json").read_text(encoding="utf-8"))
            protocol["data_kind"] = "real"
            write_json(target / "protocol.json", protocol)
            with self.assertRaises(self.probe.ValidationError):
                self.probe.prepare(manifest_path, target / "protocol.json", synthetic_only=True)

    def test_protocol_operational_model_or_changed_token_ceiling_is_rejected(self):
        with self.copied_inputs() as (target, manifest_path, manifest):
            protocol_path = target / "protocol.json"
            baseline = json.loads(protocol_path.read_text(encoding="utf-8"))
            for field, value in (("model", "real-provider-model"), ("max_output_tokens", 399),
                                 ("max_output_tokens", 401), ("max_output_tokens", True)):
                with self.subTest(field=field, value=value):
                    protocol = copy.deepcopy(baseline)
                    protocol["settings"][field] = value
                    write_json(protocol_path, protocol)
                    with self.assertRaises(self.probe.ValidationError):
                        self.probe.prepare(manifest_path, protocol_path, synthetic_only=True)

    def test_duplicate_manifest_or_protocol_json_keys_are_rejected(self):
        with self.copied_inputs() as (target, manifest_path, manifest):
            baseline_manifest = manifest_path.read_text(encoding="utf-8")
            manifest_path.write_text('{"format":"c2_manifest_v1",' + baseline_manifest.lstrip()[1:], encoding="utf-8")
            with self.assertRaises(self.probe.ValidationError):
                self.probe.prepare(manifest_path, target / "protocol.json", synthetic_only=True)
            manifest_path.write_text(baseline_manifest, encoding="utf-8")
            protocol_path = target / "protocol.json"
            protocol_text = protocol_path.read_text(encoding="utf-8")
            protocol_path.write_text('{"format":"c2_protocol_v1",' + protocol_text.lstrip()[1:], encoding="utf-8")
            with self.assertRaises(self.probe.ValidationError):
                self.probe.prepare(manifest_path, protocol_path, synthetic_only=True)

    def test_duplicate_jsonl_keys_and_nonfinite_json_constants_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix="_test_", dir=ROOT) as directory:
            path = Path(directory) / "bad.jsonl"
            for text in ('{"data_kind":"synthetic","data_kind":"synthetic"}\n',
                         '{"score":NaN}\n', '{"score":Infinity}\n', '{"score":-Infinity}\n'):
                with self.subTest(text=text):
                    path.write_text(text, encoding="utf-8")
                    with self.assertRaises(self.probe.ValidationError):
                        self.probe.read_jsonl(path)

    def test_source_byte_digest_mismatch_is_rejected(self):
        with self.copied_inputs() as (target, manifest_path, manifest):
            source_path = manifest_path.parent / manifest["sources"][0]["path"]
            with source_path.open("ab") as output:
                output.write(b"\n")
            with self.assertRaises(self.probe.ValidationError):
                self.probe.prepare(manifest_path, target / "protocol.json", synthetic_only=True)

    def test_duplicate_manifest_entries_are_rejected(self):
        with self.copied_inputs() as (target, manifest_path, manifest):
            manifest["sources"].append(copy.deepcopy(manifest["sources"][0]))
            write_json(manifest_path, manifest)
            with self.assertRaises(self.probe.ValidationError):
                self.probe.prepare(manifest_path, target / "protocol.json", synthetic_only=True)

    def test_traversal_absolute_and_backslash_source_paths_are_rejected(self):
        with self.copied_inputs() as (target, manifest_path, manifest):
            for path in ("../protocol.json", "/etc/passwd", "nested/../../protocol.json", "..\\protocol.json"):
                with self.subTest(path=path):
                    manifest["sources"][0]["path"] = path
                    write_json(manifest_path, manifest)
                    with self.assertRaises(self.probe.ValidationError):
                        self.probe.prepare(manifest_path, target / "protocol.json", synthetic_only=True)

    def test_symlink_escaping_manifest_directory_is_rejected(self):
        with self.copied_inputs() as (target, manifest_path, manifest):
            source_path = manifest_path.parent / manifest["sources"][0]["path"]
            outside_source = target / "outside_manifest_directory.jsonl"
            outside_source.write_bytes(source_path.read_bytes())
            source_path.unlink()
            source_path.symlink_to(outside_source)
            with self.assertRaises(self.probe.ValidationError):
                self.probe.prepare(manifest_path, target / "protocol.json", synthetic_only=True)

    def test_duplicate_source_case_is_rejected(self):
        with self.assertRaises(self.probe.ValidationError):
            self.mutate_source(lambda rows: rows.append(copy.deepcopy(rows[0])))

    def test_missing_source_case_is_rejected(self):
        with self.assertRaises(self.probe.ValidationError):
            self.mutate_source(lambda rows: rows.pop())

    def test_surface_mismatch_is_rejected(self):
        with self.assertRaises(self.probe.ValidationError):
            self.mutate_source(lambda rows: rows[0].update(surface="An unrelated imaginary puzzle."))

    def test_mixed_source_runs_are_rejected(self):
        with self.assertRaises(self.probe.ValidationError):
            self.mutate_source(lambda rows: rows[0].update(source_run_id="synthetic-run:different"))

    def test_missing_and_duplicate_request_cells_are_design_errors(self):
        self.assert_invalid_requests(self.requests[:-1])
        self.assert_invalid_requests(self.requests + [copy.deepcopy(self.requests[0])])

    def test_missing_whole_request_case_is_design_error(self):
        self.assert_invalid_requests([row for row in self.requests if row["case_id"] == "toy_clock"])

    def test_request_surface_and_setting_inconsistency_remain_invalid_after_rehash(self):
        mutations = {
            "surface": lambda row: row["observation"].update(surface="Inconsistent imaginary puzzle"),
            "seed": lambda row: row["payload"].update(seed=666),
            "temperature": lambda row: row["payload"].update(temperature=0.75),
            "model": lambda row: row["payload"].update(model="synthetic://writer-stub/other"),
            "cap": lambda row: row["payload"].update(max_output_tokens=399),
            "cell": lambda row: row.update(cell="B"),
            "source_run": lambda row: row["source_identity"].update(source_run_id="synthetic-run:other"),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                changed = copy.deepcopy(self.requests)
                mutate(changed[0])
                self.rehash(changed[0])
                self.assert_invalid_requests(changed)

    def test_payload_boolean_is_not_a_numeric_setting_even_after_rehash(self):
        changed = copy.deepcopy(self.requests)
        changed[0]["payload"]["temperature"] = False
        self.rehash(changed[0])
        self.assert_invalid_requests(changed)

    def test_dot_is_not_a_source_file_path(self):
        with self.assertRaises(self.probe.ValidationError):
            self.probe.safe_relative(".")

    def test_tampered_request_identity_and_payload_digest_are_rejected(self):
        for field in ("request_id", "payload_sha256"):
            with self.subTest(field=field):
                changed = copy.deepcopy(self.requests)
                changed[0][field] = "0" * 64
                self.assert_invalid_requests(changed)

    def test_historical_requests_and_responses_are_rejected_by_analyze(self):
        changed = copy.deepcopy(self.requests)
        changed[0]["data_kind"] = "historical"
        self.rehash(changed[0])
        self.assert_invalid_requests(changed)
        self.responses[0]["data_kind"] = "historical"
        self.assert_invalid_responses()

    def test_duplicate_response_is_rejected(self):
        self.responses.append(copy.deepcopy(self.responses[0]))
        self.assert_invalid_responses()

    def test_foreign_response_case_is_rejected(self):
        self.responses[0]["case_id"] = "unplanned_fictional_case"
        self.assert_invalid_responses()

    def test_response_identity_payload_and_cell_mismatches_are_rejected(self):
        baseline = copy.deepcopy(self.responses)
        for field, value in (("request_id", "0" * 64), ("payload_sha256", "0" * 64), ("cell", "Z")):
            with self.subTest(field=field):
                self.responses = copy.deepcopy(baseline)
                self.responses[0][field] = value
                self.assert_invalid_responses()

    def test_scorer_identity_is_required_and_exact(self):
        baseline = copy.deepcopy(self.responses)
        for field in ("name", "version", "metric", "spec_sha256"):
            with self.subTest(field=field):
                self.responses = copy.deepcopy(baseline)
                self.responses[0]["scorer"][field] = "mismatch"
                self.assert_invalid_responses()
        self.responses = copy.deepcopy(baseline)
        del self.responses[0]["scorer"]
        self.assert_invalid_responses()

    def test_nonfinite_boolean_out_of_bounds_and_non_numeric_scores_are_rejected(self):
        baseline = copy.deepcopy(self.responses)
        for value in (float("nan"), float("inf"), -float("inf"), True, False, -0.01, 1.01, "0.2", None):
            with self.subTest(score=repr(value)):
                self.responses = copy.deepcopy(baseline)
                self.responses[0]["score"] = value
                self.assert_invalid_responses()

    def test_score_bounds_and_known_zero_usage_are_valid(self):
        self.responses[0]["score"] = 0
        self.responses[1]["score"] = 1
        self.responses[0]["usage"] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        report = self.report()
        case = next(row for row in report["case_results"] if row["case_id"] == self.responses[0]["case_id"])
        self.assertEqual(case["cells"][self.responses[0]["cell"]]["score"], 0)
        self.assertEqual(case["cells"][self.responses[1]["cell"]]["score"], 1)
        self.assertEqual(case["cells"][self.responses[0]["cell"]]["response"]["usage"],
                         {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
        for field in ("input_tokens", "output_tokens", "total_tokens"):
            self.assertEqual(report["usage"]["fields"][field]["known"], 8)
            self.assertEqual(report["usage"]["fields"][field]["unknown"], 0)

    def test_nonnegative_integer_or_null_usage(self):
        baseline = copy.deepcopy(self.responses)
        for field in ("input_tokens", "output_tokens", "total_tokens"):
            for value in (-1, 1.5, True, False, "1", float("nan"), float("inf")):
                with self.subTest(field=field, value=repr(value)):
                    self.responses = copy.deepcopy(baseline)
                    self.responses[0]["usage"][field] = value
                    self.assert_invalid_responses()

    def test_known_usage_total_must_equal_input_plus_output(self):
        self.responses[0]["usage"]["total_tokens"] = 999
        self.assert_invalid_responses()

    def test_full_raw_text_and_finish_truncation_are_retained(self):
        response = self.cell_response()
        response.update(finish_reason="length", truncated=True)
        report = self.report()
        case = next(row for row in report["case_results"] if row["case_id"] == "toy_clock")
        returned = case["cells"]["D"]["response"]
        self.assertEqual(returned, response)
        self.assertEqual(returned["text"], "  Synthetic response \u2603\n\tkept verbatim.  ")
        self.assertTrue(returned["truncated"])
        self.assertEqual(returned["finish_reason"], "length")

    def test_unknown_usage_stays_null_with_coverage_counts(self):
        response = self.cell_response("toy_kite", "C")
        response["usage"] = {"input_tokens": None, "output_tokens": None, "total_tokens": None}
        report = self.report()
        case = next(row for row in report["case_results"] if row["case_id"] == "toy_kite")
        self.assertEqual(case["cells"]["C"]["response"]["usage"], response["usage"])
        for field in response["usage"]:
            self.assertEqual(report["usage"]["fields"][field]["known"], 7)
            self.assertEqual(report["usage"]["fields"][field]["unknown"], 1)
            self.assertIsNone(report["usage"]["fields"][field]["total_if_complete"])

    def assert_one_missing_d_cell(self, report, diagnostic_reason=None):
        case = next(row for row in report["case_results"] if row["case_id"] == "toy_clock")
        self.assertIsNone(case["cells"]["D"]["score"])
        self.assertAlmostEqual(case["contrasts"]["B-A"], 0.1, delta=1e-12)
        self.assertAlmostEqual(case["contrasts"]["C-A"], 0.2, delta=1e-12)
        for name in ("D-C", "D-B", "interaction"):
            self.assertIsNone(case["contrasts"][name])
            self.assertEqual(report["means"][name], {"mean": None, "expected": 2, "available": 1, "missing": 1})
        for name in ("B-A", "C-A"):
            self.assertAlmostEqual(report["means"][name]["mean"], MEANS[name], delta=1e-12)
        diagnostics = [row for row in report["diagnostics"] if (row["case_id"], row["cell"]) == ("toy_clock", "D")]
        self.assertEqual(len(diagnostics), 1)
        if diagnostic_reason is not None:
            self.assertIn(diagnostic_reason, diagnostics[0]["reason"])

    def test_missing_response_is_not_zero_and_suppresses_only_affected_means(self):
        self.responses.remove(self.cell_response())
        report = self.report()
        self.assert_one_missing_d_cell(report)
        self.assertEqual(report["usage"]["planned"], 8)
        self.assertEqual(report["usage"]["returned"], 7)
        self.assertEqual(report["usage"]["missing_responses"], 1)
        for field in ("input_tokens", "output_tokens", "total_tokens"):
            self.assertEqual(report["usage"]["fields"][field]["unknown"], 1)
            self.assertIsNone(report["usage"]["fields"][field]["total_if_complete"])

    def test_explicit_failed_response_is_not_zero_or_dropped(self):
        response = self.cell_response()
        response.update(status="failed", score=None, error="synthetic simulated failure", finish_reason="error", truncated=None)
        response["usage"] = {key: None for key in response["usage"]}
        report = self.report()
        self.assert_one_missing_d_cell(report)
        case = next(row for row in report["case_results"] if row["case_id"] == "toy_clock")
        self.assertEqual(case["cells"]["D"]["response"], response)
        self.assertEqual(report["usage"]["returned"], 8)
        self.assertEqual(report["usage"]["missing_responses"], 0)

    def test_unscored_response_preserves_diagnostic_and_denominator(self):
        self.cell_response().update(status="unscored", score=None, error="synthetic scorer deliberately unavailable")
        self.assert_one_missing_d_cell(self.report())

    def test_all_responses_missing_retains_every_case_and_cell(self):
        report = self.report(responses=[])
        self.assertEqual(len(report["case_results"]), 2)
        self.assertEqual(len(report["diagnostics"]), 8)
        for case in report["case_results"]:
            self.assertEqual(set(case["cells"]), {"A", "B", "C", "D"})
            self.assertTrue(all(value is None for value in case["contrasts"].values()))
        for value in report["means"].values():
            self.assertEqual(value, {"mean": None, "expected": 2, "available": 0, "missing": 2})

    def test_inconsistent_status_score_error_are_rejected(self):
        baseline = copy.deepcopy(self.responses)
        mutations = [
            {"status": "failed", "score": 0.2, "error": "simulated error"},
            {"status": "failed", "score": None, "error": None},
            {"status": "unscored", "score": None, "error": ""},
            {"status": "completed", "error": "inconsistent error"},
            {"status": "unknown-status"},
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                self.responses = copy.deepcopy(baseline)
                self.responses[0].update(mutation)
                self.assert_invalid_responses()

    def test_invalid_text_finish_or_truncation_are_rejected(self):
        baseline = copy.deepcopy(self.responses)
        for field, value in (("text", 123), ("finish_reason", 1), ("truncated", 1)):
            with self.subTest(field=field):
                self.responses = copy.deepcopy(baseline)
                self.responses[0][field] = value
                self.assert_invalid_responses()


class OfflineBoundaryTests(unittest.TestCase):
    def test_cli_prepare_requires_explicit_synthetic_only_flag(self):
        with tempfile.TemporaryDirectory(prefix="_test_", dir=ROOT) as directory:
            completed = subprocess.run(
                [sys.executable, "-B", "-S", str(ROOT / "probe.py"), "prepare",
                 "--manifest", str(ROOT / "fixtures/manifest.json"),
                 "--protocol", str(ROOT / "protocol.json"),
                 "--out", str(Path(directory) / "requests.jsonl")],
                capture_output=True, text=True, cwd=ROOT,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertFalse((Path(directory) / "requests.jsonl").exists())

    def test_code_imports_only_standard_library_and_no_network_client(self):
        tree = ast.parse((ROOT / "probe.py").read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        prohibited = {"socket", "urllib", "http", "ftplib", "smtplib", "requests",
                      "openai", "anthropic", "boto3", "httpx", "aiohttp"}
        self.assertFalse(imported & prohibited, imported & prohibited)
        self.assertTrue(imported <= set(sys.stdlib_module_names), imported - set(sys.stdlib_module_names))


if __name__ == "__main__":
    unittest.main()
