#!/usr/bin/env python3
"""Synthetic-only, standard-library compiler/analyzer; no inference or scoring client."""
import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys

BUILDER_VERSION = "c2-offline-v1"
BASE_INSTRUCTION = "Answer the puzzle using the recorded dialogue. Return only the final explanation."
EVIDENCE_INSTRUCTION = "Use only supported facts; do not present conjectures or denied/unknown claims as established."
CELLS = {
    "A": {"source": "bare", "writer": "plain"},
    "B": {"source": "bare", "writer": "evidence"},
    "C": {"source": "first", "writer": "plain"},
    "D": {"source": "first", "writer": "evidence"},
}
CONTRASTS = {
    "B-A": ("B", "A"), "D-C": ("D", "C"),
    "C-A": ("C", "A"), "D-B": ("D", "B"),
    "interaction": ("A", "B", "C", "D"),
}
USAGE_FIELDS = ("input_tokens", "output_tokens", "total_tokens")


class ValidationError(ValueError):
    """Invalid input; callers must not treat it as a valid experimental report."""


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def canonical_bytes(obj):
    try:
        return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValidationError("Not finite canonical UTF-8 JSON: " + str(exc)) from exc


def digest(obj):
    return hashlib.sha256(canonical_bytes(obj)).hexdigest()


def byte_digest(raw):
    return hashlib.sha256(raw).hexdigest()


def request_identity(request):
    return digest({k: v for k, v in request.items() if k != "request_id"})


def reject_constant(value):
    raise ValidationError("Nonfinite JSON constant: " + value)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON object key: " + key)
        result[key] = value
    return result


def decode_json(raw, label):
    try:
        return json.loads(raw, object_pairs_hook=unique_object, parse_constant=reject_constant)
    except (ValueError, UnicodeError) as exc:
        raise ValidationError("Invalid JSON in " + str(label) + ": " + str(exc)) from exc


def load_json(path):
    return decode_json(Path(path).read_bytes(), path)


def read_jsonl(path):
    rows = []
    for number, line in enumerate(Path(path).read_bytes().splitlines(keepends=True), 1):
        require(line.strip(), "Blank JSONL row in " + str(path))
        row = decode_json(line, str(path) + ":" + str(number))
        require(isinstance(row, dict), "JSONL rows must be objects")
        rows.append(row)
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(canonical_bytes(row) + b"\n" for row in rows))


def string(value, name, nonempty=True):
    require(isinstance(value, str) and (bool(value) or not nonempty), name + " must be a string")
    try:
        value.encode("utf-8")
    except UnicodeError as exc:
        raise ValidationError(name + " must be valid UTF-8") from exc


def sha(value, name):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), name + " must be a SHA-256 hex digest")


def integer(value, name):
    require(type(value) is int and value >= 0, name + " must be a nonnegative integer (not bool)")


def bounded_number(value, name, high=1):
    require(type(value) in (int, float), name + " must be numeric (not bool)")
    try:
        finite = math.isfinite(value)
    except (OverflowError, TypeError):
        finite = False
    require(finite and 0 <= value <= high, name + " must be finite and in [0," + str(high) + "]")


def synthetic(document, expected_format):
    require(isinstance(document, dict), "Input must be an object")
    require(document.get("data_kind") == "synthetic", "Only data_kind=synthetic is supported; real/historical preparation is disabled")
    require(document.get("format") == expected_format, "Unsupported format: " + expected_format)


def safe_relative(name):
    string(name, "source path")
    p = PurePosixPath(name)
    require(bool(p.parts) and not p.is_absolute() and str(p) == name and ".." not in p.parts
            and "\\" not in name and ":" not in name,
            "Source path must be safe, normalized and relative")
    return p


def validate_manifest(manifest):
    synthetic(manifest, "c2_manifest_v1")
    string(manifest.get("version"), "manifest version")
    run_id = manifest.get("source_run_id")
    string(run_id, "source_run_id")
    require(run_id.startswith("synthetic"), "Source run must be explicitly synthetic")
    entries = manifest.get("sources")
    require(isinstance(entries, list) and len(entries) == 2, "Manifest must bind exactly two sources")
    found, paths, ids = {}, set(), set()
    for entry in entries:
        require(isinstance(entry, dict), "Manifest source must be an object")
        require(set(entry) == {"role", "path", "sha256", "source_id", "source_run_id"}, "Unexpected manifest source fields")
        role = entry["role"]
        require(role in ("bare", "first") and role not in found, "Duplicate or invalid manifest source role")
        safe_relative(entry["path"])
        require(entry["path"] not in paths, "Duplicate manifest source path")
        sha(entry["sha256"], "source file digest")
        string(entry["source_id"], "source_id")
        require(entry["source_id"].startswith("synthetic:") and entry["source_id"] not in ids, "Duplicate or non-synthetic source_id")
        require(entry["source_run_id"] == run_id, "Mixed source runs in manifest")
        paths.add(entry["path"])
        ids.add(entry["source_id"])
        found[role] = entry
    return found


def validate_protocol(protocol):
    synthetic(protocol, "c2_protocol_v1")
    string(protocol.get("version"), "protocol version")
    string(protocol.get("system_text"), "system text")
    require(protocol.get("cells") == CELLS, "Protocol must use exactly the four approved cells")
    require(protocol.get("writers") == {"plain": BASE_INSTRUCTION,
            "evidence": BASE_INSTRUCTION + "\n" + EVIDENCE_INSTRUCTION},
            "Protocol must use the standardized probe instructions")
    settings = protocol.get("settings")
    require(isinstance(settings, dict) and set(settings) == {"model", "temperature", "max_output_tokens"}, "Unexpected writer settings")
    string(settings["model"], "model")
    require(settings["model"].startswith("synthetic://"), "Only a non-operational synthetic:// model identifier is allowed")
    bounded_number(settings["temperature"], "temperature", high=2)
    integer(settings["max_output_tokens"], "max_output_tokens")
    require(settings["max_output_tokens"] == 400, "This synthetic protocol fixes the output-token cap at 400")
    seeds = protocol.get("seed_schedule")
    require(isinstance(seeds, dict) and seeds, "Explicit nonempty per-case seed schedule required")
    for case_id, seed in seeds.items():
        string(case_id, "case_id")
        integer(seed, "seed")
    spec = protocol.get("scoring_spec")
    require(isinstance(spec, dict) and set(spec) == {"name", "version", "metric", "method"}, "Explicit synthetic scoring specification required")
    for name in ("name", "version", "metric"):
        string(spec[name], "scorer " + name)
    require(spec["name"].startswith("synthetic") and spec["method"] == "hand_assigned_test_values", "This core accepts only explicit synthetic hand scores")
    gates = protocol.get("real_run_gates")
    require(isinstance(gates, list) and gates and all(isinstance(x, str) and x for x in gates), "Unresolved real-run gates must remain explicit")


def project_observation(record):
    string(record.get("surface"), "surface")
    history = record.get("history")
    require(isinstance(history, list), "history must be an ordered list")
    projected = []
    for turn in history:
        require(isinstance(turn, dict), "history turns must be objects")
        string(turn.get("question"), "question", nonempty=False)
        string(turn.get("reply"), "reply", nonempty=False)
        projected.append({"question": turn["question"], "reply": turn["reply"]})
    return {"surface": record["surface"], "history": projected}


def build_payload(observation, writer, protocol, case_id):
    return dict(protocol["settings"], seed=protocol["seed_schedule"][case_id], messages=[
        {"role": "system", "content": protocol["system_text"]},
        {"role": "user", "content": canonical_bytes(observation).decode("utf-8")},
        {"role": "user", "content": protocol["writers"][writer]},
    ])


def scorer_identity(protocol):
    spec = protocol["scoring_spec"]
    return {"name": spec["name"], "version": spec["version"], "metric": spec["metric"], "spec_sha256": digest(spec)}


def prepare(manifest_path, protocol_path, *, synthetic_only=False):
    require(synthetic_only is True, "prepare requires explicit --synthetic-only")
    manifest_path, protocol_path = Path(manifest_path), Path(protocol_path)
    manifest_raw, protocol_raw = manifest_path.read_bytes(), protocol_path.read_bytes()
    manifest = decode_json(manifest_raw, manifest_path)
    protocol = decode_json(protocol_raw, protocol_path)
    sources = validate_manifest(manifest)
    validate_protocol(protocol)
    builder = {"version": BUILDER_VERSION, "code_sha256": byte_digest(Path(__file__).read_bytes())}
    manifest_context = {"sha256": digest(manifest), "byte_sha256": byte_digest(manifest_raw), "document": manifest}
    protocol_context = {"sha256": digest(protocol), "byte_sha256": byte_digest(protocol_raw), "document": protocol}
    by_source = {}
    root = manifest_path.parent.resolve()
    for role, source in sources.items():
        path = root.joinpath(*safe_relative(source["path"]).parts)
        require(path.resolve().is_relative_to(root), "Source symlink resolves outside manifest directory")
        raw = path.read_bytes()
        require(byte_digest(raw) == source["sha256"], "Source byte digest mismatch: " + source["path"])
        cases, record_ids = {}, set()
        for number, line in enumerate(raw.splitlines(keepends=True), 1):
            require(line.strip(), "Blank source JSONL row")
            row = decode_json(line, source["path"] + ":" + str(number))
            require(isinstance(row, dict) and row.get("data_kind") == "synthetic", "Source rows must be synthetic objects")
            for field in ("case_id", "record_id", "source_id", "source_run_id"):
                string(row.get(field), field)
            require(row["source_id"] == source["source_id"] and row["source_run_id"] == source["source_run_id"], "Mixed source runs or source identities in records")
            require(row["case_id"] not in cases, "Duplicate case identity in source")
            require(row["record_id"] not in record_ids, "Duplicate record identity in source")
            observation = project_observation(row)
            surface_hash = byte_digest(observation["surface"].encode("utf-8"))
            identity = {
                "role": role, "source_id": source["source_id"], "source_run_id": source["source_run_id"],
                "file_path": source["path"], "file_sha256": source["sha256"],
                "member": {"kind": "standalone_jsonl", "path": source["path"], "sha256": source["sha256"]},
                "record_id": row["record_id"], "record_sha256": digest(row), "line_sha256": byte_digest(line),
                "case_id": row["case_id"], "surface_sha256": surface_hash, "dialogue_sha256": digest(observation["history"]),
            }
            cases[row["case_id"]] = (observation, identity)
            record_ids.add(row["record_id"])
        require(cases, "Source must contain at least one case")
        by_source[role] = cases
    require(set(by_source["bare"]) == set(by_source["first"]), "Acquisition sources have different case sets")
    require(set(by_source["bare"]) == set(protocol["seed_schedule"]), "Seed schedule must exactly cover the planned case set")
    requests = []
    for case_id in sorted(by_source["bare"]):
        require(by_source["bare"][case_id][0]["surface"] == by_source["first"][case_id][0]["surface"], "Source surface mismatch for " + case_id)
        for cell, design in CELLS.items():
            observation, identity = by_source[design["source"]][case_id]
            payload = build_payload(observation, design["writer"], protocol, case_id)
            request = {"format": "c2_request_v1", "data_kind": "synthetic", "case_id": case_id,
                "cell": cell, "writer": design["writer"], "surface_sha256": identity["surface_sha256"],
                "builder": builder, "protocol": protocol_context, "manifest": manifest_context,
                "source_identity": identity, "observation": observation, "payload": payload, "payload_sha256": digest(payload)}
            request["request_id"] = request_identity(request)
            requests.append(request)
    validate_requests(requests)
    return requests


def validate_requests(requests):
    require(isinstance(requests, list) and requests, "Nonempty request list required")
    expected_keys = {"format", "data_kind", "case_id", "cell", "writer", "surface_sha256", "builder",
                     "protocol", "manifest", "source_identity", "observation", "payload", "payload_sha256", "request_id"}
    by_case, ids = {}, set()
    common = None
    for request in requests:
        synthetic(request, "c2_request_v1")
        require(set(request) == expected_keys, "Unexpected request fields")
        case_id, cell = request["case_id"], request["cell"]
        string(case_id, "case_id")
        require(isinstance(cell, str) and cell in CELLS, "Unknown request cell")
        sha(request["request_id"], "request_id")
        require(request["request_id"] == request_identity(request), "Request identity mismatch")
        require(request["request_id"] not in ids, "Duplicate request identity")
        ids.add(request["request_id"])
        bucket = by_case.setdefault(case_id, {})
        require(cell not in bucket, "Duplicate request case/cell")
        builder = request["builder"]
        require(isinstance(builder, dict) and set(builder) == {"version", "code_sha256"}, "Invalid builder identity")
        string(builder["version"], "builder version")
        sha(builder["code_sha256"], "builder code digest")
        for name in ("protocol", "manifest"):
            context = request[name]
            require(isinstance(context, dict) and set(context) == {"sha256", "byte_sha256", "document"}, "Invalid " + name + " identity")
            sha(context["byte_sha256"], name + " byte digest")
            require(context["sha256"] == digest(context["document"]), name + " canonical identity mismatch")
        protocol = request["protocol"]["document"]
        manifest = request["manifest"]["document"]
        validate_protocol(protocol)
        sources = validate_manifest(manifest)
        signature = canonical_bytes({k: request[k] for k in ("builder", "protocol", "manifest")})
        if common is None:
            common = signature
        require(signature == common, "Mixed builder, protocol or manifest/source runs")
        design = CELLS[cell]
        require(request["writer"] == design["writer"], "Cell/writer mismatch")
        require(case_id in protocol["seed_schedule"], "Unplanned request case")
        observation = request["observation"]
        require(isinstance(observation, dict) and set(observation) == {"surface", "history"}, "Unexpected observation fields")
        projected = project_observation(observation)
        require(projected == observation, "Unexpected history fields in request projection")
        identity = request["source_identity"]
        require(isinstance(identity, dict) and set(identity) == {"role", "source_id", "source_run_id", "file_path", "file_sha256", "member", "record_id", "record_sha256", "line_sha256", "case_id", "surface_sha256", "dialogue_sha256"}, "Invalid source/record identity fields")
        source = sources[design["source"]]
        require(identity["role"] == design["source"] and identity["source_id"] == source["source_id"]
                and identity["source_run_id"] == source["source_run_id"]
                and identity["file_path"] == source["path"] and identity["file_sha256"] == source["sha256"], "Source identity does not match manifest/cell")
        require(identity["member"] == {"kind": "standalone_jsonl", "path": source["path"], "sha256": source["sha256"]}, "Source member identity mismatch")
        string(identity["record_id"], "record_id")
        for field in ("record_sha256", "line_sha256", "file_sha256"):
            sha(identity[field], field)
        surface_hash = byte_digest(observation["surface"].encode("utf-8"))
        require(identity["case_id"] == case_id and identity["surface_sha256"] == surface_hash
                and request["surface_sha256"] == surface_hash, "Case/surface identity mismatch")
        require(identity["dialogue_sha256"] == digest(observation["history"]), "Dialogue identity mismatch")
        require(canonical_bytes(request["payload"]) == canonical_bytes(build_payload(observation, design["writer"], protocol, case_id)), "Payload has inconsistent settings, messages or observation")
        require(request["payload_sha256"] == digest(request["payload"]), "Payload digest mismatch")
        bucket[cell] = request
    require(set(by_case) == set(protocol["seed_schedule"]), "Request case set differs from planned seed schedule")
    source_record_ids = {"bare": set(), "first": set()}
    for case_id, cells in by_case.items():
        require(set(cells) == set(CELLS), "Missing request cells for " + case_id)
        require(len({cells[c]["surface_sha256"] for c in CELLS}) == 1, "Inconsistent surfaces across request cells")
        for a, b in (("A", "B"), ("C", "D")):
            require(cells[a]["source_identity"] == cells[b]["source_identity"] and cells[a]["observation"] == cells[b]["observation"], "Paired writers must use the identical source record/history")
            role, record_id = cells[a]["source_identity"]["role"], cells[a]["source_identity"]["record_id"]
            require(record_id not in source_record_ids[role], "Duplicate source record across cases")
            source_record_ids[role].add(record_id)
    return by_case


def validate_response(response, requests_by_id, expected_scorer):
    synthetic(response, "c2_response_v1")
    require(set(response) == {"format", "data_kind", "case_id", "cell", "request_id", "payload_sha256", "scorer", "status", "text", "finish_reason", "truncated", "usage", "score", "error"}, "Unexpected response fields")
    request_id = response["request_id"]
    require(isinstance(request_id, str) and request_id in requests_by_id, "Foreign response request identity")
    request = requests_by_id[request_id]
    require(response["case_id"] == request["case_id"] and response["cell"] == request["cell"]
            and response["payload_sha256"] == request["payload_sha256"], "Response/request identity mismatch")
    require(response["scorer"] == expected_scorer, "Response/scorer identity mismatch")
    status = response["status"]
    require(status in ("completed", "failed", "unscored"), "Invalid response status")
    string(response["text"], "full response text", nonempty=False)
    require(response["finish_reason"] is None or isinstance(response["finish_reason"], str), "finish_reason must be string or null")
    require(response["truncated"] is None or type(response["truncated"]) is bool, "truncated must be bool or null")
    if status == "completed":
        bounded_number(response["score"], "score")
        require(response["error"] is None, "Completed response must not carry an error")
    else:
        require(response["score"] is None, "Failed/unscored response must retain a null score")
        string(response["error"], "failure/unscored diagnostic")
    usage = response["usage"]
    require(isinstance(usage, dict) and set(usage) == set(USAGE_FIELDS), "Explicit usage fields required")
    for field, value in usage.items():
        if value is not None:
            integer(value, field)
    if all(usage[field] is not None for field in USAGE_FIELDS):
        require(usage["input_tokens"] + usage["output_tokens"] == usage["total_tokens"], "Inconsistent usage total")


def analyze(requests, responses):
    by_case = validate_requests(requests)
    require(isinstance(responses, list), "Responses must be a list")
    requests_by_id = {request["request_id"]: request for request in requests}
    protocol = requests[0]["protocol"]["document"]
    scorer = scorer_identity(protocol)
    by_id = {}
    for response in responses:
        require(isinstance(response, dict), "Response must be an object")
        validate_response(response, requests_by_id, scorer)
        require(response["request_id"] not in by_id, "Duplicate response request/cell")
        by_id[response["request_id"]] = response
    cases, diagnostics = [], []
    for case_id in sorted(by_case):
        cells = {}
        for cell in CELLS:
            request = by_case[case_id][cell]
            response = by_id.get(request["request_id"])
            status = response["status"] if response is not None else "missing"
            score = response["score"] if response is not None else None
            cells[cell] = {"request_id": request["request_id"], "status": status, "score": score, "response": response}
            if status != "completed":
                diagnostic = {"case_id": case_id, "cell": cell, "reason": status}
                if response is not None:
                    diagnostic["error"] = response["error"]
                diagnostics.append(diagnostic)
        contrasts = {}
        for name, required in CONTRASTS.items():
            if any(cells[c]["score"] is None for c in required):
                contrasts[name] = None
            elif name == "interaction":
                contrasts[name] = (cells["D"]["score"] - cells["C"]["score"]) - (cells["B"]["score"] - cells["A"]["score"])
            else:
                contrasts[name] = cells[required[0]]["score"] - cells[required[1]]["score"]
        cases.append({"case_id": case_id, "cells": cells, "contrasts": contrasts})
    planned = len(cases)
    means = {}
    for name in CONTRASTS:
        values = [case["contrasts"][name] for case in cases if case["contrasts"][name] is not None]
        available = len(values)
        means[name] = {"mean": math.fsum(values) / planned if available == planned else None,
                       "expected": planned, "available": available, "missing": planned - available}
    usage = {"planned": len(requests), "returned": len(by_id), "missing_responses": len(requests) - len(by_id), "fields": {}}
    for field in USAGE_FIELDS:
        values = [response["usage"][field] for response in by_id.values() if response["usage"][field] is not None]
        usage["fields"][field] = {"known": len(values), "unknown": len(requests) - len(values),
                                   "total_if_complete": sum(values) if len(values) == len(requests) else None}
    return {"format": "c2_analysis_v1", "data_kind": "synthetic", "planned_cases": planned,
        "planned_requests": len(requests), "case_results": cases, "means": means, "diagnostics": diagnostics,
        "usage": usage, "scorer": scorer,
        "provenance": {"builder": requests[0]["builder"],
                       "protocol": {k: v for k, v in requests[0]["protocol"].items() if k != "document"},
                       "manifest": {k: v for k, v in requests[0]["manifest"].items() if k != "document"}},
        "limitations": ["Hand-assigned synthetic test scores; no model findings.",
                        "History utility is conditional on writers, not an equal-compute acquisition effect.",
                        "Matching seeds does not synchronize backend draws.",
                        "Hashes establish supplied-content consistency, not historical authenticity.",
                        "Independent code review pending; real inference and historical preparation disabled."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare", help="Compile only manifest-bound synthetic dialogues")
    p.add_argument("--manifest", required=True)
    p.add_argument("--protocol", required=True)
    p.add_argument("--synthetic-only", action="store_true", required=True)
    p.add_argument("--out", required=True)
    a = commands.add_parser("analyze", help="Analyze explicit synthetic scored responses")
    a.add_argument("--requests", required=True)
    a.add_argument("--responses", required=True)
    a.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            requests = prepare(args.manifest, args.protocol, synthetic_only=args.synthetic_only)
            write_jsonl(args.out, requests)
            print(json.dumps({"data_kind": "synthetic", "cases": len(requests) // 4, "requests": len(requests)}, sort_keys=True))
        else:
            report = analyze(read_jsonl(args.requests), read_jsonl(args.responses))
            path = Path(args.out)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            print(json.dumps({"data_kind": "synthetic", "planned_cases": report["planned_cases"], "diagnostics": len(report["diagnostics"]), "means": report["means"]}, sort_keys=True, allow_nan=False))
        return 0
    except (ValidationError, OSError) as exc:
        print("Validation error: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
