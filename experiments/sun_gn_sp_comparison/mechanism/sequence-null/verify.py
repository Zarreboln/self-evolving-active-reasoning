"""Small standard-library publication verifier; no full diagnostic rerun.

Verifies every published row, deterministic draw, and summary statistic.
Recomputes the three fixed anchors and random rows 1 and 1000 only using the
original mathematical code in analyze.py. This is not a new native-code check,
a full 1,000-row partition audit, a model run, or a scientific success test.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

from analyze import ANCHORS, SCHEDULES, SEED, identified_counts, load, require, summarize, universe, validate_rows

FILES = {"analyze.py", "verify.py", "random-sequences.json", "result.json", "distribution.svg", "README.md"}
RESULT_SHA = "4477a2abe191d374d756411b61bd2c39b1a0f6a07b17beed1c62645e709cafe4"
ROWS_SHA = "5b9d20b4343fb31e31fa23e44168000780bb713cf56ae0cb7f2cce6006dad77e"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def verify(base):
    manifest_raw = (base / "MANIFEST.json").read_bytes()
    manifest = load(base / "MANIFEST.json")
    require(set(manifest["files"]) == FILES, "Unexpected manifest file set")
    for name, expected in manifest["files"].items():
        raw = (base / name).read_bytes()
        require(expected == {"bytes": len(raw), "sha256": sha(raw)}, "File changed: " + name)
    require(sha((base / "result.json").read_bytes()) == RESULT_SHA, "Original result changed")
    require(sha((base / "random-sequences.json").read_bytes()) == ROWS_SHA, "Original random rows changed")
    rows = validate_rows(load(base / "random-sequences.json"))
    original = load(base / "result.json")
    require(original["status"] == "COMPLETE_POSTHOC_ALGORITHMIC_DIAGNOSTIC" and
            original["random_rows_sha256"] == ROWS_SHA, "Original provenance mismatch")
    rng = random.Random(SEED)
    words = universe()
    for row in rows:
        require(row["queries"] == [words[i] for i in rng.sample(range(5040), 9)], "Seeded draw differs")
    summary = summarize(rows)
    require(summary == {key: original[key] for key in summary}, "Numerical summary differs")
    anchors = {}
    for name, queries in SCHEDULES.items():
        prefixes = tuple(k for arm, k, _ in ANCHORS if arm == name)
        actual = identified_counts(queries, prefixes)
        for arm, k, expected in ANCHORS:
            if arm == name:
                require(actual[k] == expected, "Fixed anchor differs")
                anchors[f"{name}_at_{k}"] = actual[k]
    samples = []
    for index in (0, 999):
        row = rows[index]
        actual = identified_counts(row["queries"], (7, 9))
        require(all(actual[k] == row["identified_at_" + str(k)] for k in (7, 9)), "Sample partition differs")
        samples.append({"sequence_index": index + 1, "identified_at_7": actual[7], "identified_at_9": actual[9]})
    return {"status": "PUBLICATION_CONSISTENCY_CHECKS_PASSED", "manifest_sha256": sha(manifest_raw),
            "original_result_sha256": RESULT_SHA, "original_rows_sha256": ROWS_SHA,
            "file_hashes_checked": len(FILES), "rows_schema_and_draw_checked": 1000,
            "all_preserved_summary_statistics_recomputed": True,
            "fixed_anchor_counts_recomputed": anchors, "random_partition_rows_recomputed": samples,
            "full_random_partition_recomputation": False,
            "upstream_native_validation_reexecuted": False, "models": 0, "network": 0, "GPU": 0,
            "learning_effect_p_value_computed": False, "new_scientific_result_claimed": False,
            "python": sys.version.split()[0]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-report", action="store_true", help="Create verification.json exclusively; existing report stops")
    args = parser.parse_args()
    base = Path(__file__).resolve().parent
    report_path = base / "verification.json"
    if args.write_report:
        require(not report_path.exists(), "Existing verification report; do not overwrite")
    started = time.perf_counter()
    result = verify(base)
    result["wall_seconds"] = time.perf_counter() - started
    raw = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.write_report:
        with report_path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(raw)
    print(raw, end="")


if __name__ == "__main__":
    main()
