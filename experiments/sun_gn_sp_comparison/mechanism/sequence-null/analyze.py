"""Portable, original arithmetic for the published sequence reference.

Standard library only. No AR-Bench code, private cases, models or network.
Default: recompute summaries from the published counts. --recompute-all also
recomputes all 1,000 sequence partitions; this optional mode was NOT run when
assembling this appendix. It never overwrites the preserved input/results.
"""
from collections import Counter
from functools import lru_cache
from itertools import permutations
import argparse
import json
import math
from pathlib import Path
import statistics

POPULATION = 5040
SEED = 20261026
SCHEDULES = {
    "E0": ["1234", "5678", "9012", "3490", "6789", "4390", "2879"],
    "E1": ["1234", "5678", "9012", "3456", "7890", "1357", "2468", "0379", "1478"],
}
ANCHORS = (("E0", 7, 2729), ("E1", 7, 3199), ("E1", 9, 4583))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result

    def finite(value):
        result = float(value)
        require(math.isfinite(result), "Nonfinite JSON number")
        return result

    def reject(value):
        raise ValueError("Nonfinite JSON constant: " + value)

    return json.loads(Path(path).read_bytes(), object_pairs_hook=pairs,
                      parse_float=finite, parse_constant=reject)


@lru_cache(maxsize=1)
def universe():
    """Lexicographic public universe, including leading zero."""
    return tuple("".join(x) for x in permutations("0123456789", 4))


def legal_query(query):
    return (isinstance(query, str) and len(query) == 4 and
            all(c in "0123456789" for c in query) and len(set(query)) == 4)


@lru_cache(maxsize=POPULATION)
def feedback_vector(query):
    """Original mathematical implementation for four distinct digits.

    Bulls count positional matches. Cows count shared digits minus bulls.
    Distinct digits make set intersection equal to multiset intersection.
    The injective code 5*bulls+cows preserves BOTH observations; it is not
    the benchmark's scalar score. No upstream implementation is imported.
    """
    require(legal_query(query), "Illegal query")
    qset = set(query)
    values = []
    for secret in universe():
        bulls = sum(a == b for a, b in zip(secret, query))
        cows = len(set(secret) & qset) - bulls
        values.append(5 * bulls + cows)
    return bytes(values)


def identified_counts(queries, prefixes):
    """Count singleton feedback-signature classes, not actor answers."""
    require(isinstance(queries, list) and all(legal_query(q) for q in queries), "Query schema")
    require(len(set(queries)) == len(queries), "Repeated query within a sequence")
    require(prefixes and all(type(k) is int and 1 <= k <= len(queries) for k in prefixes), "Prefix schema")
    signatures = [0] * POPULATION
    result = {}
    for length, query in enumerate(queries, 1):
        signatures = [25 * before + code for before, code in zip(signatures, feedback_vector(query))]
        if length in prefixes:
            sizes = Counter(signatures)
            result[length] = sum(size == 1 for size in sizes.values())
    return result


def validate_rows(document):
    require(isinstance(document, dict) and set(document) == {"seed", "rows"}, "Data fields")
    require(type(document["seed"]) is int and document["seed"] == SEED, "Data seed")
    rows = document["rows"]
    require(isinstance(rows, list) and len(rows) == 1000, "Exactly 1,000 rows required")
    for index, row in enumerate(rows, 1):
        require(isinstance(row, dict) and set(row) == {
            "sequence_index", "queries", "identified_at_7", "identified_at_9"}, "Row fields")
        require(type(row["sequence_index"]) is int and row["sequence_index"] == index, "Sequence order")
        require(isinstance(row["queries"], list) and len(row["queries"]) == 9 and
                all(legal_query(q) for q in row["queries"]) and len(set(row["queries"])) == 9,
                "Nine distinct legal queries required")
        a, b = row["identified_at_7"], row["identified_at_9"]
        require(type(a) is int and type(b) is int and 0 <= a <= b <= POPULATION, "Count bounds/monotonicity")
    return rows


def distribution(counts):
    ordered = sorted(counts)
    percentiles = {}
    for percent in (0, 1, 5, 25, 50, 75, 95, 99, 100):
        index = max(0, math.ceil(percent * len(ordered) / 100) - 1)
        value = ordered[index]
        percentiles[str(percent)] = {"identified": value, "fraction": value / POPULATION}
    return {"sequences": len(counts), "mean_identified": statistics.mean(counts),
            "mean_fraction": statistics.mean(counts) / POPULATION,
            "population_sd_identified": statistics.pstdev(counts),
            "population_sd_fraction": statistics.pstdev(counts) / POPULATION,
            "nearest_rank_percentiles": percentiles}


def summarize(rows):
    counts = {k: [r["identified_at_" + str(k)] for r in rows] for k in (7, 9)}
    comparisons = []
    for name, prefix, anchor in ANCHORS:
        values = counts[prefix]
        below = sum(x < anchor for x in values)
        equal = sum(x == anchor for x in values)
        above = sum(x > anchor for x in values)
        comparisons.append({"schedule": name, "prefix": prefix, "anchor_identified": anchor,
            "fraction_identified": anchor / POPULATION,
            "random_strictly_below": below, "random_equal": equal, "random_strictly_above": above,
            "empirical_fraction_less_or_equal": (below + equal) / 1000,
            "empirical_midrank_percentile": (below + .5 * equal) / 10,
            "random_fraction_at_least_as_good": (equal + above) / 1000,
            "meaning": "Descriptive random-sequence rank, not a learning-effect p-value"})
    return {"distributions": {str(k): distribution(values) for k, values in counts.items()},
            "comparisons": comparisons}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recompute-all", action="store_true", help="Opt-in full mathematical count verification; no model calls")
    args = parser.parse_args()
    base = Path(__file__).resolve().parent
    rows = validate_rows(load(base / "random-sequences.json"))
    if args.recompute_all:
        for row in rows:
            actual = identified_counts(row["queries"], (7, 9))
            require(all(actual[k] == row["identified_at_" + str(k)] for k in (7, 9)),
                    "Partition mismatch at sequence " + str(row["sequence_index"]))
    summary = summarize(rows)
    original = load(base / "result.json")
    require(summary == {key: original[key] for key in summary}, "Preserved summary differs")
    print(json.dumps({"status": "PRESERVED_SUMMARY_MATCHES", "full_partitions_recomputed": args.recompute_all,
                      **summary}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
