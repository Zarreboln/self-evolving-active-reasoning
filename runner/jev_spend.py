"""Sum what Jev has actually been billed for, from the shim's own verdict log.

jev_referee_shim.py records the usage block of every call it makes, so the spend is a sum over the
log rather than an estimate from episode counts. Jev bills input tokens only; output is free.
"""
import json
import sys

RATE_IN = 0.042 / 1_000_000          # USD per input token, from TypeSafe's pricing

for path in sys.argv[1:]:
    n = tin = tout = 0
    bad = 0
    labels = {}
    try:
        f = open(path)
    except OSError:
        print(f"{path}: not present")
        continue
    for line in f:
        try:
            r = json.loads(line)
        except Exception:
            bad += 1
            continue
        u = r.get("usage") or {}
        n += 1
        tin += u.get("input_tokens", 0) or 0
        tout += u.get("output_tokens", 0) or 0
        labels[r.get("label")] = labels.get(r.get("label"), 0) + 1
    print(f"{path}")
    print(f"  calls {n}   input tokens {tin:,}   output tokens {tout:,} (not billed)"
          + (f"   unparsable lines {bad}" if bad else ""))
    print(f"  labels {labels}")
    print(f"  spend  ${tin * RATE_IN:.4f}")
