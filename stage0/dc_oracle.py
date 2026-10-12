"""DC (Detective Cases) Stage 0 -- terminal-only, not a per-turn exact oracle.

GN and CD both admit an exact oracle: at every turn there's a mechanically
computable feasible set (remaining numbers / remaining truth tables), so
ex-ante efficiency and a true belief-trap signal are both well-defined.

DC does not. Each suspect's motive/opportunity/access_to_weapon fields are
free-text evidence sentences, not booleans: checked across all 100 test
cases, every suspect (murderer and innocents alike) has non-empty content in
every category, and ~19% of these sentences (388/2024) contain negation
("did not", "unable to", ...) -- i.e. the field can assert presence OR
absence of the attribute. Telling them apart requires reading what the
sentence claims, which this module deliberately does not attempt (that's an
LLM judgment call, not a ground-truth computation, and using one to build
the "ground truth" would be circular).

What IS mechanical and verified 100/100 on data/dc/test.json:
  - each case has exactly 5 suspects; 4 carry an `is_murderer` key (real
    suspects), 1 does not (a filler, never guilty, minimal fields);
  - of the 4 real suspects, exactly 1 has is_murderer == True;
  - each real suspect carries its own key_question list (3 entries for the
    murderer: motive/opportunity/access; fewer for innocents) -- usable as a
    coverage-style process proxy (did the transcript ever address these),
    the same tier as what SP needs, not an exact per-turn belief state.

So this module gives you the terminal label space (4-way, chance = 25%) and
the key_question lists for a coverage proxy. It does not give you a
per-turn feasible-set size or a zero_info flag the way cd_oracle.py does.
"""

import json


def load_instances(path):
    return json.load(open(path))


def real_suspects(case):
    """The 4 suspects eligible to be the murderer (excludes the filler,
    identified by lacking an is_murderer key -- verified 100/100 cases)."""
    return [s for s in case["suspects"] if "is_murderer" in s]


def truth_index(case):
    """Index into real_suspects(case) of the true murderer."""
    reals = real_suspects(case)
    for i, s in enumerate(reals):
        if s["is_murderer"]:
            return i
    raise ValueError("no is_murderer==True among real suspects")


def key_questions(case):
    """{suspect_name: [key_question, ...]} for the 4 real suspects, for a
    coverage proxy: did the transcript's Q&A ever address each question's
    content. This is the same granularity as whatever proxy is used for SP,
    not a mechanical belief-state computation."""
    return {s["name"]: s.get("key_question", []) for s in real_suspects(case)}


def selftest(path):
    cases = load_instances(path)
    assert len(cases) > 0
    for c in cases:
        reals = real_suspects(c)
        assert len(reals) == 4, f"expected 4 real suspects, got {len(reals)}"
        assert sum(s["is_murderer"] for s in reals) == 1
        truth_index(c)  # must not raise
    print(f"dc_oracle selftest: {len(cases)}/{len(cases)} cases OK "
          f"(4 real suspects, 1 true murderer, terminal label space only)")


if __name__ == "__main__":
    import sys
    selftest(sys.argv[1] if len(sys.argv) > 1 else "../AR-Bench/data/dc/test.json")
