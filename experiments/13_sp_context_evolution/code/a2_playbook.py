"""Control A2: a playbook written by the same reflector model from the task description alone (no TRAIN experience,
no labels), held to the same rule schema and length caps. If an evolved playbook does not beat this, its gain is the
reflector's prior knowledge, not learning from experience."""
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evolve  # noqa: E402
from llm import chat  # noqa: E402

PROMPT = (evolve.INTRO + "(empty)\n\nYou have NO example games. From your general knowledge of situation puzzles "
          "(lateral-thinking riddles) and of good yes/no questioning, write the best playbook you can.\n\n"
          + evolve.SCHEMA.replace("At most 3 ops.", f"Use only ADD ops, at most {evolve.MAX_RULES} of them, "
                                  f"and keep the whole playbook under {evolve.MAX_WORDS} words."))


def main(seed=0):
    out = evolve.OUT / f"a2_s{seed}"
    out.mkdir(parents=True, exist_ok=True)
    txt = chat(evolve.REFLECTOR, [{"role": "user", "content": PROMPT}], role="reflector", temperature=0.7,
               max_tokens=12000, seed=seed, reasoning=True, tag=f"a2-s{seed}")
    m = re.search(r"\{.*\}", txt or "", re.S)
    ops = json.loads(m.group(0)).get("ops", []) if m else []
    rules = []
    for op in ops:
        cand, _ = evolve.apply_ops(rules, [op], round_i=0)
        if cand is not None:
            rules = cand
    final = {"mech": "a2", "seed": seed, "rules": rules, "playbook": evolve.render(rules),
             "sha": hashlib.sha256(evolve.render(rules).encode()).hexdigest()[:12]}
    (out / "final_playbook.json").write_text(json.dumps(final, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "reflector_raw.txt").write_text(txt or "", encoding="utf-8")
    print(f"a2 s{seed}: {len(rules)} rules, {evolve.words(rules)} words")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
