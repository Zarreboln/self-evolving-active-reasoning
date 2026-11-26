"""The meta-skill bank and one keep / add / revise decision.

apply_decision() follows the released builder.reflect_skill line by line (operation keep/add/revise,
SupportSkill.from_dict with the evidence allowlist, 768-byte fields, new IDs must not exist, a revision
must name an existing skill and keep its evidence IDs). Changes, from the paper:
  * evidence is a batch of episodes (one development task's search problems), so the allowlist is the
    batch's IDs plus earlier evidence, and a new or revised skill must cite at least one ID of this batch
    (paper §3.1: "must cite supporting evidence from that batch"; released: "the current episode");
  * when + provide + use <= 192 tokens together (paper B.1), counted with the released approximate counter;
  * superseded versions are archived (paper B.1);
  * the shared-store rules of experiment B (no names, no answers; rules.py).
"""
from __future__ import annotations

from mb.baselines.metaskill import official, rules
from mb.baselines.metaskill.config import SKILL_FIELD_BYTES, SKILL_TOKENS


class ReflectionError(ValueError):
    pass


def apply_decision(skills: list[dict], decision, batch_ids: list[str], answers: dict | None):
    """Returns (new skills, superseded skill or None, operation). Raises ReflectionError."""
    SupportSkill = official.support_skill_cls()
    if not isinstance(decision, dict) or "operation" not in decision:
        raise ReflectionError("decision.json must be an object with an operation")
    op = decision["operation"]
    if op == "keep":
        return list(skills), None, "keep"
    if op not in ("add", "revise"):
        raise ReflectionError("unknown skill operation")
    known = set(batch_ids) | {e for s in skills for e in s["evidence_ids"]}
    try:
        skill = SupportSkill.from_dict(decision["skill"], known_episode_ids=known).to_dict()
    except (KeyError, TypeError, ValueError) as e:
        raise ReflectionError(f"invalid skill: {e}") from e
    if not set(batch_ids) & set(skill["evidence_ids"]):
        raise ReflectionError("an added or revised skill must cite at least one episode ID of this batch")
    for f in ("when", "provide", "use"):
        if len(skill[f].encode("utf-8")) > SKILL_FIELD_BYTES:
            raise ReflectionError(f"skill field {f} exceeds {SKILL_FIELD_BYTES} bytes")
    n = sum(official.approx_tokens(skill[f]) for f in ("when", "provide", "use"))
    if n > SKILL_TOKENS:
        raise ReflectionError(f"when + provide + use has {n} tokens; the limit is {SKILL_TOKENS}")
    result, old = list(skills), None
    if op == "add":
        if any(s["skill_id"] == skill["skill_id"] for s in skills):
            raise ReflectionError("new skill ID already exists")
        result.append(skill)
    else:
        prior = decision.get("prior_skill_id")
        idx = next((i for i, s in enumerate(skills) if s["skill_id"] == prior), None)
        if idx is None:
            raise ReflectionError("revised skill ID does not exist")
        if not set(skills[idx]["evidence_ids"]) <= set(skill["evidence_ids"]):
            raise ReflectionError("revision must preserve prior evidence IDs")
        if any(s["skill_id"] == skill["skill_id"] for i, s in enumerate(skills) if i != idx):
            raise ReflectionError("revised skill ID collides")
        old = skills[idx]
        result[idx] = skill
    bad = rules.bank_problems([skill], answers)
    if bad:
        raise ReflectionError("shared-store rule: " + "; ".join(bad))
    return result, old, op
