"""Everything taken verbatim from the released code, loaded at run time from the pinned clone.

The repository has no license file, so none of its text or code is copied into ours (the same rule as
AR-Bench in THIRD_PARTY.md): prompts, the runner boundary and the skill schema are read from
third_party_b/metaskill at run time. What we change is applied here, by position, and documented.

    builder_system()        builder.BUILDER_SYSTEM, verbatim
    build_instruction()     the user instruction of builder.generate_bundle for a first build, verbatim
    repair_message(err)     the interface-repair message of builder.generate_bundle, verbatim
    reflection_system()     the system prompt of builder.reflect_skill, verbatim (read from its source)
    runner_boundary()       harness.compiler.FIXED_RUNNER_BOUNDARY with its last line (the finish tool,
                            which our environments do not have) replaced by our reply protocol
    neutral_instruction()   schema.neutral_bundle().instruction (the instruction of H0)
    SupportSkill            schema.SupportSkill (validation of when/provide/use skills)
    approx_tokens(text)     token_count.approximate_token_count
"""
from __future__ import annotations

import ast
import functools
import subprocess
import sys

from mb.baselines.metaskill.config import CLONE, CLONE_COMMIT

SRC = CLONE / "src"


def check_clone():
    if not (SRC / "meta_skill_ai4ai" / "builder.py").exists():
        raise RuntimeError(f"official clone missing: git clone https://github.com/qiancheng-apodex/"
                           f"MetaSkill-AI4AI {CLONE} && git -C {CLONE} checkout {CLONE_COMMIT}")
    head = subprocess.run(["git", "-C", str(CLONE), "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    if head and head != CLONE_COMMIT:
        raise RuntimeError(f"{CLONE} is at {head}, expected the pinned {CLONE_COMMIT}")


@functools.lru_cache(maxsize=None)
def _pkg():
    check_clone()
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    import meta_skill_ai4ai.builder as builder               # noqa: E402  (no API call at import)
    import meta_skill_ai4ai.harness.compiler as compiler
    import meta_skill_ai4ai.modular.schema as schema
    import meta_skill_ai4ai.token_count as token_count
    return builder, compiler, schema, token_count


def _strings_in(func_name: str) -> list[str]:
    """String constants of one function of builder.py, in source order (implicit concatenation joined)."""
    tree = ast.parse((SRC / "meta_skill_ai4ai" / "builder.py").read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == func_name)
    out = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            out.append(node.value)
    return out


def builder_system() -> str:
    return _pkg()[0].BUILDER_SYSTEM


def build_instruction() -> str:
    """generate_bundle's instruction when there is no parent bundle ("Build a reusable harness ...")."""
    return next(s for s in _strings_in("generate_bundle") if s.startswith("Build a reusable harness"))


def repair_message(error: str) -> str:
    """generate_bundle's repair turn; the f-string's literal parts are joined around the error."""
    tree = ast.parse((SRC / "meta_skill_ai4ai" / "builder.py").read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "generate_bundle")
    js = next(n for n in ast.walk(fn) if isinstance(n, ast.JoinedStr)
              and isinstance(n.values[0], ast.Constant) and n.values[0].value.startswith("Correct only"))
    parts = []
    for v in js.values:
        parts.append(v.value if isinstance(v, ast.Constant) else error)
    return "".join(parts)


def reflection_system() -> str:
    """reflect_skill's system prompt: the parenthesised string assigned to `system`."""
    tree = ast.parse((SRC / "meta_skill_ai4ai" / "builder.py").read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "reflect_skill")
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "system"
                                                for t in node.targets):
            return node.value.value
    raise RuntimeError("reflect_skill system prompt not found")


PROTOCOL = ("- End every reply with exactly one line: `Action: <action>` to act in the environment, or "
            "`Tool: <name> <JSON object of arguments>` to call a harness tool listed below. The environment, "
            "not the model, ends the episode.")


def runner_boundary() -> str:
    lines = _pkg()[1].FIXED_RUNNER_BOUNDARY.splitlines()
    assert "finish" in lines[-1], "the released boundary changed; review the replacement"
    return "\n".join(lines[:-1] + [PROTOCOL])


def neutral_instruction() -> str:
    return _pkg()[2].neutral_bundle().instruction


def support_skill_cls():
    return _pkg()[2].SupportSkill


def approx_tokens(text: str) -> int:
    return _pkg()[3].approximate_token_count(text)
