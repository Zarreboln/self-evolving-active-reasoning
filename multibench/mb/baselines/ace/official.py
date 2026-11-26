"""ACE's own code, loaded from the pinned checkout in third_party_b/ace (not copied).

Only files without third-party dependencies are imported: the prompt modules and playbook_utils.py.
playbook_utils.py imports `get_section_slug` from ACE's utils.py, which also loads openai, tiktoken
and a .env file at import time; we give it a stand-in `utils` module holding that one function, taken
verbatim from utils.py's source.
"""
from __future__ import annotations

import ast
import importlib.util
import sys
import types
from pathlib import Path

ACE_DIR = Path(__file__).resolve().parents[4] / "third_party_b" / "ace"
ACE_COMMIT = "82709de050e1db6e6ef2f07bcb0393560b94992a"          # ace-agent/ace, 2026-08-24
ACE_APPWORLD_COMMIT = "9f3e92155345a9159f3a8b25abc334eeca05b545"  # its ace-appworld submodule


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _utils_shim():
    src = (ACE_DIR / "utils.py").read_text()
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "get_section_slug")
    shim = types.ModuleType("ace_utils_shim")
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(ACE_DIR / "utils.py"), "exec"), shim.__dict__)
    return shim


def _playbook_utils():
    saved = sys.modules.get("utils")
    sys.modules["utils"] = _utils_shim()
    try:
        return _load("ace_playbook_utils", ACE_DIR / "playbook_utils.py")
    finally:
        if saved is None:
            sys.modules.pop("utils", None)
        else:
            sys.modules["utils"] = saved


if not (ACE_DIR / "playbook_utils.py").exists():
    raise ImportError(f"ACE checkout missing at {ACE_DIR}: git clone https://github.com/ace-agent/ace "
                      f"and check out {ACE_COMMIT}")

reflector_prompts = _load("ace_prompts_reflector", ACE_DIR / "ace" / "prompts" / "reflector.py")
curator_prompts = _load("ace_prompts_curator", ACE_DIR / "ace" / "prompts" / "curator.py")
generator_prompts = _load("ace_prompts_generator", ACE_DIR / "ace" / "prompts" / "generator.py")
pb = _playbook_utils()

REFLECTOR_PROMPT_NO_GT = reflector_prompts.REFLECTOR_PROMPT_NO_GT
CURATOR_PROMPT_NO_GT = curator_prompts.CURATOR_PROMPT_NO_GT
GENERATOR_PROMPT = generator_prompts.GENERATOR_PROMPT

# ace/ace.py ACE._initialize_empty_playbook (verbatim)
EMPTY_PLAYBOOK = """## STRATEGIES & INSIGHTS

## FORMULAS & CALCULATIONS

## CODE SNIPPETS & TEMPLATES

## COMMON MISTAKES TO AVOID

## PROBLEM-SOLVING HEURISTICS

## CONTEXT CLUES & INDICATORS

## OTHERS"""

# The ACE generator's environment-feedback strings (ace/ace.py _train_single_sample, verbatim).
FEEDBACK_WRONG = "Predicted answer does not match ground truth"
FEEDBACK_RIGHT = "Predicted answer matches ground truth"

# Official hyperparameters. Paper (arXiv 2510.04618, Sec. 4 "Implementation details"): batch size 1,
# at most 5 Reflector refinement rounds, at most 5 offline epochs, one model for all three roles in
# its non-thinking mode. Repo defaults (eval/*/run.py, ace/ace.py): curator_frequency 1, max_tokens
# 4096 per role call, temperature 0.0 (llm.py), json_mode off, bulletpoint analyzer off.
MAX_NUM_ROUNDS = 5
NUM_EPOCHS = 5
CURATOR_FREQUENCY = 1
ROLE_MAX_TOKENS = 4096
