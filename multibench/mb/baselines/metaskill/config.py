"""Fixed settings of the Learning Meta-Skills baseline in experiment B.

Experiment settings come from notes/exp_b_design.md (fixed 2026-11-26) and are not changed here.
Method settings come from the paper (arXiv 2609.38143 v2) and the released code (MetaSkill-AI4AI,
pinned below); each one names its source.
"""
from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
MB = HERE.parents[1]                                  # multibench/mb
REPO = MB.parents[1]                                  # ar_harness
EVOLVE = MB / "evolve"
CLONE = REPO / "third_party_b" / "metaskill"          # github.com/qiancheng-apodex/MetaSkill-AI4AI
CLONE_COMMIT = "72a7b02df9820c629b11c881b4b060ecef7c0ddb"   # 2026-09-29 "update readme"; no license file
SANDBOX = REPO / "sandbox"
RUNS = Path.home() / "evo_b_runs"                     # outside every repo and workspace (as orchestrate.py)

# ---- experiment B (notes/exp_b_design.md) ----
DEV_CELLS = ["wordle:basic5", "wordle:basic7", "hangman:basic", "babyai:GoToLocal", "babyai:Open",
             "sciworld:test-conductivity", "dc:all"]
# Test tasks. 20Q is one task made of three themes pooled; its problems come from the order file.
TEST_TASKS = {
    "mastermind_basic": ["mastermind:basic"],
    "alfworld_pick_and_place_simple": ["alfworld:pick_and_place_simple"],
    "twentyq_basic_pooled": ["twentyq:basic-people", "twentyq:basic-places", "twentyq:basic-things"],
    "minesweeper_basic": ["minesweeper:basic"],
    "wordle_hard5": ["wordle:hard5"],
    "hangman_hard": ["hangman:hard"],
    "sciworld_grow_plant": ["sciworld:grow-plant"],
}
TEST_PROBLEMS = {"alfworld_pick_and_place_simple": 24, "sciworld_grow_plant": 33}   # others: 50
ROUNDS = 8                    # equal budget: 8 rounds x 140 evaluated episodes
SEARCH_N = 20                 # search problems per development task, the same every round
PATIENCE = 3                  # early stop: 3 rounds in a row without a gain beyond noise

# ---- the method (paper / released code) ----
INTERFACE_VERSION = "modular-v2.0"          # released schema.INTERFACE_VERSION
MAX_REPAIRS = 2                             # paper §4.1, B.2: at most two interface-repair attempts
SKILL_TOKENS = 192                          # paper B.1: when+provide+use <= 192 tokens (approximate count)
SKILL_FIELD_BYTES = 768                     # released builder.reflect_skill: each field <= 768 UTF-8 bytes
BUILD_OUTPUT_TOKENS = 16_000                # paper §4.1: 16K output tokens for harness generation
REFLECT_OUTPUT_TOKENS = 8_000               # paper §4.1: 8K for a skill update
TARGET_MODEL = "Qwen3.5-9B (thinking off, 64K-token context, served by vLLM)"
BUILDER_MODEL = "claude-opus-5-5"           # Claude Code in the Docker sandbox (subscription)
