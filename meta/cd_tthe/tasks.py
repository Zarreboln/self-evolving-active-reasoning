"""Task registry for the TTHE loop: everything task-specific the controller, the worker, the prompts
and the audits need. `cd` = Circuit Decoding (Multi-Turn Puzzles), `sp` = Situation Puzzles (AR-Bench)."""
from __future__ import annotations

SPECS = {
    "cd": {
        "name": "Circuit Decoding", "agents": "agents", "base_file": "harness_base.py",
        "base_import": "from harness_base import CDHarness", "base_class": "CDHarness", "budget": 18,
        "env_module": "bridge", "signals_module": "signals",
        "public_signals": ("health", "crashed", "parsed_guess", "n_queries", "repeat"),
        "outcome_keys": ("parsed_guess", "joint_truth_table_correct", "circuit_wise_accuracy"),
        "metrics": (("circuit_wise_accuracy", "circuit-wise acc"), ("joint_truth_table_correct", "joint table correct")),
        "status_fields": (("parsed_final_guess", "parsed_guess"), ("queries_used", "n_queries")),
        "history_title": "Queries and answers (in order)",
        "data_rule": r"cd_test\.jsonl|mt_puzzles/[A-Za-z0-9_]+\.(?:jsonl|txt)",
        "audit_forbidden": r"_truth|\btruth\b|INSTANCES|load_instances|cd_test|circuit_joint_lookup",
        "allowed_import": "harness_base",
        "unit": "queries", "final": "the 24-bit table",
    },
    "sp": {
        "name": "Situation Puzzles (AR-Bench)", "agents": "agents_sp", "base_file": "sp_harness_base.py",
        "base_import": "from sp_harness_base import SPHarness", "base_class": "SPHarness", "budget": 25,
        "env_module": "sp_bridge", "signals_module": "sp_signals",
        "public_signals": ("health", "crashed", "parsed_answer", "n_questions", "repeat", "yes_rate", "no_rate",
                           "unknown_rate", "not_question_rate", "mean_q_words", "answer_words"),
        "outcome_keys": ("parsed_answer", "f1_char", "keypoint_recall"),
        "metrics": (("f1_char", "F1 (char)"), ("keypoint_recall", "key-question recall")),
        "status_fields": (("parsed_final_answer", "parsed_answer"), ("questions_used", "n_questions"), ("repeat", "repeat"),
                          ("yes", "yes_rate"), ("no", "no_rate"), ("unknown", "unknown_rate"), ("not_a_question", "not_question_rate")),
        "history_title": "Questions and referee replies (in order)",
        "data_rule": r"AR-Bench/|arbench|data/sp/|sp/test\.json|sp/train\.json",
        "audit_forbidden": r"_bottom|_keypoints|key_question|story_tree|INSTANCES|load_instances|import arbench|from arbench|test\.json",
        "allowed_import": "sp_harness_base",
        "unit": "questions", "final": "the final answer",
    },
}


def history_line(task, h):
    if task == "cd":
        (c, a, b, x), bit = h
        return f"- {c}({a}, {b}, {x}) = {bit}"
    q, a = h
    return f"- Q: {q}  ->  {a}"


def outcome_line(task, g):
    g = g or {}
    if task == "cd":
        return f" | OUTCOME: joint truth table correct={g.get('joint_truth_table_correct')} circuit-wise accuracy={round(3 * float(g.get('circuit_wise_accuracy') or 0))}/3"
    f1 = g.get("f1_char"); kr = g.get("keypoint_recall")
    return f" | OUTCOME: F1 (char) vs the true story={f1 if f1 is None else round(float(f1), 3)} key-question recall={kr if kr is None else round(float(kr), 2)}"
