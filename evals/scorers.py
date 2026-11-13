"""Custom evaluation scorers for Agent answers, tool selection, and trajectory efficiency.

These functions are intentionally written from scratch without external black-box eval libraries,
ensuring 100% transparency, determinism, and explainability for engineering interviews.
"""

import re
from typing import List, Literal


def normalize_text(text: str) -> str:
    """Lowercase and strip whitespace, punctuation, and markdown formatting."""
    text = text.lower().strip()
    text = re.sub(r"[\.,;:!\?'\"\(\)\[\]\{\}]", "", text)
    return re.sub(r"\s+", " ", text)


def extract_numbers(text: str) -> List[float]:
    """Extract all integer and floating point numbers from a text string."""
    matches = re.findall(r"[-+]?\d*\.?\d+", text.replace(",", ""))
    results = []
    for m in matches:
        try:
            results.append(float(m))
        except ValueError:
            continue
    return results


def score_answer(
    actual: str,
    expected: str,
    eval_type: Literal["exact", "contains", "numeric", "regex"] = "contains",
    tolerance_pct: float = 2.0,
) -> float:
    """Score answer accuracy from 0.0 (fail) to 1.0 (perfect).
    
    Args:
        actual: The agent's generated answer.
        expected: Ground truth answer.
        eval_type: Scoring strategy.
        tolerance_pct: Acceptable numerical error percentage for 'numeric' eval.
    """
    if not actual or not actual.strip():
        return 0.0

    actual_norm = normalize_text(actual)
    expected_norm = normalize_text(expected)

    # 1. Exact Match
    if eval_type == "exact":
        return 1.0 if actual_norm == expected_norm else 0.0

    # 2. Substring / Entity Containment
    if eval_type == "contains":
        if expected_norm in actual_norm:
            return 1.0
        # Partial match if multiple words
        words = expected_norm.split()
        if len(words) > 1:
            matched_words = sum(1 for w in words if w in actual_norm)
            return round(matched_words / len(words), 2)
        return 0.0

    # 3. Numeric Proximity
    if eval_type == "numeric":
        expected_nums = extract_numbers(expected)
        actual_nums = extract_numbers(actual)

        if not expected_nums or not actual_nums:
            return 1.0 if expected_norm in actual_norm else 0.0

        target = expected_nums[0]
        # Find closest number in actual response
        closest = min(actual_nums, key=lambda x: abs(x - target))
        if target == 0:
            return 1.0 if closest == 0 else 0.0

        pct_diff = abs(closest - target) / abs(target) * 100.0
        if pct_diff <= tolerance_pct:
            return 1.0
        elif pct_diff <= 10.0:
            return round(max(0.0, 1.0 - (pct_diff / 10.0)), 2)
        return 0.0

    # 4. Regular Expression
    if eval_type == "regex":
        return 1.0 if re.search(expected, actual, re.IGNORECASE) else 0.0

    return 0.0


def score_tools(tools_used: List[str], expected_tools: List[str]) -> float:
    """Score tool selection accuracy via Precision, Recall, and F1."""
    used_set = set(t.lower() for t in tools_used)
    expected_set = set(t.lower() for t in expected_tools)

    if not expected_set:
        # Task did not require any tools
        if not used_set:
            return 1.0
        # Penalize unnecessary tool calls
        return max(0.5, 1.0 - 0.2 * len(used_set))

    if not used_set:
        # Needed tools but called none
        return 0.0

    true_positives = len(used_set.intersection(expected_set))
    precision = true_positives / len(used_set)
    recall = true_positives / len(expected_set)

    if precision + recall == 0:
        return 0.0

    f1 = 2 * (precision * recall) / (precision + recall)
    return round(f1, 2)


def score_trajectory(steps: int, expected_steps: int = 2, max_steps: int = 5) -> float:
    """Measure trajectory efficiency: penalizes wandering in loops."""
    if steps <= expected_steps:
        return 1.0
    if steps >= max_steps:
        return 0.4
    # Linear penalty for extra steps
    penalty = (steps - expected_steps) * 0.2
    return round(max(0.4, 1.0 - penalty), 2)


def composite_score(
    answer_score: float,
    tool_score: float,
    trajectory_score: float,
    weights: tuple = (0.60, 0.25, 0.15),
) -> float:
    """Weighted composite metric across correctness, tool usage, and efficiency."""
    w_ans, w_tool, w_traj = weights
    total = (w_ans * answer_score) + (w_tool * tool_score) + (w_traj * trajectory_score)
    return round(total, 3)

