"""Tests for Agent Tools and Custom Evaluation Scorers."""

import pytest
from app.agent import SafeCalculator, knowledge_lookup, python_repl
from evals.scorers import composite_score, score_answer, score_tools, score_trajectory


def test_safe_calculator():
    assert SafeCalculator.evaluate("144 * 12") == "1728"
    assert SafeCalculator.evaluate("(100 - 25) / 5") == "15"
    assert SafeCalculator.evaluate("2 ** 8") == "256"


def test_knowledge_lookup():
    res = knowledge_lookup("q3_revenue")
    assert "$14.2 Million" in res

    uptime = knowledge_lookup("uptime")
    assert "99.982%" in uptime


def test_python_repl():
    code = """
numbers = [1, 2, 3, 4, 5]
result = sum(numbers)
"""
    assert python_repl(code) == "15"


def test_scorers():
    # 1. Answer scoring
    assert score_answer("Paris is the capital.", "Paris", eval_type="contains") == 1.0
    assert score_answer("The result is 10500.", "10500", eval_type="numeric") == 1.0
    assert score_answer("The result is 10800.", "10500", eval_type="numeric", tolerance_pct=1.0) < 1.0

    # 2. Tool scoring
    assert score_tools(["calculator"], ["calculator"]) == 1.0
    assert score_tools([], []) == 1.0
    assert score_tools([], ["calculator"]) == 0.0

    # 3. Trajectory scoring
    assert score_trajectory(steps=2, expected_steps=2) == 1.0
    assert score_trajectory(steps=5, expected_steps=2) == 0.4

    # 4. Composite score
    comp = composite_score(1.0, 1.0, 1.0)
    assert comp == 1.0
