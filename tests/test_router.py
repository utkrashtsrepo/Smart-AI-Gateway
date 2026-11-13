"""Unit tests for AutoRouter heuristics, complexity scoring, and route explanations."""

import pytest
from app.router import AutoRouter
from app.schemas import ChatMessage


@pytest.fixture
def test_router():
    return AutoRouter(threshold=0.40)


def test_simple_factoid_routes_to_cheap(test_router):
    messages = [ChatMessage(role="user", content="What is the capital of France?")]
    decision = test_router.route(messages)
    assert decision.routed_model == test_router.cheap_model
    assert decision.complexity_score < 0.40
    assert "CHEAP" in decision.route_reason


def test_code_detection_routes_to_strong(test_router):
    code_prompt = """
    Please review this function for edge cases:
    ```python
    def binary_search(arr, target):
        low, high = 0, len(arr) - 1
        while low <= high:
            mid = (low + high) // 2
            if arr[mid] == target:
                return mid
        return -1
    ```
    """
    messages = [ChatMessage(role="user", content=code_prompt)]
    decision = test_router.route(messages)
    assert decision.routed_model == test_router.strong_model
    assert decision.complexity_score >= 0.40
    assert decision.signals.get("code_detected") is True
    assert "STRONG" in decision.route_reason


def test_multi_step_reasoning_routes_to_strong(test_router):
    reasoning_prompt = "Step by step, prove that the square root of 2 is irrational and calculate its approximation."
    messages = [ChatMessage(role="user", content=reasoning_prompt)]
    decision = test_router.route(messages)
    assert decision.routed_model == test_router.strong_model
    assert decision.complexity_score >= 0.40
    assert len(decision.signals.get("reasoning_keywords", [])) > 0


def test_policy_override_always_cheap(test_router):
    # Even with high complexity, override should force cheap
    complex_prompt = "```python\ndef solve(): pass\n``` step by step system design tradeoff"
    messages = [ChatMessage(role="user", content=complex_prompt)]
    decision = test_router.route(messages, policy_override="always_cheap")
    assert decision.routed_model == test_router.cheap_model
    assert decision.policy == "always_cheap"


def test_policy_override_always_strong(test_router):
    # Even for simple greeting, override should force strong
    messages = [ChatMessage(role="user", content="hi")]
    decision = test_router.route(messages, policy_override="always_strong")
    assert decision.routed_model == test_router.strong_model
    assert decision.policy == "always_strong"


# =========================================================================
# EDGE-CASE TESTS FOR WORD BOUNDARY & SYSTEM ARCHITECTURE
# =========================================================================

def test_greeting_word_boundary_does_not_collide_with_architecture(test_router):
    """Verify that 'hi' does not substring-match inside 'architecture' or 'machine'."""
    # A prompt with 'architecture' must NOT trigger 'hi' as a simple keyword
    prompt = "Explain modern event-driven system architecture and its main tradeoffs."
    messages = [ChatMessage(role="user", content=prompt)]
    score, reasons, signals = test_router.analyze_text(prompt)

    # Must NOT contain 'hi' in simple keywords
    assert "hi" not in signals.get("simple_keywords", [])
    # Must detect 'architecture' or 'tradeoffs' as reasoning keywords
    assert any(kw in ["architecture", "system architecture", "tradeoffs", "tradeoff", "event-driven", "event driven"] for kw in signals.get("reasoning_keywords", []))

    decision = test_router.route(messages)
    assert decision.routed_model == test_router.strong_model
    assert decision.complexity_score >= 0.40
    assert "STRONG" in decision.route_reason


def test_isolated_greeting_routes_to_cheap(test_router):
    """Verify that standalone greetings like 'hi' or 'hello' route to cheap."""
    for greeting in ["hi", "Hi there!", "hello team", "hey, good morning"]:
        messages = [ChatMessage(role="user", content=greeting)]
        decision = test_router.route(messages)
        assert decision.routed_model == test_router.cheap_model
        assert decision.complexity_score < 0.40
        assert "CHEAP" in decision.route_reason


def test_distributed_systems_and_scalability_route_to_strong(test_router):
    """Verify distributed systems, scalability, and system design prompts route to strong tier."""
    test_prompts = [
        "How do you design a distributed systems rate limiting service?",
        "Compare Paxos vs Raft consensus algorithm for cluster replication.",
        "What are the tradeoffs of horizontal scaling with database sharding and CAP theorem?",
        "Explain high availability, fault tolerance, and circuit breaker patterns in microservices.",
        "How would you handle deadlock and concurrency in a high-throughput message queue like Kafka?",
    ]

    for prompt in test_prompts:
        messages = [ChatMessage(role="user", content=prompt)]
        decision = test_router.route(messages)
        assert decision.routed_model == test_router.strong_model, f"Failed for: {prompt}"
        assert decision.complexity_score >= 0.40
        assert "STRONG" in decision.route_reason
        assert len(decision.signals.get("reasoning_keywords", [])) > 0


def test_simple_factoid_with_short_words_routes_to_cheap(test_router):
    """Verify simple queries remain cheap and don't falsely trigger complex rules."""
    simple_prompts = [
        "What is the capital of Japan?",
        "Who is the CEO of Google?",
        "Translate 'Good night' to French.",
        "Define what an API is in one sentence.",
        "Yes or no: is Python an interpreted language?",
    ]

    for prompt in simple_prompts:
        messages = [ChatMessage(role="user", content=prompt)]
        decision = test_router.route(messages)
        assert decision.routed_model == test_router.cheap_model, f"Failed for: {prompt}"
        assert decision.complexity_score < 0.40
        assert "CHEAP" in decision.route_reason


