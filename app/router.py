"""Explainable Multi-Signal Auto-Router for LLM task classification and model selection.

This module evaluates incoming requests and dynamically routes them to either:
1. CHEAP_MODEL: For simple, conversational, factoid, or extraction tasks.
2. STRONG_MODEL: For complex reasoning, code generation/debugging, multi-step math, or architecture.

The router uses transparent heuristic rules that can be inspected, logged, and defended in interviews.
"""

import re
from typing import Any, Dict, List, Tuple
from app.config import get_settings
from app.schemas import ChatMessage, RouteDecision


# Regex patterns for code detection
CODE_PATTERNS = [
    r"```[a-zA-Z]*\n[\s\S]*?```",  # Markdown code blocks
    r"\b(def|class|import|from\s+\w+\s+import|lambda|async\s+def)\b",  # Python
    r"\b(function|const|let|var|console\.log|export\s+default)\b",  # JS/TS
    r"\b(SELECT\s+.+\s+FROM\s+.+|INSERT\s+INTO|UPDATE\s+.+\s+SET|DELETE\s+FROM)\b",  # SQL
    r"[\{\}\[\];]{3,}",  # Dense syntax tokens
    r"\b(traceback|NullPointerException|SyntaxError|TypeError|ValueError)\b",  # Errors
]

# Keywords indicating complex logic, math, system architecture, or multi-step reasoning
REASONING_KEYWORDS = [
    # Multi-step & mathematical reasoning
    "step by step",
    "calculate",
    "solve",
    "equation",
    "derivative",
    "integral",
    "probability",
    "matrix",
    "algorithm",
    "prove that",
    "derive",
    "compare and contrast",
    "tradeoff",
    "tradeoffs",
    "root cause analysis",
    "optimize",
    "time complexity",
    "space complexity",
    "big o",
    "multi-step",
    # System Architecture, Distributed Systems & Scalability
    "system architecture",
    "software architecture",
    "architectural",
    "architecture",
    "system design",
    "distributed systems",
    "distributed system",
    "scalability",
    "high availability",
    "fault tolerance",
    "horizontal scaling",
    "vertical scaling",
    "event driven",
    "event-driven",
    "microservices",
    "microservice",
    "load balancer",
    "load balancing",
    "cap theorem",
    "acid vs base",
    "database sharding",
    "sharding",
    "replication",
    "consensus algorithm",
    "raft",
    "paxos",
    "message queue",
    "kafka",
    "concurrency",
    "deadlock",
    "rate limiting",
    "circuit breaker",
]

# Keywords indicating lightweight, trivial, or direct retrieval tasks
SIMPLE_KEYWORDS = [
    "what is the capital",
    "who is",
    "who was",
    "define",
    "translate",
    "sentiment of",
    "fix typo",
    "spell check",
    "yes or no",
    "in one sentence",
    "extract the email",
    "summarize briefly",
    "hello",
    "hi",
    "hey",
    "greetings",
    "thank you",
]


def match_keywords_with_word_boundaries(text: str, keywords: List[str]) -> List[str]:
    """Match keywords using regex word boundaries to prevent substring collisions (e.g. 'hi' in 'architecture')."""
    matched = []
    for kw in keywords:
        # Escape any special regex characters in the keyword
        escaped_kw = re.escape(kw.strip().lower())
        # Replace escaped spaces with \s+ to match variable whitespace between multi-word phrases
        pattern_str = r"\b" + re.sub(r"\\ ", r"\\s+", escaped_kw) + r"\b"
        if re.search(pattern_str, text, re.IGNORECASE):
            matched.append(kw)
    return matched


class AutoRouter:
    """Heuristic, multi-signal complexity analyzer and model selector."""

    def __init__(self, threshold: float | None = None):
        settings = get_settings()
        self.threshold = threshold if threshold is not None else settings.ROUTER_COMPLEXITY_THRESHOLD
        self.cheap_model = settings.CHEAP_MODEL
        self.strong_model = settings.STRONG_MODEL

    def analyze_text(self, text: str) -> Tuple[float, List[str], Dict[str, Any]]:
        """Compute complexity score (0.0 to 1.0) and collect triggered signals."""
        reasons: List[str] = []
        signals: Dict[str, Any] = {}
        score: float = 0.0

        lower_text = text.lower()
        word_count = len(text.split())
        char_count = len(text)
        signals["word_count"] = word_count
        signals["char_count"] = char_count

        # 1. Length Signal
        if word_count > 160 or char_count > 900:
            score += 0.35
            reasons.append(f"high context length ({word_count} words, +0.35)")
            signals["length_signal"] = "high"
        elif word_count > 60:
            score += 0.15
            reasons.append(f"moderate prompt length ({word_count} words, +0.15)")
            signals["length_signal"] = "moderate"
        else:
            signals["length_signal"] = "short"

        # 2. Code & Programming Signal
        code_matched = []
        for pattern in CODE_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                code_matched.append(pattern)
        
        if code_matched:
            score += 0.45
            reasons.append("code syntax / structure detected (+0.45)")
            signals["code_detected"] = True
        else:
            signals["code_detected"] = False

        # 3. Reasoning & Mathematical Keywords (Word-boundary matching)
        math_matches = match_keywords_with_word_boundaries(lower_text, REASONING_KEYWORDS)
        if math_matches:
            # Score scales slightly with number of reasoning indicators
            boost = min(0.50, 0.35 + (0.05 * len(math_matches)))
            score += boost
            reasons.append(f"reasoning keywords [{', '.join(math_matches[:3])}] (+{boost:.2f})")
            signals["reasoning_keywords"] = math_matches
        else:
            signals["reasoning_keywords"] = []

        # 4. Simple Factoid / Greeting Dampening (Word-boundary matching)
        # Only dampen if no strong reasoning or code was detected
        simple_matches = match_keywords_with_word_boundaries(lower_text, SIMPLE_KEYWORDS)
        if simple_matches and not code_matched and not math_matches:
            reduction = 0.20
            score -= reduction
            reasons.append(f"simple extraction / greeting marker [{simple_matches[0]}] (-{reduction:.2f})")
            signals["simple_keywords"] = simple_matches
        else:
            signals["simple_keywords"] = []

        # Clamp score within [0.0, 1.0]
        final_score = round(max(0.0, min(1.0, score)), 3)
        signals["complexity_score"] = final_score

        return final_score, reasons, signals

    def route(
        self,
        messages: List[ChatMessage],
        requested_model: str = "auto",
        policy_override: str | None = None,
    ) -> RouteDecision:
        """Determine destination model and generate explanation.
        
        Policies:
        - always_cheap: forces cheap model regardless of complexity.
        - always_strong: forces strong model regardless of complexity.
        - auto: applies multi-signal complexity scoring.
        """
        # Manual Policy Overrides
        if policy_override == "always_cheap" or requested_model.lower() == "cheap":
            return RouteDecision(
                routed_model=self.cheap_model,
                policy="always_cheap",
                complexity_score=0.0,
                route_reason="Policy override: forced to always_cheap",
                signals={"policy_override": "always_cheap"},
            )

        if policy_override == "always_strong" or requested_model.lower() == "strong":
            return RouteDecision(
                routed_model=self.strong_model,
                policy="always_strong",
                complexity_score=1.0,
                route_reason="Policy override: forced to always_strong",
                signals={"policy_override": "always_strong"},
            )

        # If user explicitly requested a specific concrete model (not 'auto')
        if requested_model not in ("auto", "", "cheap", "strong"):
            return RouteDecision(
                routed_model=requested_model,
                policy="explicit",
                complexity_score=0.5,
                route_reason=f"Explicit model request: {requested_model}",
                signals={"explicit_model": requested_model},
            )

        # Extract user messages for task complexity evaluation
        # (Focus on user requests to prevent static system prompts/tool definitions from inflating complexity)
        user_texts = [m.content for m in messages if m.role == "user" and m.content]
        if user_texts:
            text_to_analyze = "\n".join(user_texts)
        else:
            text_to_analyze = "\n".join([m.content for m in messages if m.content])

        score, reasons, signals = self.analyze_text(text_to_analyze)

        if score >= self.threshold:
            selected_model = self.strong_model
            tier = "STRONG"
        else:
            selected_model = self.cheap_model
            tier = "CHEAP"

        reason_str = f"Routed to {tier} (score: {score:.2f} vs threshold {self.threshold:.2f}): " + (
            "; ".join(reasons) if reasons else "baseline low-complexity text"
        )

        return RouteDecision(
            routed_model=selected_model,
            policy="auto",
            complexity_score=score,
            route_reason=reason_str,
            signals=signals,
        )


# Global router instance
router = AutoRouter()

