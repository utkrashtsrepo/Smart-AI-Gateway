"""High-fidelity Mock Provider for zero-cost offline development, testing, and deterministic evaluations."""

import asyncio
import re
from typing import Any, Dict, List, Optional
from app.providers.base import BaseProvider, ProviderResponse


class MockProvider(BaseProvider):
    """Simulates realistic LLM responses with deterministic logic and simulated failure modes."""

    def __init__(self, failure_rate: float = 0.0):
        self.failure_rate = failure_rate
        self.call_count = 0
        self.fail_next_n_calls = 0

    def set_forced_failures(self, count: int) -> None:
        """Instruct provider to fail the next N calls to test retries and fallback."""
        self.fail_next_n_calls = count

    async def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> ProviderResponse:
        self.call_count += 1

        # Check for simulated transient errors
        if self.fail_next_n_calls > 0:
            self.fail_next_n_calls -= 1
            raise RuntimeError("MockProvider simulated 503 Service Unavailable")

        # Simulate micro latency
        await asyncio.sleep(0.01)

        last_user_message = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user_message = m.get("content", "")
                break

        model_lower = model.lower()
        is_strong = ("120b" in model_lower or "70b" in model_lower or "strong" in model_lower) and not ("20b" in model_lower or "8b" in model_lower)
        content = self._generate_simulated_reply(last_user_message, is_strong=is_strong)

        # Estimate realistic token counts
        prompt_text = "".join([m.get("content", "") for m in messages])
        prompt_tokens = max(10, len(prompt_text) // 4)
        completion_tokens = max(5, len(content) // 4)
        total_tokens = prompt_tokens + completion_tokens

        return ProviderResponse(
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            raw_response={
                "id": f"mock-cmpl-{self.call_count}",
                "model": model,
                "choices": [{"message": {"role": "assistant", "content": content}}],
            },
            model_name=model,
        )

    def _generate_simulated_reply(self, prompt: str, is_strong: bool) -> str:
        lower = prompt.lower()

        # Handle ReAct agent observation loop (when prompt contains Observation:)
        obs_match = re.search(r"Observation:\s*(.+)", prompt, re.DOTALL)
        if obs_match:
            obs_val = obs_match.group(1).strip()
            # If observation gave an answer or number, finalize it
            if is_strong:
                return f"Thought: Based on the tool observation ({obs_val}), I have verified the result.\nFinal Answer: {obs_val}"
            else:
                return f"Final Answer: {obs_val}"

        # 1. Factoid and Knowledge queries
        factoids = {
            "france": "Paris",
            "japan": "Tokyo",
            "germany": "Berlin",
            "india": "New Delhi",
            "italy": "Rome",
            "canada": "Ottawa",
            "australia": "Canberra",
            "api stands for": "Application Programming Interface",
            "what does http stand for": "Hypertext Transfer Protocol",
            "chemical symbol for gold": "Au",
            "sentiment": "positive",
            "spanish": "Hola",
            "support@gateway.io": "support@gateway.io",
        }
        for query_key, ans in factoids.items():
            if query_key in lower:
                return f"Thought: I know the final answer directly.\nFinal Answer: {ans}"

        # 2. Knowledge Base Retrieval tasks
        if "q3 revenue" in lower or "q3" in lower and "revenue" in lower:
            if is_strong:
                return "Thought: I need to query the internal knowledge base for Q3 revenue.\nAction: knowledge_lookup\nAction Input: q3_revenue"
            else:
                return "I think our revenue was around 10 million."

        if "uptime" in lower:
            if is_strong:
                return "Thought: Looking up server uptime in the repository.\nAction: knowledge_lookup\nAction Input: server_uptime"
            else:
                return "The server uptime is 99%."

        if "llama_8b" in lower or "pricing" in lower and "8b" in lower:
            if is_strong:
                return "Thought: Looking up model pricing for 8b.\nAction: knowledge_lookup\nAction Input: pricing_llama_8b"
            else:
                return "Pricing is roughly cheap."

        if "llama_70b" in lower or "pricing" in lower and "70b" in lower:
            if is_strong:
                return "Thought: Looking up model pricing for 70b.\nAction: knowledge_lookup\nAction Input: pricing_llama_70b"
            else:
                return "Pricing is high."

        if "architect" in lower or "alex chen" in lower:
            if is_strong:
                return "Thought: Querying the lead engineer.\nAction: knowledge_lookup\nAction Input: lead_engineer"
            else:
                return "Alex Chen"

        if "mars" in lower and "distance" in lower:
            if is_strong:
                return "Thought: Looking up distance to Mars.\nAction: knowledge_lookup\nAction Input: mars_distance"
            else:
                return "About 200 million km."

        if "paris" in lower and "population" in lower:
            if is_strong:
                return "Thought: Looking up Paris population.\nAction: knowledge_lookup\nAction Input: paris_population"
            else:
                return "2 million"

        # 3. Code Generation tasks (python_repl)
        if "python_repl" in lower or "def fib" in lower or "prime numbers between" in lower or "occurrences of the letter 'e'" in lower:
            if is_strong:
                if "prime" in lower:
                    code = "primes = [x for x in range(10, 31) if all(x % i != 0 for i in range(2, int(x**0.5)+1))]; result = sum(primes)"
                    return f"Thought: I will execute python code to compute the prime sum.\nAction: python_repl\nAction Input: {code}"
                elif "fib" in lower:
                    code = "def fib(n): a, b = 1, 1\nfor _ in range(n-2): a, b = b, a+b\nreturn b\nresult = fib(12)"
                    return f"Thought: Writing fibonacci function.\nAction: python_repl\nAction Input: {code}"
                elif "occurrences" in lower or "letter 'e'" in lower:
                    code = "s = 'distributed asynchronous microservices architecture'; result = s.count('e')"
                    return f"Thought: Counting occurrences in python.\nAction: python_repl\nAction Input: {code}"
                else:
                    return "Thought: Executing code.\nAction: python_repl\nAction Input: result = 42"
            else:
                return "Final Answer: unable to run python REPL accurately."

        # 4. Arithmetic & Complex Multi-Step Math (Calculator)
        math_match = re.search(r"(\d+[\d\s\*\/\+\-\(\)\.\^]+)", prompt)
        if "calculator" in lower or "calculate" in lower or "evaluate" in lower or math_match:
            expr_str = ""
            if "125 * 84" in prompt:
                expr_str = "125 * 84"
            elif "4500 / 15" in prompt or "380" in prompt:
                expr_str = "(4500 / 15) + 380"
            elif "2 ** 10" in prompt:
                expr_str = "2 ** 10"
            elif "15 * 34.50" in prompt or "34.50" in prompt:
                expr_str = "15 * 34.50"
            elif "9876" in prompt:
                expr_str = "9876 - 4321 + 555"
            elif "2,500,000" in prompt or "2.107" in prompt:
                expr_str = "(2.5 * 0.59) + (0.8 * 0.79)"
            elif "650" in prompt and "35%" in prompt:
                expr_str = "8 * 650 * 1.35"
            elif "144 / 12" in prompt:
                expr_str = "((144 / 12) + 25) * 3 - 15"
            elif "binary search tree" in lower:
                return "Thought: Analyzing binary search tree depth for 7 elements: log2(7+1) = 3 comparisons.\nFinal Answer: 3"
            elif math_match:
                expr_str = math_match.group(1).strip()

            if expr_str:
                if is_strong:
                    return f"Thought: I must invoke the calculator tool to evaluate the mathematical expression accurately.\nAction: calculator\nAction Input: {expr_str}"
                else:
                    # Cheap model attempts to guess math or gives approximations
                    return f"Thought: I will guess.\nFinal Answer: approximately 1000"

        # 5. Default generic response
        if is_strong:
            return f"Thought: I know the answer.\nFinal Answer: Comprehensive verified answer for '{prompt.strip()}'."
        return f"Final Answer: Answer for '{prompt.strip()}'."

