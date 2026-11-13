"""Autonomous Tool-Calling Agent equipped with Calculator, Knowledge Lookup, and Python REPL.

The agent queries the Smart AI Gateway, demonstrating how auto-routing, caching,
and reliability behave during multi-step agentic workflows.
"""

import ast
import operator
import re
import time
from typing import Any, Callable, Dict, List, Optional
from pydantic import BaseModel

from app.schemas import ChatCompletionRequest, ChatMessage


# =====================================================================
# TOOLS
# =====================================================================

class SafeCalculator:
    """Safe arithmetic expression evaluator using AST (no unsafe eval)."""

    ALLOWED_OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
        ast.USub: operator.neg,
    }

    @classmethod
    def evaluate(cls, expression: str) -> str:
        try:
            clean_expr = expression.replace(",", "").replace("^", "**").strip()
            node = ast.parse(clean_expr, mode="eval")

            def _eval_node(n):
                if isinstance(n, ast.Expression):
                    return _eval_node(n.body)
                elif isinstance(n, ast.Constant):
                    return n.value
                elif isinstance(n, ast.UnaryOp):
                    op_type = type(n.op)
                    if op_type in cls.ALLOWED_OPERATORS:
                        return cls.ALLOWED_OPERATORS[op_type](_eval_node(n.operand))
                elif isinstance(n, ast.BinOp):
                    op_type = type(n.op)
                    if op_type in cls.ALLOWED_OPERATORS:
                        left = _eval_node(n.left)
                        right = _eval_node(n.right)
                        return cls.ALLOWED_OPERATORS[op_type](left, right)
                raise ValueError(f"Unsupported operation in: {expression}")

            val = _eval_node(node)
            if isinstance(val, float) and val.is_integer():
                val = int(val)
            return str(round(val, 4) if isinstance(val, float) else val)
        except Exception as e:
            return f"Error evaluating expression '{expression}': {e}"


# Static mock knowledge repository for Agent retrieval
KNOWLEDGE_STORE: Dict[str, str] = {
    "q3_revenue": "In Q3 2026, the company reported total gross revenue of $14.2 Million, a 18% YoY growth.",
    "server_uptime": "The European cluster achieved 99.982% uptime over the past 90 days.",
    "pricing_llama_8b": "Llama-3.1-8b-instant costs $0.05 per 1M input tokens and $0.08 per 1M output tokens.",
    "pricing_llama_70b": "Llama-3.3-70b-versatile costs $0.59 per 1M input tokens and $0.79 per 1M output tokens.",
    "mars_distance": "The average distance between Earth and Mars is approximately 225 million kilometers.",
    "paris_population": "The urban population of Paris is approximately 2.1 million residents.",
    "lead_engineer": "The principal AI systems architect is Alex Chen.",
}


def knowledge_lookup(query: str) -> str:
    """Lookup facts, system metrics, and domain documents."""
    q_clean = query.lower().strip()
    # Match direct keys or keywords
    for key, text in KNOWLEDGE_STORE.items():
        if key in q_clean or all(word in text.lower() for word in q_clean.split() if len(word) > 3):
            return text
    # Partial word match
    for key, text in KNOWLEDGE_STORE.items():
        if any(w in key for w in q_clean.split()):
            return text
    return f"No document matched query '{query}'. Available topics: {', '.join(KNOWLEDGE_STORE.keys())}"


def python_repl(code: str) -> str:
    """Safe execution of short Python expressions/statements."""
    try:
        # Strip markdown syntax if passed
        clean_code = re.sub(r"```python|```", "", code).strip()
        local_scope: Dict[str, Any] = {}
        exec(clean_code, {"__builtins__": {"len": len, "range": range, "sum": sum, "min": min, "max": max, "sorted": sorted}}, local_scope)
        if "result" in local_scope:
            return str(local_scope["result"])
        if local_scope:
            return str(local_scope)
        return "Executed successfully (no output variable 'result')."
    except Exception as e:
        return f"Python execution error: {e}"


TOOLS: Dict[str, Callable[[str], str]] = {
    "calculator": SafeCalculator.evaluate,
    "knowledge_lookup": knowledge_lookup,
    "python_repl": python_repl,
}


SYSTEM_PROMPT = """You are a smart, analytical AI assistant with access to three specialized tools:
1. calculator: Evaluates mathematical expressions. Input: string math expression like '144 * 12 + 50'.
2. knowledge_lookup: Looks up company metrics, distances, facts, and documents. Input: keyword topic.
3. python_repl: Executes python code to calculate or process data. Input: python code storing answer in 'result'.

To use a tool, respond with EXACTLY:
Thought: <your reasoning about what to do>
Action: <calculator, knowledge_lookup, or python_repl>
Action Input: <the input string for the tool>

When you have the final answer, respond with:
Thought: I know the final answer.
Final Answer: <your definitive answer>
"""


class AgentTrajectory(BaseModel):
    task_id: str
    question: str
    final_answer: str
    tools_used: List[str]
    steps_taken: int
    latency_ms: float
    cost_usd: float
    models_used: List[str]
    trace: List[Dict[str, str]]


class ToolCallingAgent:
    """ReAct loop agent interacting through the Gateway completions endpoint."""

    def __init__(self, gateway_client: Any = None, max_steps: int = 5):
        self.gateway_client = gateway_client
        self.max_steps = max_steps

    async def execute_task(
        self,
        task_id: str,
        question: str,
        policy: Optional[str] = None,
        model: str = "auto",
        use_cache: bool = True,
    ) -> AgentTrajectory:
        """Run the ReAct agent cycle for a given prompt."""
        # Late import to prevent circular dependency
        from app.main import chat_completions

        start_time = time.perf_counter()
        tools_used: List[str] = []
        models_used: List[str] = []
        trace: List[Dict[str, str]] = []
        total_cost: float = 0.0

        messages: List[ChatMessage] = [
            ChatMessage(role="system", content=SYSTEM_PROMPT),
            ChatMessage(role="user", content=question),
        ]

        final_answer = ""
        steps_taken = 0

        for step in range(self.max_steps):
            steps_taken += 1
            req = ChatCompletionRequest(
                messages=messages,
                model=model,
                temperature=0.0,
                policy=policy,  # type: ignore
                use_cache=use_cache,
            )

            resp = await chat_completions(req)
            ai_text = resp.choices[0].message.content if resp.choices else ""
            meta = resp.gateway_metadata
            routed_model = meta.get("routed_model", resp.model)
            cost = meta.get("cost_usd", 0.0)

            total_cost += cost
            models_used.append(routed_model)
            trace.append({"step": str(step + 1), "thought_and_action": ai_text})

            # Check if final answer reached
            if "Final Answer:" in ai_text:
                final_answer = ai_text.split("Final Answer:")[-1].strip()
                break

            # Parse tool action
            action_match = re.search(r"Action:\s*([a-zA-Z_]+)", ai_text)
            input_match = re.search(r"Action Input:\s*(.+)", ai_text, re.DOTALL)

            if action_match:
                tool_name = action_match.group(1).strip()
                tool_input = input_match.group(1).strip() if input_match else ""
                # Strip trailing thoughts if any
                if "\nThought:" in tool_input:
                    tool_input = tool_input.split("\nThought:")[0].strip()

                if tool_name in TOOLS:
                    tools_used.append(tool_name)
                    obs = TOOLS[tool_name](tool_input)
                else:
                    obs = f"Error: Tool '{tool_name}' does not exist. Choose from {list(TOOLS.keys())}"

                trace.append({"tool": tool_name, "input": tool_input, "observation": obs})
                # Append assistant thoughts and observation to history
                messages.append(ChatMessage(role="assistant", content=ai_text))
                messages.append(ChatMessage(role="user", content=f"Observation: {obs}"))
            else:
                # Direct answer without explicit Final Answer keyword
                final_answer = ai_text.strip()
                break

        if not final_answer:
            final_answer = trace[-1].get("thought_and_action", "No answer produced")

        total_latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return AgentTrajectory(
            task_id=task_id,
            question=question,
            final_answer=final_answer,
            tools_used=tools_used,
            steps_taken=steps_taken,
            latency_ms=total_latency_ms,
            cost_usd=round(total_cost, 6),
            models_used=models_used,
            trace=trace,
        )

