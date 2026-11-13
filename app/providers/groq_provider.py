"""Groq API provider client using async HTTPX."""

import json
import re
from typing import Any, Dict, List, Optional
import httpx

from app.config import get_settings
from app.providers.base import BaseProvider, ProviderResponse


class GroqProvider(BaseProvider):
    """Async client for Groq Cloud API."""

    GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, api_key: Optional[str] = None):
        settings = get_settings()
        self.api_key = api_key or settings.GROQ_API_KEY
        self.default_timeout = settings.REQUEST_TIMEOUT_SECONDS

    async def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> ProviderResponse:
        """Call Groq API asynchronously."""
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is not configured.")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Clean message payloads: exclude None fields like name: null which Groq rejects
        cleaned_messages = []
        for m in messages:
            msg_obj = {"role": m.get("role", "user"), "content": m.get("content", "")}
            if m.get("name") is not None:
                msg_obj["name"] = m["name"]
            cleaned_messages.append(msg_obj)

        payload: Dict[str, Any] = {
            "model": model,
            "messages": cleaned_messages,
            "temperature": temperature,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens

        req_timeout = timeout or self.default_timeout

        async with httpx.AsyncClient(timeout=req_timeout) as client:
            response = await client.post(
                self.GROQ_API_URL,
                headers=headers,
                json=payload,
            )

            if response.status_code != 200:
                error_detail = response.text
                error_json = {}
                try:
                    error_json = response.json()
                    error_detail = error_json.get("error", {}).get("message", error_detail)
                except Exception:
                    pass

                # If model emitted tool format without declared tools, extract failed generation
                if error_json.get("error", {}).get("code") == "tool_use_failed":
                    failed_gen = str(error_json.get("error", {}).get("failed_generation", ""))
                    t_name = "calculator"
                    t_input = ""
                    try:
                        parsed_tool = json.loads(failed_gen)
                        t_name = parsed_tool.get("name", "calculator")
                        args = parsed_tool.get("arguments", "")
                        if isinstance(args, dict):
                            t_input = args.get("input") or args.get("expression") or json.dumps(args)
                        else:
                            t_input = str(args)
                    except Exception:
                        # Regex extraction if failed_generation is malformed json
                        m_name = re.search(r'"name":\s*"([^"]+)"', failed_gen)
                        if m_name:
                            t_name = m_name.group(1)
                        m_args = re.search(r'"arguments":\s*(.+)}', failed_gen)
                        if m_args:
                            t_input = m_args.group(1).strip().strip('"')

                    fallback_content = f"Action: {t_name}\nAction Input: {t_input}"
                    return ProviderResponse(
                        content=fallback_content,
                        prompt_tokens=len(str(messages)) // 4,
                        completion_tokens=len(fallback_content) // 4,
                        total_tokens=(len(str(messages)) + len(fallback_content)) // 4,
                        raw_response=error_json,
                        model_name=model,
                    )

                raise RuntimeError(
                    f"Groq API Error (HTTP {response.status_code}): {error_detail}"
                )

            data = response.json()
            choices = data.get("choices", [])
            message_dict = choices[0]["message"] if choices else {}
            content = message_dict.get("content") or ""

            # If model returned native tool_calls, synthesize into ReAct text format
            tool_calls = message_dict.get("tool_calls", [])
            if not content and tool_calls:
                call = tool_calls[0]
                fn_name = call.get("function", {}).get("name", "")
                fn_args = call.get("function", {}).get("arguments", "{}")
                try:
                    parsed_args = json.loads(fn_args)
                    fn_input = parsed_args.get("expression") or parsed_args.get("input") or fn_args
                except Exception:
                    fn_input = fn_args
                content = f"Action: {fn_name}\nAction Input: {fn_input}"

            usage = data.get("usage", {})

            prompt_tokens = usage.get("prompt_tokens", len(str(messages)) // 4)
            completion_tokens = usage.get("completion_tokens", len(content) // 4)
            total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)

            return ProviderResponse(
                content=content,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                raw_response=data,
                model_name=data.get("model", model),
            )

