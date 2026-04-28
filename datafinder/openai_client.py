"""Thin wrapper around the OpenAI SDK that satisfies our
ChatClient protocol. Imported lazily so the rest of the package
runs without the openai dep installed."""

from __future__ import annotations

from typing import Any


class OpenAIChatClient:
    def __init__(self, model: str = "gpt-4o-mini") -> None:
        from openai import OpenAI  # imported here so tests don't need it
        self._client = OpenAI()
        self.model = model

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        # The SDK returns a Pydantic-like object; we coerce to the
        # plain-dict shape the agent loop reads.
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        # Convert ChatCompletion to dict form. We only access
        # choices[0].message; the rest is ignored.
        choice = resp.choices[0]
        m = choice.message
        msg: dict[str, Any] = {"role": m.role, "content": m.content}
        tcs = getattr(m, "tool_calls", None) or []
        if tcs:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in tcs
            ]
        return {"choices": [{"message": msg}]}
