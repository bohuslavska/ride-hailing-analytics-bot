"""
The tool-calling agent and its event stream.

`stream_answer` turns LangGraph's event feed into a small, stable set of events
that the transport layer can forward without interpreting them. Keeping that
vocabulary narrow means the SSE endpoint and the browser never have to know
which agent framework is underneath.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI

from src.bot.artifacts import ArtifactCollector
from src.bot.prompts import build_system_prompt
from src.bot.tools import build_tools
from src.config import settings
from src.observability import (
    CHAT_DURATION,
    CHAT_REQUESTS,
    TOOL_CALLS,
    record_token_usage,
)


class LLMNotConfiguredError(RuntimeError):
    """Raised when the agent is used without an API key configured."""


def build_model() -> ChatOpenAI:
    """Chat model pointed at OpenRouter, which is OpenAI-compatible."""
    if not settings.llm_enabled:
        raise LLMNotConfiguredError(
            "OPENROUTER_API_KEY is not set. Add it to .env to enable the chat "
            "assistant. The REST analytics endpoints work without it."
        )

    return ChatOpenAI(
        model=settings.openrouter_model,
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        temperature=settings.llm_temperature,
        streaming=True,
    )


def build_agent(collector: ArtifactCollector):
    """
    Compile an agent bound to one run's artifact collector.

    Compiling per request keeps concurrent conversations from writing artifacts
    into each other's collectors. The graph is cheap to build; the expensive
    part is the schema description inside the prompt, which is cached.
    """
    return create_agent(
        model=build_model(),
        tools=build_tools(collector),
        system_prompt=build_system_prompt(),
    )


def to_messages(
    question: str, history: list[dict[str, str]] | None = None
) -> list[BaseMessage]:
    """Convert transport-level chat history into LangChain messages."""
    messages: list[BaseMessage] = []

    for turn in history or []:
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        if turn.get("role") == "assistant":
            messages.append(AIMessage(content=content))
        else:
            messages.append(HumanMessage(content=content))

    messages.append(HumanMessage(content=question))
    return messages


def _describe_tool_call(name: str, raw_input: Any) -> dict[str, Any]:
    """Summarise a tool invocation for display, without dumping the whole payload."""
    arguments = raw_input
    if isinstance(arguments, dict) and "input" in arguments and len(arguments) == 1:
        arguments = arguments["input"]
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {"input": arguments}
    if not isinstance(arguments, dict):
        arguments = {"input": arguments}

    return {"tool": name, "arguments": arguments}


async def stream_answer(
    question: str,
    history: list[dict[str, str]] | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """
    Run the agent, yielding events as they happen.

    Event types:
      status   - a step began, for the "thinking" indicator
      tool     - a tool was called, with its arguments
      artifact - a chart or table for the browser to render
      token    - a fragment of the final answer
      error    - the run failed; the message is safe to display
      done     - the run finished, with the complete answer text

    Artifacts are drained after each tool completes rather than at the end, so
    a chart appears while the model is still composing its explanation of it.
    """
    collector = ArtifactCollector()
    started_at = time.perf_counter()

    try:
        agent = build_agent(collector)
    except LLMNotConfiguredError as error:
        CHAT_REQUESTS.labels(outcome="unavailable").inc()
        yield {"type": "error", "message": str(error)}
        return

    answer_parts: list[str] = []

    try:
        stream = agent.astream_events(
            {"messages": to_messages(question, history)},
            config={"recursion_limit": settings.agent_max_iterations * 2},
            version="v2",
        )

        async for event in stream:
            kind = event.get("event")

            if kind == "on_tool_start":
                tool_name = event.get("name", "tool")
                TOOL_CALLS.labels(tool=tool_name).inc()
                yield {
                    "type": "tool",
                    **_describe_tool_call(
                        tool_name, event.get("data", {}).get("input")
                    ),
                }

            elif kind == "on_tool_end":
                for artifact in collector.drain():
                    yield {"type": "artifact", "artifact": artifact}
                yield {"type": "status", "message": "interpreting results"}

            elif kind == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk")
                text = _chunk_text(chunk)
                if text:
                    answer_parts.append(text)
                    yield {"type": "token", "text": text}

            elif kind == "on_chat_model_end":
                # One agent run makes several model calls -- one per reasoning
                # step -- so usage is accumulated across all of them rather than
                # read once at the end.
                output = event.get("data", {}).get("output")
                record_token_usage(getattr(output, "usage_metadata", None))

    except Exception as error:  # noqa: BLE001 - reported to the user, not swallowed
        CHAT_REQUESTS.labels(outcome="failed").inc()
        CHAT_DURATION.observe(time.perf_counter() - started_at)
        yield {
            "type": "error",
            "message": f"The assistant failed to complete this request: {error}",
        }
        return

    # Anything a tool produced after the final tool_end event.
    for artifact in collector.drain():
        yield {"type": "artifact", "artifact": artifact}

    CHAT_REQUESTS.labels(outcome="answered").inc()
    CHAT_DURATION.observe(time.perf_counter() - started_at)

    yield {"type": "done", "answer": "".join(answer_parts).strip()}


def _chunk_text(chunk: Any) -> str:
    """
    Pull display text out of a streamed chunk.

    Chunk content is a plain string for most providers but a list of typed
    blocks for some, and tool-call fragments arrive on the same stream with
    empty content. Only text blocks are of interest here.
    """
    if chunk is None:
        return ""

    content = getattr(chunk, "content", chunk)

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)

    return ""


async def answer_question(
    question: str, history: list[dict[str, str]] | None = None
) -> dict[str, Any]:
    """Non-streaming convenience wrapper, used by tests and the JSON endpoint."""
    answer = ""
    artifacts: list[dict[str, Any]] = []
    tool_calls: list[dict[str, Any]] = []
    error: str | None = None

    async for event in stream_answer(question, history):
        if event["type"] == "done":
            answer = event["answer"]
        elif event["type"] == "artifact":
            artifacts.append(event["artifact"])
        elif event["type"] == "tool":
            tool_calls.append(event)
        elif event["type"] == "error":
            error = event["message"]

    return {
        "answer": answer,
        "artifacts": artifacts,
        "tool_calls": tool_calls,
        "error": error,
    }
