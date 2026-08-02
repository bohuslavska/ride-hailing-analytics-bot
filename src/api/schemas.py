"""Request and response models for the HTTP layer."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    # Trimmed server-side rather than trusted: a client could otherwise grow the
    # prompt without limit by replaying an ever-longer history.
    history: list[ChatTurn] = Field(default_factory=list, max_length=20)

    @field_validator("question")
    @classmethod
    def question_is_not_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("The question cannot be blank.")
        return cleaned

    def history_as_dicts(self) -> list[dict[str, str]]:
        return [turn.model_dump() for turn in self.history]


class ChatResponse(BaseModel):
    answer: str
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    database: bool
    # True when Redis answers, False when configured but down, None when unused.
    redis: bool | None = None
    llm_configured: bool
    calculated_rides: int | None = None
    detail: str | None = None
