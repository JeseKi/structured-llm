from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from pydantic import BaseModel

from structured_llm import StructuredClient


class Answer(BaseModel):
    value: int


@dataclass
class FakeMessage:
    content: str


@dataclass
class FakeChoice:
    message: FakeMessage


@dataclass
class FakeResponse:
    choices: list[FakeChoice]


class FakeChatCompletions:
    def __init__(self, responses: list[str], *, fail_native: bool = False) -> None:
        self.responses = responses
        self.fail_native = fail_native
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> FakeResponse:
        self.calls.append(kwargs)
        if self.fail_native and "response_format" in kwargs:
            raise ValueError("response_format json_schema unsupported")
        return FakeResponse([FakeChoice(FakeMessage(self.responses.pop(0)))])


class FakeChat:
    def __init__(self, completions: FakeChatCompletions) -> None:
        self.completions = completions


class FakeOpenAI:
    def __init__(self, completions: FakeChatCompletions) -> None:
        self.chat = FakeChat(completions)


def test_native_call_returns_model() -> None:
    completions = FakeChatCompletions(['{"value": 3}'])
    client = StructuredClient(model="test", openai_client=FakeOpenAI(completions))

    result = client.run("return a value", Answer)

    assert result == Answer(value=3)
    assert completions.calls[0]["response_format"]["type"] == "json_schema"


def test_defaults_to_openai_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://provider.example/v1")

    client = StructuredClient(model="test")

    assert client.api_key == "env-key"
    assert client.base_url == "https://provider.example/v1"


def test_explicit_provider_config_overrides_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://provider.example/v1")

    client = StructuredClient(
        model="test",
        api_key="explicit-key",
        base_url="https://explicit.example/v1",
    )

    assert client.api_key == "explicit-key"
    assert client.base_url == "https://explicit.example/v1"


def test_auto_falls_back_to_prompt_mode() -> None:
    completions = FakeChatCompletions(['{"value": 4}'], fail_native=True)
    client = StructuredClient(model="test", openai_client=FakeOpenAI(completions), mode="auto")

    result = client.run("return a value", Answer)

    assert result == Answer(value=4)
    assert "response_format" in completions.calls[0]
    assert "response_format" not in completions.calls[1]
    assert "Answer only with JSON" in completions.calls[1]["messages"][-1]["content"]


def test_validation_retry_uses_prompt() -> None:
    completions = FakeChatCompletions(['{"value": "bad"}', '{"value": 5}'])
    client = StructuredClient(model="test", openai_client=FakeOpenAI(completions), max_retries=1)

    result = client.run("return a value", Answer)

    assert result == Answer(value=5)
    assert len(completions.calls) == 2
    assert "Validation error" in completions.calls[1]["messages"][-1]["content"]


@pytest.mark.asyncio
async def test_async_call_returns_model() -> None:
    completions = FakeChatCompletions(['{"value": 6}'])
    client = StructuredClient(model="test", async_openai_client=FakeOpenAI(completions))

    result = await client.arun("return a value", Answer)

    assert result == Answer(value=6)
