from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, Field

from structured_llm import ImageInput, StructuredClient


class Answer(BaseModel):
    value: int


class DescribedAnswer(BaseModel):
    value: int = Field(description="The numeric answer to return")


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


@dataclass
class FakeResponsesResponse:
    output_text: str


class FakeResponses:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> FakeResponsesResponse:
        self.calls.append(kwargs)
        return FakeResponsesResponse(self.responses.pop(0))


class FakeOpenAI:
    def __init__(
        self,
        completions: FakeChatCompletions | None = None,
        responses: FakeResponses | None = None,
    ) -> None:
        if completions is not None:
            self.chat = FakeChat(completions)
        if responses is not None:
            self.responses = responses


def test_native_call_returns_model() -> None:
    completions = FakeChatCompletions(['{"value": 3}'])
    client = StructuredClient(
        model="test", openai_client=FakeOpenAI(completions), mode="native"
    )

    result = client.run("return a value", Answer)

    assert result == Answer(value=3)
    assert completions.calls[0]["response_format"]["type"] == "json_schema"


def test_default_call_uses_prompt_mode_without_response_format() -> None:
    completions = FakeChatCompletions(['{"value": 3}'])
    client = StructuredClient(model="test", openai_client=FakeOpenAI(completions))

    result = client.run("return a value", DescribedAnswer)

    assert result == DescribedAnswer(value=3)
    assert "response_format" not in completions.calls[0]
    assert "Return a JSON value" in completions.calls[0]["messages"][-1]["content"]
    assert (
        "The numeric answer to return"
        in completions.calls[0]["messages"][-1]["content"]
    )


def test_defaults_to_openai_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://provider.example/v1")

    client = StructuredClient(model="test")

    assert client.api_key == "env-key"
    assert client.base_url == "https://provider.example/v1"


def test_explicit_provider_config_overrides_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://provider.example/v1")

    client = StructuredClient(
        model="test",
        api_key="explicit-key",
        base_url="https://explicit.example/v1",
    )

    assert client.api_key == "explicit-key"
    assert client.base_url == "https://explicit.example/v1"


def test_debug_prints_request_context_and_raw_output(
    capsys: pytest.CaptureFixture[str],
) -> None:
    completions = FakeChatCompletions(['{"value": 7}'])
    client = StructuredClient(
        model="test",
        openai_client=FakeOpenAI(completions),
        mode="prompt",
        debug=True,
    )

    result = client.run("return a value", DescribedAnswer)

    captured = capsys.readouterr()
    assert result == DescribedAnswer(value=7)
    assert captured.out == ""
    assert "[structured-llm debug] request chat.completions.create" in captured.err
    assert '"model": "test"' in captured.err
    assert '"messages": [' in captured.err
    assert "response_format" not in captured.err
    assert "The numeric answer to return" in captured.err
    assert "[structured-llm debug] raw output" in captured.err
    assert '{"value": 7}' in captured.err


def test_auto_falls_back_to_prompt_mode() -> None:
    completions = FakeChatCompletions(['{"value": 4}'], fail_native=True)
    client = StructuredClient(
        model="test", openai_client=FakeOpenAI(completions), mode="auto"
    )

    result = client.run("return a value", Answer)

    assert result == Answer(value=4)
    assert "response_format" in completions.calls[0]
    assert "response_format" not in completions.calls[1]
    assert "Return a JSON value" in completions.calls[1]["messages"][-1]["content"]


def test_validation_retry_uses_prompt() -> None:
    completions = FakeChatCompletions(['{"value": "bad"}', '{"value": 5}'])
    client = StructuredClient(
        model="test", openai_client=FakeOpenAI(completions), max_retries=1
    )

    result = client.run("return a value", Answer)

    assert result == Answer(value=5)
    assert len(completions.calls) == 2
    assert "Validation error" in completions.calls[1]["messages"][-1]["content"]


def test_chat_call_sends_images_as_content_parts() -> None:
    completions = FakeChatCompletions(['{"value": 8}'])
    client = StructuredClient(model="test", openai_client=FakeOpenAI(completions))

    result = client.run(
        "read these images",
        Answer,
        images=[
            ImageInput.from_url("https://example.com/first.png", detail="low"),
            ImageInput.from_data_url("data:image/png;base64,aGVsbG8=", detail="high"),
        ],
    )

    content = completions.calls[0]["messages"][-1]["content"]
    assert result == Answer(value=8)
    assert content[0]["type"] == "text"
    assert content[1] == {
        "type": "image_url",
        "image_url": {"url": "https://example.com/first.png", "detail": "low"},
    }
    assert content[2] == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,aGVsbG8=", "detail": "high"},
    }


def test_responses_call_sends_images_as_input_parts() -> None:
    responses = FakeResponses(['{"value": 9}'])
    client = StructuredClient(
        model="test",
        endpoint="responses",
        openai_client=FakeOpenAI(responses=responses),
    )

    result = client.run(
        "read this image",
        Answer,
        system="be precise",
        images=[ImageInput.from_url("https://example.com/receipt.jpg")],
    )

    content = responses.calls[0]["input"][0]["content"]
    assert result == Answer(value=9)
    assert content[0]["type"] == "input_text"
    assert content[0]["text"].startswith("be precise\n\nread this image")
    assert content[1] == {
        "type": "input_image",
        "image_url": "https://example.com/receipt.jpg",
        "detail": "auto",
    }


def test_retry_keeps_images() -> None:
    completions = FakeChatCompletions(['{"value": "bad"}', '{"value": 10}'])
    client = StructuredClient(
        model="test", openai_client=FakeOpenAI(completions), max_retries=1
    )
    image = ImageInput.from_url("https://example.com/receipt.jpg")

    result = client.run("read this receipt", Answer, images=[image])

    assert result == Answer(value=10)
    first_content = completions.calls[0]["messages"][-1]["content"]
    retry_content = completions.calls[1]["messages"][-1]["content"]
    assert retry_content[1:] == first_content[1:]


def test_image_input_from_file_encodes_supported_image(tmp_path: Path) -> None:
    image_path = tmp_path / "receipt.png"
    image_path.write_bytes(b"receipt bytes")

    image = ImageInput.from_file(image_path)

    assert image.image_url == "data:image/png;base64,cmVjZWlwdCBieXRlcw=="


def test_invalid_image_input_is_rejected_before_provider_call() -> None:
    completions = FakeChatCompletions(['{"value": 1}'])
    client = StructuredClient(model="test", openai_client=FakeOpenAI(completions))

    with pytest.raises(ValueError, match="fully qualified"):
        ImageInput.from_url("receipt.png")
    with pytest.raises(ValueError, match="supported base64"):
        ImageInput("data:image/svg+xml;base64,PHN2Zy8+")
    with pytest.raises(TypeError, match="ImageInput"):
        client.run("read this", Answer, images=["https://example.com/receipt.png"])  # type: ignore[list-item]
    assert completions.calls == []


def test_debug_redacts_data_url(capsys: pytest.CaptureFixture[str]) -> None:
    completions = FakeChatCompletions(['{"value": 11}'])
    client = StructuredClient(
        model="test", openai_client=FakeOpenAI(completions), debug=True
    )

    client.run(
        "read this",
        Answer,
        images=[ImageInput.from_data_url("data:image/png;base64,c2VjcmV0LWltYWdl")],
    )

    captured = capsys.readouterr()
    assert "c2VjcmV0LWltYWdl" not in captured.err
    assert "<redacted 16 base64 chars>" in captured.err


@pytest.mark.asyncio
async def test_async_call_returns_model() -> None:
    completions = FakeChatCompletions(['{"value": 6}'])
    client = StructuredClient(model="test", async_openai_client=FakeOpenAI(completions))

    result = await client.arun(
        "return a value",
        Answer,
        images=[ImageInput.from_url("https://example.com/receipt.jpg")],
    )

    assert result == Answer(value=6)
    assert completions.calls[0]["messages"][-1]["content"][1]["type"] == "image_url"
