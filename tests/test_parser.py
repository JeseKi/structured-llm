from typing import Literal

import pytest
from pydantic import BaseModel

from structured_llm import StructuredClient, StructuredParseError, StructuredValidationError


class Receipt(BaseModel):
    merchant: str
    total: float
    venue: Literal["barisa", "ox_burger"]


def test_parse_plain_json() -> None:
    parsed = StructuredClient(model="test").parse(
        '{"merchant": "Barisa", "total": 12.5, "venue": "barisa"}',
        Receipt,
    )
    assert parsed == Receipt(merchant="Barisa", total=12.5, venue="barisa")


def test_parse_markdown_json() -> None:
    parsed = StructuredClient(model="test").parse(
        'Here:\n```json\n{"merchant": "Ox", "total": 9, "venue": "ox_burger"}\n```',
        Receipt,
    )
    assert parsed.venue == "ox_burger"


def test_parse_embedded_json_with_trailing_comma() -> None:
    parsed = StructuredClient(model="test").parse(
        'The answer is {"merchant": "Barisa", "total": 12.5, "venue": "barisa",} thanks',
        Receipt,
    )
    assert parsed.total == 12.5


def test_parse_error_keeps_raw_text() -> None:
    with pytest.raises(StructuredParseError) as exc_info:
        StructuredClient(model="test").parse("no json here", Receipt)
    assert exc_info.value.raw_text == "no json here"


def test_validation_error_keeps_raw_text() -> None:
    raw = '{"merchant": "Barisa", "total": "oops", "venue": "barisa"}'
    with pytest.raises(StructuredValidationError) as exc_info:
        StructuredClient(model="test").parse(raw, Receipt)
    assert exc_info.value.raw_text == raw

