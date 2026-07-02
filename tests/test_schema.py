from enum import Enum
from typing import Literal

from pydantic import BaseModel

from structured_llm.schema import build_schema_spec


class Venue(str, Enum):
    BARISA = "barisa"
    OX = "ox_burger"


class ReceiptItem(BaseModel):
    name: str
    quantity: int


class Receipt(BaseModel):
    items: list[ReceiptItem]
    venue: Venue
    reason: Literal["curiosity", "personal_finance"]
    note: str | None = None


def test_build_schema_spec() -> None:
    spec = build_schema_spec(Receipt)
    assert spec.name == "Receipt"
    assert spec.json_schema["type"] == "object"
    assert spec.native_json_schema["additionalProperties"] is False
    assert set(spec.native_json_schema["required"]) == {"items", "venue", "reason", "note"}
    assert "items" in spec.prompt_schema
    assert '"curiosity" or "personal_finance"' in spec.prompt_schema
    assert "note:" in spec.prompt_schema
