from __future__ import annotations

import json
import re
from typing import Any, TypeVar, overload

from pydantic import ValidationError

from .errors import StructuredParseError, StructuredValidationError
from .schema import SchemaSpec, build_schema_spec


_FENCED_JSON_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")
T = TypeVar("T")


@overload
def parse_structured_text(
    text: str,
    schema: type[T],
    *,
    repair: bool = True,
    schema_name: str | None = None,
) -> T: ...


@overload
def parse_structured_text(
    text: str,
    schema: Any,
    *,
    repair: bool = True,
    schema_name: str | None = None,
) -> Any: ...


def parse_structured_text(
    text: str,
    schema: Any,
    *,
    repair: bool = True,
    schema_name: str | None = None,
) -> Any:
    spec = schema if isinstance(schema, SchemaSpec) else build_schema_spec(schema, name=schema_name)
    value = _extract_json_value(text, repair=repair, schema_name=spec.name)
    try:
        return spec.adapter.validate_python(value)
    except ValidationError as exc:
        raise StructuredValidationError(
            f"Response did not match schema {spec.name}",
            raw_text=text,
            schema_name=spec.name,
            validation_error=exc,
        ) from exc


def _extract_json_value(text: str, *, repair: bool, schema_name: str) -> Any:
    candidates = _candidate_json_strings(text)
    last_error: BaseException | None = None

    for candidate in candidates:
        for variant in _repair_variants(candidate) if repair else (candidate,):
            try:
                return json.loads(variant)
            except json.JSONDecodeError as exc:
                last_error = exc

    raise StructuredParseError(
        f"No JSON value could be extracted for schema {schema_name}",
        raw_text=text,
        schema_name=schema_name,
        cause=last_error,
    )


def _candidate_json_strings(text: str) -> list[str]:
    stripped = text.strip()
    candidates: list[str] = []
    if stripped:
        candidates.append(stripped)

    for match in _FENCED_JSON_RE.finditer(text):
        block = match.group(1).strip()
        if block:
            candidates.append(block)

    extracted = _first_balanced_json(text)
    if extracted and extracted not in candidates:
        candidates.append(extracted)

    return candidates


def _repair_variants(candidate: str) -> tuple[str, ...]:
    stripped = candidate.strip()
    no_trailing_commas = _TRAILING_COMMA_RE.sub(r"\1", stripped)
    variants = [stripped]
    if no_trailing_commas != stripped:
        variants.append(no_trailing_commas)
    return tuple(dict.fromkeys(variants))


def _first_balanced_json(text: str) -> str | None:
    for start, char in enumerate(text):
        if char not in "{[":
            continue
        result = _balanced_from(text, start)
        if result is not None:
            return result
    return None


def _balanced_from(text: str, start: int) -> str | None:
    opener = text[start]
    closer = "}" if opener == "{" else "]"
    stack = [closer]
    in_string = False
    escape = False

    for idx in range(start + 1, len(text)):
        char = text[idx]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append("}" if char == "{" else "]")
        elif char in "}]":
            if not stack or char != stack[-1]:
                return None
            stack.pop()
            if not stack:
                return text[start : idx + 1]

    return None
