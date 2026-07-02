from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

from pydantic import BaseModel, TypeAdapter

from .errors import UnsupportedSchemaError


@dataclass(frozen=True)
class SchemaSpec:
    name: str
    adapter: TypeAdapter[Any]
    json_schema: dict[str, Any]
    native_json_schema: dict[str, Any]
    prompt_schema: str


def build_schema_spec(schema: Any, *, name: str | None = None) -> SchemaSpec:
    try:
        adapter = TypeAdapter(schema)
        json_schema = adapter.json_schema()
    except Exception as exc:  # pragma: no cover - exact pydantic errors vary
        raise UnsupportedSchemaError(f"Unsupported schema: {schema!r}") from exc

    schema_name = _schema_name(schema, name)
    prompt_schema = render_prompt_schema(json_schema)
    return SchemaSpec(
        name=schema_name,
        adapter=adapter,
        json_schema=json_schema,
        native_json_schema=_to_strict_json_schema(json_schema),
        prompt_schema=prompt_schema,
    )


def render_prompt_schema(json_schema: dict[str, Any]) -> str:
    defs = json_schema.get("$defs", {})
    return _render_type(json_schema, defs, set())


def _schema_name(schema: Any, explicit: str | None) -> str:
    if explicit:
        raw = explicit
    elif isinstance(schema, type):
        raw = schema.__name__
    else:
        raw = getattr(schema, "__name__", "structured_output")
    name = re.sub(r"[^a-zA-Z0-9_-]+", "_", raw).strip("_") or "structured_output"
    return name[:64]


def _render_type(schema: dict[str, Any], defs: dict[str, Any], seen: set[str]) -> str:
    if "$ref" in schema:
        ref_name = schema["$ref"].rsplit("/", 1)[-1]
        if ref_name in seen:
            return ref_name
        target = defs.get(ref_name)
        if not isinstance(target, dict):
            return ref_name
        return _render_type(target, defs, {*seen, ref_name})

    if "const" in schema:
        return repr(schema["const"]).replace("'", '"')

    if "enum" in schema:
        return " or ".join(repr(item).replace("'", '"') for item in schema["enum"])

    if "anyOf" in schema:
        return " or ".join(_render_type(item, defs, seen) for item in schema["anyOf"])

    if "oneOf" in schema:
        return " or ".join(_render_type(item, defs, seen) for item in schema["oneOf"])

    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        return " or ".join(_render_type({**schema, "type": item}, defs, seen) for item in schema_type)

    if schema_type == "object" or "properties" in schema:
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))
        if not properties:
            return "object"
        lines = ["{"]
        for key, value in properties.items():
            suffix = "" if key in required else " optional"
            rendered = _render_type(value, defs, seen)
            lines.append(f"  {key}: {rendered}{suffix},")
        lines.append("}")
        return "\n".join(lines)

    if schema_type == "array":
        item_schema = schema.get("items", {})
        return f"[{_render_type(item_schema, defs, seen)}]"

    if schema_type == "string":
        return "string"
    if schema_type == "integer":
        return "int"
    if schema_type == "number":
        return "float"
    if schema_type == "boolean":
        return "bool"
    if schema_type == "null":
        return "null"

    return "any"


def _to_strict_json_schema(schema: Any) -> Any:
    if isinstance(schema, list):
        return [_to_strict_json_schema(item) for item in schema]
    if not isinstance(schema, dict):
        return schema

    strict = {key: _to_strict_json_schema(value) for key, value in schema.items()}

    if "$defs" in strict and isinstance(strict["$defs"], dict):
        strict["$defs"] = {
            key: _to_strict_json_schema(value)
            for key, value in strict["$defs"].items()
        }

    properties = strict.get("properties")
    if isinstance(properties, dict):
        strict["additionalProperties"] = False
        strict["required"] = list(properties.keys())

    return strict


def is_pydantic_model(schema: Any) -> bool:
    return isinstance(schema, type) and issubclass(schema, BaseModel)


def is_enum(schema: Any) -> bool:
    return isinstance(schema, type) and issubclass(schema, Enum)
