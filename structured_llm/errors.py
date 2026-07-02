from __future__ import annotations

from typing import Any


class StructuredLLMError(Exception):
    """Base exception for structured_llm."""


class UnsupportedSchemaError(StructuredLLMError):
    """Raised when a Python type cannot be converted into a usable schema."""


class ProviderError(StructuredLLMError):
    """Raised when the model provider call fails."""

    def __init__(self, message: str, *, cause: BaseException | None = None) -> None:
        super().__init__(message)
        self.cause = cause


class StructuredParseError(StructuredLLMError):
    """Raised when no JSON value can be extracted from model output."""

    def __init__(
        self,
        message: str,
        *,
        raw_text: str,
        schema_name: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.raw_text = raw_text
        self.schema_name = schema_name
        self.cause = cause


class StructuredValidationError(StructuredLLMError):
    """Raised when extracted JSON does not validate against the target schema."""

    def __init__(
        self,
        message: str,
        *,
        raw_text: str,
        schema_name: str,
        validation_error: Any,
    ) -> None:
        super().__init__(message)
        self.raw_text = raw_text
        self.schema_name = schema_name
        self.validation_error = validation_error

