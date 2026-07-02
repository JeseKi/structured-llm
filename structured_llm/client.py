from __future__ import annotations

import inspect
import os
from typing import Any, Literal, TypeVar, overload

from .errors import ProviderError, StructuredLLMError, StructuredValidationError
from .parser import parse_structured_text
from .schema import SchemaSpec, build_schema_spec


Mode = Literal["auto", "native", "prompt"]
Endpoint = Literal["chat", "responses"]
T = TypeVar("T")


class StructuredClient:
    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        mode: Mode = "auto",
        endpoint: Endpoint = "chat",
        max_retries: int = 1,
        openai_client: Any | None = None,
        async_openai_client: Any | None = None,
    ) -> None:
        if mode not in {"auto", "native", "prompt"}:
            raise ValueError("mode must be 'auto', 'native', or 'prompt'")
        if endpoint not in {"chat", "responses"}:
            raise ValueError("endpoint must be 'chat' or 'responses'")
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")

        self.model = model
        self.api_key = _value_or_env(api_key, "OPENAI_API_KEY")
        self.base_url = _value_or_env(base_url, "OPENAI_BASE_URL")
        self.mode = mode
        self.endpoint = endpoint
        self.max_retries = max_retries
        self._client = openai_client
        self._async_client = async_openai_client

    @overload
    def run(
        self,
        prompt: str,
        schema: type[T],
        *,
        system: str | None = None,
        **model_options: Any,
    ) -> T: ...

    @overload
    def run(
        self,
        prompt: str,
        schema: Any,
        *,
        system: str | None = None,
        **model_options: Any,
    ) -> Any: ...

    def run(
        self,
        prompt: str,
        schema: Any,
        *,
        system: str | None = None,
        **model_options: Any,
    ) -> Any:
        spec = build_schema_spec(schema)
        raw_text = self._complete(prompt, spec, system=system, **model_options)
        return self._parse_with_retry(raw_text, spec, prompt=prompt, system=system, model_options=model_options)

    @overload
    async def arun(
        self,
        prompt: str,
        schema: type[T],
        *,
        system: str | None = None,
        **model_options: Any,
    ) -> T: ...

    @overload
    async def arun(
        self,
        prompt: str,
        schema: Any,
        *,
        system: str | None = None,
        **model_options: Any,
    ) -> Any: ...

    async def arun(
        self,
        prompt: str,
        schema: Any,
        *,
        system: str | None = None,
        **model_options: Any,
    ) -> Any:
        spec = build_schema_spec(schema)
        raw_text = await self._acomplete(prompt, spec, system=system, **model_options)
        return await self._aparse_with_retry(
            raw_text,
            spec,
            prompt=prompt,
            system=system,
            model_options=model_options,
        )

    @overload
    def parse(self, text: str, schema: type[T], *, repair: bool = True) -> T: ...

    @overload
    def parse(self, text: str, schema: Any, *, repair: bool = True) -> Any: ...

    def parse(self, text: str, schema: Any, *, repair: bool = True) -> Any:
        return parse_structured_text(text, schema, repair=repair)

    def _complete(
        self,
        prompt: str,
        spec: SchemaSpec,
        *,
        system: str | None,
        **model_options: Any,
    ) -> str:
        if self.mode == "prompt":
            return self._call_prompt(prompt, spec, system=system, **model_options)
        if self.mode == "native":
            return self._call_native(prompt, spec, system=system, **model_options)
        try:
            return self._call_native(prompt, spec, system=system, **model_options)
        except ProviderError as exc:
            if not _looks_like_unsupported_native(exc):
                raise
            return self._call_prompt(prompt, spec, system=system, **model_options)

    async def _acomplete(
        self,
        prompt: str,
        spec: SchemaSpec,
        *,
        system: str | None,
        **model_options: Any,
    ) -> str:
        if self.mode == "prompt":
            return await self._acall_prompt(prompt, spec, system=system, **model_options)
        if self.mode == "native":
            return await self._acall_native(prompt, spec, system=system, **model_options)
        try:
            return await self._acall_native(prompt, spec, system=system, **model_options)
        except ProviderError as exc:
            if not _looks_like_unsupported_native(exc):
                raise
            return await self._acall_prompt(prompt, spec, system=system, **model_options)

    def _parse_with_retry(
        self,
        raw_text: str,
        spec: SchemaSpec,
        *,
        prompt: str,
        system: str | None,
        model_options: dict[str, Any],
    ) -> Any:
        try:
            return parse_structured_text(raw_text, spec)
        except StructuredValidationError as exc:
            if self.max_retries <= 0:
                raise
            retry_prompt = _retry_prompt(prompt, spec, exc)
            retry_text = self._call_prompt(retry_prompt, spec, system=system, **model_options)
            return parse_structured_text(retry_text, spec)

    async def _aparse_with_retry(
        self,
        raw_text: str,
        spec: SchemaSpec,
        *,
        prompt: str,
        system: str | None,
        model_options: dict[str, Any],
    ) -> Any:
        try:
            return parse_structured_text(raw_text, spec)
        except StructuredValidationError as exc:
            if self.max_retries <= 0:
                raise
            retry_prompt = _retry_prompt(prompt, spec, exc)
            retry_text = await self._acall_prompt(retry_prompt, spec, system=system, **model_options)
            return parse_structured_text(retry_text, spec)

    def _call_native(self, prompt: str, spec: SchemaSpec, *, system: str | None, **model_options: Any) -> str:
        client = self._sync_client()
        try:
            response = self._invoke_sync(client, prompt, spec, system=system, native=True, **model_options)
            return _extract_response_text(response)
        except StructuredLLMError:
            raise
        except Exception as exc:
            raise ProviderError("Provider native structured output call failed", cause=exc) from exc

    def _call_prompt(self, prompt: str, spec: SchemaSpec, *, system: str | None, **model_options: Any) -> str:
        client = self._sync_client()
        prompt = _prompt_with_schema(prompt, spec)
        try:
            response = self._invoke_sync(client, prompt, spec, system=system, native=False, **model_options)
            return _extract_response_text(response)
        except StructuredLLMError:
            raise
        except Exception as exc:
            raise ProviderError("Provider prompt structured output call failed", cause=exc) from exc

    async def _acall_native(self, prompt: str, spec: SchemaSpec, *, system: str | None, **model_options: Any) -> str:
        client = self._async_openai_client()
        try:
            response = await self._invoke_async(client, prompt, spec, system=system, native=True, **model_options)
            return _extract_response_text(response)
        except StructuredLLMError:
            raise
        except Exception as exc:
            raise ProviderError("Provider native structured output call failed", cause=exc) from exc

    async def _acall_prompt(self, prompt: str, spec: SchemaSpec, *, system: str | None, **model_options: Any) -> str:
        client = self._async_openai_client()
        prompt = _prompt_with_schema(prompt, spec)
        try:
            response = await self._invoke_async(client, prompt, spec, system=system, native=False, **model_options)
            return _extract_response_text(response)
        except StructuredLLMError:
            raise
        except Exception as exc:
            raise ProviderError("Provider prompt structured output call failed", cause=exc) from exc

    def _invoke_sync(
        self,
        client: Any,
        prompt: str,
        spec: SchemaSpec,
        *,
        system: str | None,
        native: bool,
        **model_options: Any,
    ) -> Any:
        if self.endpoint == "responses":
            kwargs = self._responses_kwargs(prompt, spec, system=system, native=native, **model_options)
            return client.responses.create(**kwargs)
        kwargs = self._chat_kwargs(prompt, spec, system=system, native=native, **model_options)
        return client.chat.completions.create(**kwargs)

    async def _invoke_async(
        self,
        client: Any,
        prompt: str,
        spec: SchemaSpec,
        *,
        system: str | None,
        native: bool,
        **model_options: Any,
    ) -> Any:
        if self.endpoint == "responses":
            kwargs = self._responses_kwargs(prompt, spec, system=system, native=native, **model_options)
            return await _maybe_await(client.responses.create(**kwargs))
        kwargs = self._chat_kwargs(prompt, spec, system=system, native=native, **model_options)
        return await _maybe_await(client.chat.completions.create(**kwargs))

    def _chat_kwargs(
        self,
        prompt: str,
        spec: SchemaSpec,
        *,
        system: str | None,
        native: bool,
        **model_options: Any,
    ) -> dict[str, Any]:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        kwargs = {"model": self.model, "messages": messages, **model_options}
        if native:
            kwargs["response_format"] = _chat_response_format(spec)
        return kwargs

    def _responses_kwargs(
        self,
        prompt: str,
        spec: SchemaSpec,
        *,
        system: str | None,
        native: bool,
        **model_options: Any,
    ) -> dict[str, Any]:
        input_text = prompt if system is None else f"{system}\n\n{prompt}"
        kwargs = {"model": self.model, "input": input_text, **model_options}
        if native:
            kwargs["text"] = {"format": _responses_text_format(spec)}
        return kwargs

    def _sync_client(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - dependency is declared
                raise ProviderError("openai package is required for provider calls", cause=exc) from exc
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def _async_openai_client(self) -> Any:
        if self._async_client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:  # pragma: no cover - dependency is declared
                raise ProviderError("openai package is required for provider calls", cause=exc) from exc
            self._async_client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._async_client


def _chat_response_format(spec: SchemaSpec) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": spec.name,
            "strict": True,
            "schema": spec.native_json_schema,
        },
    }


def _responses_text_format(spec: SchemaSpec) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "name": spec.name,
        "strict": True,
        "schema": spec.native_json_schema,
    }


def _prompt_with_schema(prompt: str, spec: SchemaSpec) -> str:
    return (
        f"{prompt.rstrip()}\n\n"
        "Answer only with JSON matching this schema:\n"
        f"{spec.prompt_schema}"
    )


def _retry_prompt(prompt: str, spec: SchemaSpec, exc: StructuredValidationError) -> str:
    return (
        f"{prompt.rstrip()}\n\n"
        "The previous response did not match the required JSON schema.\n"
        f"Validation error:\n{exc.validation_error}\n\n"
        "Return corrected JSON only, matching this schema:\n"
        f"{spec.prompt_schema}"
    )


def _looks_like_unsupported_native(exc: ProviderError) -> bool:
    text = f"{exc} {exc.cause}".lower()
    markers = [
        "response_format",
        "json_schema",
        "unsupported",
        "not supported",
        "unknown parameter",
        "invalid parameter",
        "extra fields not permitted",
    ]
    return any(marker in text for marker in markers)


def _value_or_env(value: str | None, env_name: str) -> str | None:
    if value is not None:
        return value
    return os.environ.get(env_name)


def _extract_response_text(response: Any) -> str:
    if isinstance(response, str):
        return response
    output_text = _get_attr_or_item(response, "output_text")
    if isinstance(output_text, str):
        return output_text

    choices = _get_attr_or_item(response, "choices")
    if choices:
        message = _get_attr_or_item(choices[0], "message")
        content = _get_attr_or_item(message, "content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )

    output = _get_attr_or_item(response, "output")
    if output:
        texts: list[str] = []
        for item in output:
            content = _get_attr_or_item(item, "content")
            if not content:
                continue
            for part in content:
                text = _get_attr_or_item(part, "text")
                if isinstance(text, str):
                    texts.append(text)
        if texts:
            return "".join(texts)

    raise ProviderError("Provider response did not contain text output")


def _get_attr_or_item(obj: Any, key: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value
