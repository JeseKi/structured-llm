from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse


ImageDetail = Literal["auto", "low", "high", "original"]

_DETAILS = {"auto", "low", "high", "original"}
_MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}
_SUPPORTED_MIME_TYPES = frozenset(_MIME_TYPES.values())


@dataclass(frozen=True)
class ImageInput:
    """An image supplied to a structured LLM request."""

    image_url: str
    detail: ImageDetail = "auto"

    def __post_init__(self) -> None:
        if not isinstance(self.image_url, str) or not self.image_url:
            raise ValueError("image_url must be a non-empty string")
        if self.detail not in _DETAILS:
            raise ValueError("detail must be 'auto', 'low', 'high', or 'original'")
        if self.image_url.startswith("data:"):
            _validate_data_url(self.image_url)
        else:
            _validate_http_url(self.image_url)

    @classmethod
    def from_url(cls, url: str, *, detail: ImageDetail = "auto") -> ImageInput:
        return cls(url, detail)

    @classmethod
    def from_data_url(
        cls, data_url: str, *, detail: ImageDetail = "auto"
    ) -> ImageInput:
        return cls(data_url, detail)

    @classmethod
    def from_file(cls, path: str | Path, *, detail: ImageDetail = "auto") -> ImageInput:
        image_path = Path(path)
        if not image_path.is_file():
            raise ValueError(f"image path is not a file: {image_path}")
        mime_type = _MIME_TYPES.get(image_path.suffix.lower())
        if mime_type is None:
            supported = ", ".join(sorted(_MIME_TYPES))
            raise ValueError(
                f"unsupported image file type; supported extensions: {supported}"
            )
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        return cls.from_data_url(f"data:{mime_type};base64,{encoded}", detail=detail)


def _validate_http_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("image URL must be a fully qualified http(s) URL")


def _validate_data_url(data_url: str) -> None:
    header, separator, encoded = data_url.partition(",")
    mime_type, *parameters = header[5:].split(";")
    if (
        not separator
        or mime_type not in _SUPPORTED_MIME_TYPES
        or "base64" not in parameters
        or not encoded
    ):
        raise ValueError("data URL must contain a supported base64-encoded image")
    try:
        base64.b64decode(encoded, validate=True)
    except ValueError as exc:
        raise ValueError("data URL contains invalid base64 image data") from exc
