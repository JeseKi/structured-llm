import argparse
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from structured_llm import ImageInput, StructuredClient

OPENAI_MODEL = "gpt-5.6-luna"


class ImageExtraction(BaseModel):
    summary: str = Field(description="图片主体内容的简短客观摘要")
    visible_text: list[str] = Field(description="图片中可辨认的文字，按阅读顺序列出")
    objects: list[str] = Field(description="主要对象、人物、场景或视觉元素")
    key_values: dict[str, str] = Field(
        description="图片中明确标注的关键字段和值，例如价格、日期、名称、状态或编号"
    )
    notes: list[str] = Field(description="不确定、模糊、遮挡或需要人工确认的信息")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从文本或任意图片中提取结构化关键信息")
    parser.add_argument(
        "--image",
        help="图片 URL、本地路径或 data:image/...;base64,...；提供后启用视觉提取",
    )
    return parser.parse_args()


def image_input(source: str) -> ImageInput:
    if source.startswith("data:"):
        return ImageInput.from_data_url(source, detail="high")
    if urlparse(source).scheme in {"http", "https"}:
        return ImageInput.from_url(source, detail="high")
    return ImageInput.from_file(Path(source), detail="high")


load_dotenv()
args = parse_args()

client = StructuredClient(model=OPENAI_MODEL, debug=True)

prompt = "提取关键信息: 新款无线耳机，售价 299 元，限时优惠至 6 月 30 日。"
images = None
if args.image:
    prompt = (
        "从这张图片中提取通用关键信息，包括主体摘要、可见文字、主要视觉元素和明确标注的键值信息。"
        "仅提取图片中可见或可辨认的内容；不要猜测。将模糊、遮挡或无法确定的内容写入 notes。"
    )
    images = [image_input(args.image)]

extraction: ImageExtraction = client.run(
    prompt=prompt,
    schema=ImageExtraction,
    images=images,
)
print(extraction)
print("---")
print(extraction.model_dump_json(indent=4))
