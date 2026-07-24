---
name: structured-llm
description: 在 Python 项目中接入 structured-llm 结构化 LLM 输出库。Use when the user wants to install, configure, demonstrate, or implement text/image structured LLM extraction with Pydantic schemas, OpenAI-compatible providers, `.env` variables, prompt-based JSON output parsing, debug request inspection, or migration away from provider-specific `response_format`.
---

# Structured LLM

## 目标

使用 `structured-llm` 时，把 Pydantic 类型作为结构化输出 schema，调用 OpenAI-compatible 模型，默认通过通用 output-format prompt 约束模型返回 JSON，再用本地解析、轻量修复和 Pydantic 校验得到类型化结果。

优先使用该库的默认 prompt 模式。不要默认依赖 OpenAI 的 `response_format`，因为很多 OpenAI-compatible 供应商不支持它。

## 接入流程

1. 安装依赖：

```bash
uv add structured-llm pydantic python-dotenv
```

如果项目不用 `uv`，使用当前项目的包管理器安装 `structured-llm`、`pydantic`、`python-dotenv`。

2. 配置环境变量：

```env
OPENAI_API_KEY=...
OPENAI_BASE_URL=https://your-openai-compatible-provider/v1
```

`OPENAI_BASE_URL` 可省略，此时 OpenAI SDK 使用自己的默认地址。代码里显式传入 `api_key` 或 `base_url` 时，显式参数优先。

3. 定义 Pydantic schema。字段应使用 `Field(description=...)` 描述含义，因为这些描述会进入 output-format prompt：

```python
from pydantic import BaseModel, Field


class ReceiptItem(BaseModel):
    name: str = Field(description="购买的商品名称")
    quantity: int = Field(description="该商品的购买数量")
    price: float = Field(description="该商品的单价")


class Receipt(BaseModel):
    merchant: str = Field(description="商户或店铺名称")
    items: list[ReceiptItem] = Field(description="收据中的商品明细")
    total: float = Field(description="收据最终支付总金额")
```

4. 调用模型并获取类型化结果：

```python
from dotenv import load_dotenv
from structured_llm import StructuredClient

load_dotenv()

client = StructuredClient(model="gpt-5.4-mini")

receipt = client.run(
    prompt="提取收据信息: 咖啡店, 2杯卡布奇诺, 每杯4.50, 总计9块。",
    schema=Receipt,
)

print(receipt.total)
```

当 `schema` 是 `Receipt` 这类类型对象时，`client.run(..., Receipt)` 的静态返回类型应是 `Receipt`。

## 默认行为

默认构造方式：

```python
client = StructuredClient(model="...")
```

等价于：

```python
client = StructuredClient(model="...", mode="prompt")
```

该模式不会发送 `response_format`。它会把用户 prompt 和 output format 合并到普通 chat/request payload 中，让供应商只需支持基础 OpenAI-compatible chat/responses API。

只有在用户明确要求使用 provider-native structured output，或目标供应商确认支持 OpenAI JSON Schema structured outputs 时，才使用：

```python
StructuredClient(model="...", mode="native")
```

需要“先试 native，不支持再回退 prompt”时，才使用：

```python
StructuredClient(model="...", mode="auto")
```

## 图片输入

当任务需要从图片中提取结构化信息时，使用 `ImageInput`，并确认模型和 OpenAI-compatible provider 都支持视觉输入。`run` 与 `arun` 通过 `images` 接收一个或多个图片；图片会随请求计入输入 token。

```python
from pydantic import BaseModel, Field
from structured_llm import ImageInput, StructuredClient


class ImageExtraction(BaseModel):
    summary: str = Field(description="图片主体的客观摘要")
    visible_text: list[str] = Field(description="可辨认文字，按阅读顺序列出")
    key_values: dict[str, str] = Field(description="图片中明确标注的键值信息")


client = StructuredClient(model="gpt-5.4-mini", mode="auto")
result = client.run(
    prompt="提取图片中的关键信息；只输出可见内容，不要猜测。",
    schema=ImageExtraction,
    images=[ImageInput.from_file("document.png", detail="high")],
)
```

图片来源必须显式构造：

```python
ImageInput.from_url("https://example.com/image.png")
ImageInput.from_file("./image.jpg")
ImageInput.from_data_url("data:image/png;base64,iVBORw0KGgo...")
```

`detail` 可为 `"auto"`（默认）、`"low"`、`"high"` 或 `"original"`。本地文件支持 PNG、JPEG、WEBP、GIF，并会编码为 Data URL；不要传裸 Base64，因为它没有 MIME 类型。Chat 与 Responses 端点都支持该统一 API。

## Debug

调试模型输入输出时启用：

```python
client = StructuredClient(model="...", debug=True)
```

debug 会向 `stderr` 打印：

- 传给 OpenAI-compatible SDK 的 request payload，包括 `model`、`messages` 或 `input`、以及模型参数。
- 模型解析前的原始输出。

默认 prompt 模式下，debug 输出不应包含 `response_format`。如果看到 `response_format`，说明代码显式使用了 `mode="native"` 或 `mode="auto"`。

图片请求中的 Data URL 会在 debug 输出中脱敏，不会打印其 Base64 主体。

## 异步调用

在 async 项目中使用 `arun`：

```python
receipt = await client.arun(
    prompt="提取收据信息: ...",
    schema=Receipt,
)
```

## 只解析已有模型输出

当项目已经从其他 LLM 调用拿到了文本，只需要解析和校验时：

```python
from structured_llm import parse_structured_text

raw = """
```json
{"merchant": "Coffee Shop", "items": [], "total": 9.0}
```
"""

receipt = parse_structured_text(raw, Receipt)
```

或者：

```python
receipt = client.parse(raw, Receipt)
```

解析器会尝试从普通文本、Markdown JSON fence、嵌入文本中的首个平衡 JSON 对象或数组中提取 JSON，并修复简单尾逗号，再交给 Pydantic 校验。

## 项目落地建议

- 把 schema 放在业务模块附近，例如 `app/extraction/schemas.py`。
- 把 LLM 调用封装成小函数，例如 `extract_receipt(text: str) -> Receipt`，不要在业务流程里散落 prompt 和 schema。
- 在测试中优先覆盖解析层：用 `client.parse(...)` 或 `parse_structured_text(...)` 验证模型可能返回的 JSON 文本。
- 开发期使用 `debug=True`；生产期默认关闭，避免日志包含用户输入或模型输出。
- 对供应商兼容性不确定时，坚持默认 `mode="prompt"`。
- 对视觉任务，先验证所选模型和 provider 支持图片；provider-native JSON Schema 可用时优先 `mode="auto"`，以便不支持时自动回退 prompt 模式。
- 使用 `Field(description=...)` 描述字段语义，尤其是枚举、金额、日期、单位、可选字段。

## 常见错误

- 不要要求默认使用 `response_format`；它不是通用 OpenAI-compatible 能力。
- 不要把 schema 写成普通 dict 后期待静态类型推导。若需要 `receipt` 被推导为 `Receipt`，传入 `Receipt` 类型对象。
- 不要让模型输出解释、Markdown、代码块或额外文本。库的 prompt 会要求只返回 JSON，但业务 prompt 也应避免鼓励解释。
- 如果解析失败，先查看 debug raw output，再判断是模型没有返回 JSON、字段语义不清、还是 schema 过严。
- 若图片请求的 raw output 为空，先排查模型/provider 的视觉支持、图片 URL 是否可由 provider 服务端访问、Data URL 是否完整，以及 provider 是否返回了拒绝或过滤信息；这不是本地 JSON 解析器能够修复的问题。
