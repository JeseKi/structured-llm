# structured-llm

一个轻量的 Python 结构化输出运行时。

它直接使用 Pydantic 类型作为 schema，不需要 `.baml` 文件、CLI、代码生成，也不需要额外的运行时编译器。默认行为是先尝试 OpenAI 原生 JSON Schema structured outputs；如果 OpenAI-compatible 服务不支持该请求格式，则降级为「schema prompt + 本地 JSON 提取/轻量修复 + Pydantic 校验」。

## 安装依赖

普通运行依赖：

```bash
uv sync --no-config --default-index https://pypi.org/simple
```

开发依赖安装在 `dev` dependency group 中，包括 `pytest`、`pytest-asyncio`、`ruff`、`mypy`：

```bash
uv sync --group dev --no-config --default-index https://pypi.org/simple
```

如果需要新增开发工具依赖：

```bash
uv add <package> --group dev --no-config --default-index https://pypi.org/simple
```

## 使用示例

```python
from pydantic import BaseModel
from structured_llm import StructuredClient


class Receipt(BaseModel):
    merchant: str
    total: float


client = StructuredClient(model="gpt-4o-mini")
receipt = client.run("Extract the receipt: Coffee $4.50", Receipt)

print(receipt.merchant)
print(receipt.total)
```

默认 OpenAI-compatible provider 会从 `OPENAI_API_KEY` 和 `OPENAI_BASE_URL` 读取配置；如果代码里显式传入 `api_key` 或 `base_url`，会优先使用显式参数。`examples/receipt_extraction.py` 会通过 `python-dotenv` 自动加载本地 `.env`。

只解析已有的 LLM 文本输出：

```python
raw = """
```json
{"merchant": "Coffee Shop", "total": 4.5}
```
"""

receipt = client.parse(raw, Receipt)
```

## 开发命令

运行测试：

```bash
PYTHONDONTWRITEBYTECODE=1 uv run --no-config --default-index https://pypi.org/simple --group dev python -m pytest -p no:cacheprovider
```

运行 Ruff：

```bash
uv run --no-config --default-index https://pypi.org/simple --group dev ruff check .
```

运行 mypy：

```bash
uv run --no-config --default-index https://pypi.org/simple --group dev mypy structured_llm
```

## 当前范围

- 支持同步调用：`StructuredClient.run(...)`
- 支持异步调用：`StructuredClient.arun(...)`
- 支持本地解析：`StructuredClient.parse(...)`
- 支持 Pydantic `BaseModel`、`list[...]`、`dict[...]`、`Literal`、`Enum` 等 `TypeAdapter` 可处理的类型
- 暂不支持流式 partial object、多 provider 内置适配、BAML DSL/codegen
