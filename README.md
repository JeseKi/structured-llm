# structured-llm

一个轻量的 Python 结构化输出运行时。

它直接使用 Pydantic 类型作为 schema，不需要 `.baml` 文件、CLI、代码生成，也不需要额外的运行时编译器。默认行为是 BAML 风格的「output format prompt + 本地 JSON 提取/轻量修复 + Pydantic 校验」，因此不依赖特定供应商是否支持 `response_format`。

## 安装

环境要求：Python `>=3.10`（本仓库开发用 `3.13`）。

### 1. 在你自己的项目中使用（作为第三方依赖）

本包已发布为 `structured-llm`，安装时会自动带上运行时依赖 `openai>=1.0.0`、`pydantic>=2.0.0`、`python-dotenv>=1.2.2`，不需要手动逐个安装。

用 `pip`：

```bash
pip install structured-llm
```

用 `uv`（推荐，在已有项目目录下执行）：

```bash
uv add structured-llm
```

验证是否安装成功：

```bash
python -c "from importlib.metadata import version; print(version('structured-llm'))"
```

装好后配置 OpenAI-compatible 的 Key 即可使用（详见下面的使用示例）：

```bash
export OPENAI_API_KEY="..."
# 可选，默认走 OpenAI 官方地址；第三方兼容服务才需要改
export OPENAI_BASE_URL="https://your-openai-compatible-provider/v1"
```

### 2. 克隆本仓库做二次开发

只有你要改 `structured-llm` 本身的代码、跑测试、提 PR 时，才需要这一步。普通使用者看第 1 节就够了。

```bash
git clone https://github.com/JeseKi/structured-llm.git
cd structured-llm
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

> 说明：上面命令里的 `--no-config --default-index https://pypi.org/simple` 是为了强制走 PyPI 官方源、忽略本地可能配置的镜像。如果你本地没有换源需求，直接用 `uv sync` / `uv sync --group dev` 即可。

## 使用示例

```python
from pydantic import BaseModel, Field
from structured_llm import StructuredClient


class Receipt(BaseModel):
    merchant: str = Field(description="商户或店铺名称")
    total: float = Field(description="收据最终支付总金额")


client = StructuredClient(model="gpt-4o-mini", debug=True)
receipt = client.run("Extract the receipt: Coffee $4.50", Receipt)

print(receipt.merchant)
print(receipt.total)
```

默认 OpenAI-compatible provider 会从 `OPENAI_API_KEY` 和 `OPENAI_BASE_URL` 读取配置；如果代码里显式传入 `api_key` 或 `base_url`，会优先使用显式参数。`examples/receipt_extraction.py` 会通过 `python-dotenv` 自动加载本地 `.env`。

`Field(description=...)` 会渲染到默认 output format prompt。`debug=True` 会把传给 OpenAI-compatible SDK 的 request payload 和模型解析前的原始输出打印到 stderr。默认不会发送 `response_format`；只有显式设置 `mode="native"` 或 `mode="auto"` 时才会尝试 provider-native structured output。

### 图片输入

视觉模型可与结构化输出一起使用。通过 `ImageInput` 显式提供远程 URL、本地图片或 Base64 Data URL；`run()` 和 `arun()` 都支持多个图片，且同时兼容 `endpoint="chat"` 和 `endpoint="responses"`：

```python
from structured_llm import ImageInput, StructuredClient

client = StructuredClient(model="gpt-4o-mini")
receipt = client.run(
    "提取这张收据中的商户和总金额",
    Receipt,
    images=[ImageInput.from_file("receipt.jpg", detail="high")],
)
```

可用的 `detail` 为 `"auto"`（默认）、`"low"`、`"high"` 和 `"original"`。本地文件支持 PNG、JPEG、WEBP、GIF，并会在内存中转为 Data URL；远程图片请使用 `ImageInput.from_url(...)`，已有 Data URL 请使用 `ImageInput.from_data_url(...)`。模型必须支持视觉输入，图片会计入输入 token。为避免日志泄露二进制图片内容，`debug=True` 会脱敏 Data URL 的 Base64 主体。

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
- 支持图片输入：URL、本地图片、Base64 Data URL
- 支持本地解析：`StructuredClient.parse(...)`
- 支持 Pydantic `BaseModel`、`list[...]`、`dict[...]`、`Literal`、`Enum` 等 `TypeAdapter` 可处理的类型
- 暂不支持流式 partial object、多 provider 内置适配、BAML DSL/codegen
