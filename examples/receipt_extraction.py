from dotenv import load_dotenv
from pydantic import BaseModel

from structured_llm import StructuredClient

OPENAI_MODEL = "gpt-5.4-mini"

class ReceiptItem(BaseModel):
    name: str
    quantity: int
    price: float


class Receipt(BaseModel):
    merchant: str
    items: list[ReceiptItem]
    total: float


load_dotenv()

client = StructuredClient(model=OPENAI_MODEL)

receipt: Receipt = client.run(
    prompt="提取收据信息: 咖啡店, 2杯卡布奇诺, 每杯4.50, 总计9块。",
    schema=Receipt,
)
print(receipt)
print("---")
print(receipt.model_dump_json(indent=4))