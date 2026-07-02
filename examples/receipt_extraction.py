from dotenv import load_dotenv
from pydantic import BaseModel, Field

from structured_llm import StructuredClient

OPENAI_MODEL = "gpt-5.4-mini"


class ReceiptItem(BaseModel):
    name: str = Field(description="购买的商品名称")
    quantity: int = Field(description="该商品的购买数量")
    price: float = Field(description="该商品的单价")


class Receipt(BaseModel):
    merchant: str = Field(description="商户或店铺名称")
    items: list[ReceiptItem] = Field(description="收据中的商品明细")
    total: float = Field(description="收据最终支付总金额")


load_dotenv()

client = StructuredClient(model=OPENAI_MODEL, debug=True)

receipt: Receipt = client.run(
    prompt="提取收据信息: 咖啡店, 2杯卡布奇诺, 每杯4.50, 总计9块。",
    schema=Receipt,
)
print(receipt)
print("---")
print(receipt.model_dump_json(indent=4))