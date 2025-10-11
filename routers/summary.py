# 新增：总结接口

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Literal
from ..models.chat_models import ChatRequest
from ..services.llm_clients import smart_call, DEFAULT_MODEL

router = APIRouter(prefix="/api")

class Record(BaseModel):
    role: Literal["system", "user", "assistant", "tool"] = "user"
    content: str

class SummaryReq(BaseModel):
    model: Optional[str] = None
    records: List[Record] = []       # 传入需要被总结的对话（按时间正序）
    prompt: Optional[str] = None     # 可选：业务自定义提示词
    style: Literal["brief","bullet","action","daily"] = "brief"
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.3

class SummaryResp(BaseModel):
    summary: str

SYSTEM_SUMMARY = (
    "你是一个严谨的会议与对话总结助手。请基于提供的历史对话，输出高质量的中文摘要。"
    "保持客观、去除寒暄，突出要点、结论与后续行动。"
)

STYLE_TPL = {
    "brief": "以不超过 8 行的精炼段落概括关键信息与结论。",
    "bullet": "使用有序要点输出：1) 关键结论 2) 证据/细节 3) 风险或分歧 4) 待办与责任人。",
    "action": "仅输出行动清单（谁在何时做什么，成功判定标准）。",
    "daily":  "以日报格式输出：今日进展/问题/明日计划/需协助。",
}

@router.post("/summary", response_model=SummaryResp)
async def summarize(body: SummaryReq):
    if not body.records:
        raise HTTPException(status_code=400, detail="records 不能为空")

    # 组织 messages
    style_note = STYLE_TPL.get(body.style, STYLE_TPL["brief"])
    user_prompt = body.prompt or "请对以下对话进行总结。"
    messages = [{"role":"system","content": SYSTEM_SUMMARY + style_note},
                {"role":"user","content": user_prompt}]

    # 将历史对话拼入
    for r in body.records:
        messages.append({"role": r.role, "content": r.content})

    req = ChatRequest(
        model=body.model or DEFAULT_MODEL,
        messages=[type("M", (), m) for m in messages],  # 轻量把 dict 适配成有属性的对象
        temperature=body.temperature,
        max_completion_tokens=body.max_tokens
    )

    text = await smart_call(req)
    return SummaryResp(summary=text.strip())