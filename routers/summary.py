# 新增：总结接口

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Literal
from models.chat_models import ChatRequest
from services.llm_clients import smart_call, DEFAULT_MODEL
from typing import List, Optional, Union, Dict, Any, Literal
from models.record_model import Record,SummaryReq,SummarizeResultResp
import json
import re

router = APIRouter(prefix="/summary")

SYSTEM_SUMMARY = (
    "你是一个严谨的中文总结助手。请基于给定内容生成“每日总结”，并提取 3 个中文情绪关键词。"
    "返回 JSON，且键名必须严格为：article、moodKeywords、model、tokenUsageJson。"
    "moodKeywords 用中文逗号分隔 3 个词，例如：专注, 放松, 感恩。"
)


STYLE_TPL = {
    "brief":  "总结要精炼（不超过 8 行），突出事实与结论。",
    "bullet": "用要点列出：关键结论/证据/风险/待办。",
    "action": "仅输出行动项（负责人/截至时间/成功标准）。",
    "daily":  "以日报格式输出：今日进展/问题/明日计划/需协助。",
}

# ================= 主逻辑 =================
@router.post("/daily", response_model=SummarizeResultResp)
async def summarize(body: SummaryReq):
    if body.type != "daily_summary":
        raise HTTPException(status_code=400, detail="type 必须为 'daily_summary'")
    if not body.text.strip():
        raise HTTPException(status_code=400, detail="text 不能为空")

    # system + user 指令
    system_msg = "你是一个中文总结助手，请只输出 JSON。"
    user_prompt = (
        "请基于以下内容生成“每日总结”并提取 3 个中文情绪关键词：\n"
        "仅返回一个 JSON 对象，键名必须严格为："
        "article、moodKeywords、model、tokenUsageJson。\n"
        "moodKeywords 使用中文逗号分隔，例如：专注, 放松, 感恩。\n\n"
        "=== 待总结内容 ===\n" + body.text
    )

    req = ChatRequest(
        model=DEFAULT_MODEL,
        messages=[
            type("M", (), {"role": "system", "content": system_msg}),
            type("M", (), {"role": "user", "content": user_prompt})
        ],
        max_completion_tokens=200
    )

    raw = await smart_call(req)
    obj = _parse_llm_output(raw or "")
    if not obj:
        obj = {"article": "Agent 返回空响应", "moodKeywords": "sad,sad,sad", "model": DEFAULT_MODEL, "tokenUsageJson": "{}"}

    return SummarizeResultResp(
        article=obj.get("article", "（空）"),
        moodKeywords=obj.get("moodKeywords", "平静, 专注, 期待"),
        model=obj.get("model", DEFAULT_MODEL),
        tokenUsageJson=obj.get("tokenUsageJson", "")
    )


# ================= 工具函数 =================
def _parse_llm_output(raw: str) -> Dict[str, Any]:
    """兼容多种 LLM 输出格式（JSON / ```json``` / 文本段落）"""
    if not raw:
        return {}
    raw = raw.strip()

    # 1) 纯 JSON
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            inner = obj.get("article")
            if isinstance(inner, str):
                try:
                    inner_obj = json.loads(inner)
                    if isinstance(inner_obj, dict) and "article" in inner_obj:
                        # 把内层结构展平
                        obj.update(inner_obj)
                except Exception:
                    pass
            return obj
    except Exception:
        pass

    # 2) ```json``` 包裹
    m = re.search(r"```json\s*(\{.*?\})\s*```", raw, flags=re.S)
    if m:
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict):
                inner = obj.get("article")
                if isinstance(inner, str):
                    try:
                        inner_obj = json.loads(inner)
                        if isinstance(inner_obj, dict) and "article" in inner_obj:
                            obj.update(inner_obj)
                    except Exception:
                        pass
                return obj
        except Exception:
            pass

    # 3) 纯文本兜底（保持原样）
    article = ""
    mood = ""
    m1 = re.search(r"(?:^|\n)\s*#+\s*每日总结\s*(.+?)(?:\n#|\Z)", raw, flags=re.S)
    if m1:
        article = m1.group(1).strip()
    else:
        article = raw[:800].strip()
    m2 = re.search(r"(?:^|\n)\s*#+\s*今日情绪关键词\s*[:：]?\s*(.+)", raw)
    if m2:
        mood = m2.group(1).strip()

    return {
        "article": article or "（空）",
        "moodKeywords": mood or "平静, 希望, 未来",
        "model": "qwen-plus",
        "tokenUsageJson": ""
    }