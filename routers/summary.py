# 新增：总结接口

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Literal
from models.chat_models import ChatRequest, Message
from services.llm_clients import smart_call, DEFAULT_MODEL
from typing import List, Optional, Union, Dict, Any, Literal
from models.record_model import Record,SummaryReq,SummarizeResultResp
import json
import os
import re
import logging

# 是否允许在缺少 moodKeywords 时进行一次极简补充调用
FILL_MOOD_WITH_LLM = True

router = APIRouter(prefix="/summary")

# 用 uvicorn 的 logger，确保日志出现在 docker logs / uvicorn 输出里
logger = logging.getLogger("uvicorn.error")


# ================= 主逻辑 =================
@router.post("/daily", response_model=SummarizeResultResp)
async def summarize(body: SummaryReq):
    if body.type != "daily_summary":
        logger.error(f"Invalid summary type: {body.type}")
        raise HTTPException(status_code=400, detail="type 必须为 'daily_summary'")
    if not body.text.strip():
        logger.error("Summary text is empty")
        raise HTTPException(status_code=400, detail="text 不能为空")

    cfg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "summary_prompts.json")
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            prompts = json.load(f)
    except Exception:
        logger.exception("Failed to load summary prompts json")
        raise HTTPException(status_code=500, detail="summary prompts json 加载失败")

    system_messages_cfg = prompts.get("systemMessages") or []
    user_messages_cfg = prompts.get("userMessages") or []
    content_prefix = prompts.get("content_prefix") or "=== 待总结内容 ===\n"

    system_messages = [Message(role=m.get("role"), content=m.get("content", "")) for m in system_messages_cfg if (m or {}).get("role") == "system"]
    user_messages = [str((m or {}).get("content", "")) for m in user_messages_cfg if (m or {}).get("role") == "user"]
    user_combined = "\n".join([c for c in user_messages if c.strip()])
    user_combined = (user_combined + "\n" + content_prefix + body.text).strip()


    req = ChatRequest(
        model=DEFAULT_MODEL,
        messages=[*system_messages, Message(role="user", content=user_combined)],
        max_completion_tokens=2000,
    )
    logger.info("daily summary request"+str(req))
    raw = await smart_call(req)
    logger.info("daily summary response"+raw)
    try:
        s = str(raw)
        logger.info("LLM raw output len=%d head=%s", len(s), s)
    except Exception:
        logger.exception("Failed to log LLM raw output")
    obj = _parse_llm_output(raw or "")

    # 解析失败 → 直接返回空 json
    if not obj:
        return SummarizeResultResp(
            article="返回空响应",
            moodKeywords="",
            actionKeywords="",
            articleTitle="",
            model="",
            tokenUsageJson="",
        )
        # 解析成功 → 填入，有哪些给哪些
    return SummarizeResultResp(
        article=_clean_text(obj.get("article", "")),
        moodKeywords=_clean_text(obj.get("moodKeywords", "")),
        actionKeywords=_clean_text(obj.get("actionKeywords", "")),
        articleTitle=_clean_text(obj.get("articleTitle", "")),
        model=_clean_text(obj.get("model", "")),
        tokenUsageJson=_as_str(obj.get("tokenUsageJson", "")),
    )


def _parse_llm_output(raw: str) -> Dict[str, Any]:
    if not raw:
        return {}

    raw = raw.strip()

    # 有些模型会输出 ```json ... ```，这里先去掉外壳
    if raw.startswith("```") and raw.endswith("```"):
        raw = raw.strip("`").strip()
        # 再防止前面有 json
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()

    # 第一次尝试
    first = _try_parse_json(raw)
    if isinstance(first, dict):
        return first
    if isinstance(first, str):
        second = _try_parse_json(first)
        if isinstance(second, dict):
            return second

    # 解析失败，直接返回空
    return {}


def _clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return s.strip()

def _try_parse_json(s: str):
    try:
        return json.loads(s)
    except Exception:
        return None
    
def _as_str(v) -> str:
    if v is None:
        return ""
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)
