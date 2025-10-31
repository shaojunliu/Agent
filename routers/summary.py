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

# 是否允许在缺少 moodKeywords 时进行一次极简补充调用
FILL_MOOD_WITH_LLM = True

router = APIRouter(prefix="/summary")


# ================= 主逻辑 =================
@router.post("/daily", response_model=SummarizeResultResp)
async def summarize(body: SummaryReq):
    if body.type != "daily_summary":
        raise HTTPException(status_code=400, detail="type 必须为 'daily_summary'")
    if not body.text.strip():
        raise HTTPException(status_code=400, detail="text 不能为空")

    # ===== 提示词：最前面先给“强制 JSON 模版” =====
    system_msg = "你是一个日记总结助手，只能输出 JSON，不要输出解释，不要输出 markdown，不要输出代码块。"

    SYSTEM_SUMMARY = (
        "下面是你必须遵守的输出格式，请只返回这个 JSON，字段名必须一模一样，不要多字段，不要少字段：\n"
        "{\n"
        "    \"article\":\"文章总结...\",\n"
        "    \"moodKeywords\":\"期待,焦虑,满足\",\n"
        "    \"actionKeywords\":\"跑步,工作\",\n"
        "    \"articleTitle\":\"美好的一天\",\n"
        "    \"model\":\"qwen-plus\",\n"
        "    \"tokenUsageJson\":\"\"\n"
        "}\n"
        "如果无法生成，请返回一个空 JSON：{}\n"
        "下面是写作要求，请严格参考：\n"
        "你是一个的中文总结助手。请基于给定内容生成“每日总结”。\n"
        "articleTitle为文章总结的标题,不超过8个纯中文字符。\n"
        "moodKeywords 用中文逗号分隔 3 个词，例如：幸福, 轻松, 感恩。情绪关键字分为（喜悦、愤怒、悲伤、恐惧、厌恶、惊讶、内疚、亲密）9个大类。情绪关键字从子类选择从以下选择:喜悦包含：开心、轻松、满足、愉快、自豪、兴奋、平静、安心、幸福；愤怒包含:烦躁、不满、生气、愤慨、恼火、怨恨、冲动；悲伤包含:失落、沮丧、孤独、难过、惆怅、思念、遗憾；恐惧包含:紧张、焦虑、担心、不安、恐惧、害怕、担忧；厌恶包含:排斥、厌倦、嫌弃、反感、冷漠；惊讶包含:惊喜、震惊、意外、困惑、好奇；内疚包含:羞耻、内疚、后悔、自责、尴尬；亲密包含:亲近、温柔、体贴、感激、信赖、喜爱。\n"
        "actionKeywords 为总结文本中的行为关键字，例如：休息，工作，运动。\n"
        "article 需要用第一人称日记视角，不要出现“用户”“你”。不超过 400 字。\n"
        "=== 待总结内容 ===\n"
        + body.text
    )


    req = ChatRequest(
        model=DEFAULT_MODEL,
        messages=[
            type("M", (), {"role": "system", "content": system_msg}),
            type("M", (), {"role": "user", "content": SYSTEM_SUMMARY}),
        ],
        max_completion_tokens=200,
    )

    raw = await smart_call(req)
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
    """
    只做 JSON 解析：
    - 能直接 json.loads 成 dict → 用
    - 能 json.loads 成 str → 再解一次
    - 其他情况 → 返回 {}
    """
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