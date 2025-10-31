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

    # system + user 指令
    system_msg = "你是一个中文总结助手，请只输出 JSON。"
    SYSTEM_SUMMARY = (
    "你是一个的中文总结助手。请基于给定内容生成“每日总结\n"
    "仅返回一个JSON 对象，且键名必须严格为：article、moodKeywords、actionKeywords、articleTitle、model、tokenUsageJson。\n"
    "articleTitle为文章总结的标题,不超过8个纯中文字符\n"
    "moodKeywords 用中文逗号分隔 3 个词，例如：幸福, 轻松, 感恩。情绪关键字分为（喜悦、愤怒、悲伤、恐惧、厌恶、惊讶、内疚、亲密）9个大类。情绪关键字从子类选择从以下选择:喜悦包含：开心、轻松、满足、愉快、自豪、兴奋、平静、安心、幸福；愤怒包含:烦躁、不满、生气、愤慨、恼火、怨恨、冲动；悲伤包含:失落、沮丧、孤独、难过、惆怅、思念、遗憾；恐惧包含:紧张、焦虑、担心、不安、恐惧、害怕、担忧；厌恶包含:排斥、厌倦、嫌弃、反感、冷漠；惊讶包含:惊喜、震惊、意外、困惑、好奇；内疚包含:羞耻、内疚、后悔、自责、尴尬；亲密包含:亲近、温柔、体贴、感激、信赖、喜爱\n"
    "actionKeywords 为总结文本中的行为关键字分，例如：休息，工作，运动\n"
    "article文章原则 ：-共情：理解用户的处境与情绪，不轻易否定或过度解读。- 积极关注：关注用户的资源、努力与亮点，而非只强调不足。-温暖：在语言中传递支持感，让用户觉得被理解与陪伴。 - 真诚可信：不浮夸、不套路，保持自然、真实的人格化表达。文章中不要出现‘用户’、‘你’ 要以第一人称日记形式总结\n"
    "article文章框架整体输出文案在 400 字内。1. 看见：承认情绪 2. 理解：镜像认知 3. 回应：回应行为 4. 总结：赋予意义\n"
    "article文章策略：  根据记录内容的情绪（情绪正负向、情绪强度）、认知（认知合理性、信念强度）、行为（行为动机、行为方式和行为结果）等，进行不同程度的总结响应：如果输入便像消极应该积极引导\n"
    "=== 待总结内容 ===\n" + body.text
)


    req = ChatRequest(
        model=DEFAULT_MODEL,
        messages=[
            type("M", (), {"role": "system", "content": system_msg}),
            type("M", (), {"role": "user", "content": SYSTEM_SUMMARY})
        ],
        max_completion_tokens=200
    )

    raw = await smart_call(req)
    obj = _parse_llm_output(raw or "")
    if not obj:
        obj = {"article": "Agent 返回空响应", "moodKeywords": "sad,sad,sad","actionKeywords":"none,none,none","articleTitle":"title", "model": DEFAULT_MODEL, "tokenUsageJson": ""}

    # 若关键词缺失，用一次极简 LLM 调用补齐（可通过 FILL_MOOD_WITH_LLM 控制）
    if not obj.get("moodKeywords"):
        obj["moodKeywords"] = await _maybe_gen_mood_with_llm(obj.get("article",""))

    if not obj.get("actionKeywords"):
        obj["actionKeywords"] = await _maybe_gen_mood_with_llm(obj.get("article",""))  

    if not obj.get("articleTitle"):
        obj["articleTitle"] = await _maybe_gen_mood_with_llm(obj.get("article",""))         

    # 解析后的 obj 中可能含有 tokenUsageJson（dict）
    token_usage = _as_str(obj.get("tokenUsageJson"))
    
    return SummarizeResultResp(
        article=obj.get("article", "（空）"),
        moodKeywords=obj.get("moodKeywords", "平静，专注，期待"),  # 注意中文逗号
        actionKeywords=obj.get("actionKeywords", "平静，专注，期待"),  # 注意中文逗号
        articleTitle=obj.get("articleTitle", "平凡的一天"),
        model=obj.get("model", DEFAULT_MODEL),
        tokenUsageJson=token_usage
    )


# ================= 工具函数 =================
async def _maybe_gen_mood_with_llm(text: str) -> str:
    """缺少关键词时，用一次极简 LLM 调用补齐（可通过开关关闭）"""
    from models.chat_models import ChatRequest
    from services.llm_clients import smart_call, DEFAULT_MODEL
    if not FILL_MOOD_WITH_LLM or not text:
        return ""
    sys = "只输出三个中文情绪关键词，用中文逗号分隔。不要输出解释。"
    usr = f"请从以下内容提取三个情绪关键词：\n{text[:1200]}，三个行为关键字：\n{text[:1200]}，一个文章标题：\n{text[:1200]}"
    req = ChatRequest(
        model=DEFAULT_MODEL,
        messages=[type("M", (), {"role":"system","content":sys}),
                  type("M", (), {"role":"user","content":usr})],
        temperature=0.2,
        max_completion_tokens=32
    )
    out = await smart_call(req)
    out = _clean_text(out or "")
    # 简单规范化：只取前三个、用中文逗号连接
    parts = re.split(r"[,\uFF0C/|， ]+", out)
    parts = [p.strip() for p in parts if p.strip()]
    return "，".join(parts[:3])

def _parse_llm_output(raw: str) -> Dict[str, Any]:
    """尽可能把 LLM 输出规整成 {article, moodKeywords, actionKeywords, articleTitle, model, tokenUsageJson}"""
    if not raw:
        return {}
    raw = raw.strip()

    # ① 先尝试直接当 JSON 解析（包括 ```json ... ``` 这种）
    candidates = [raw]

    # 抓 ```json ... ``` 的情况
    m = re.search(r"```json\s*(\{.*?\})\s*```", raw, flags=re.S)
    if m:
        candidates.append(m.group(1))

    obj = None
    for cand in candidates:
        maybe = _try_parse_json(cand)
        # 第一种情况：直接就是 dict，这个最好
        if isinstance(maybe, dict):
            obj = maybe
            break
        # 第二种情况：是一个“JSON 字符串”，再解一层
        if isinstance(maybe, str):
            maybe2 = _try_parse_json(maybe)
            if isinstance(maybe2, dict):
                obj = maybe2
                break

    if not isinstance(obj, dict):
        obj = {}

    # ========== 以下保留你原来的补救逻辑 ==========
    # 2) 若 article 是“内嵌 JSON 字符串”，把它展开合并
    inner = {}
    article_val = obj.get("article")
    if isinstance(article_val, str):
        inner = _extract_inner_from_article(article_val)
        if "article" in inner:
            obj["article"] = inner["article"]
        else:
            obj["article"] = _clean_text(article_val)
        if "moodKeywords" in inner and not obj.get("moodKeywords"):
            obj["moodKeywords"] = inner["moodKeywords"]
        if "actionKeywords" in inner and not obj.get("actionKeywords"):
            obj["actionKeywords"] = inner["actionKeywords"]
        if "articleTitle" in inner and not obj.get("articleTitle"):
            obj["articleTitle"] = inner["articleTitle"]

    # 3) 补救 article
    if not obj.get("article"):
        m1 = re.search(r"(?:^|\n)\s*#+\s*每日总结\s*(.+?)(?:\n#|\Z)", raw, flags=re.S)
        obj["article"] = _clean_text(m1.group(1)) if m1 else _clean_text(raw[:800])

    # 4) 清洗 article
    obj["article"] = _clean_text(obj.get("article", ""))

    # 5) 补救情绪关键字
    if not obj.get("moodKeywords"):
        m2 = re.search(r"(?:^|\n)\s*#+\s*今日情绪关键词\s*[:：]?\s*([^\n]+)", raw)
        if m2:
            obj["moodKeywords"] = _clean_text(m2.group(1))

    # 6) 补救 actionKeywords
    if not obj.get("actionKeywords"):
        m3 = re.search(r"(?:^|\n)\s*#+\s*今日行为关键词\s*[:：]?\s*([^\n]+)", raw)
        if m3:
            obj["actionKeywords"] = _clean_text(m3.group(1))

    # 7) 补救 articleTitle
    if not obj.get("articleTitle"):
        m4 = re.search(r"(?:^|\n)\s*#+\s*今日文章标题\s*[:：]?\s*([^\n]+)", raw)
        if m4:
            obj["articleTitle"] = _clean_text(m4.group(1))

    # 8) 规范化
    if not obj.get("model"):
        obj["model"] = DEFAULT_MODEL
    if not obj.get("tokenUsageJson"):
        obj["tokenUsageJson"] = ""

    return obj

def _clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    # 先把常见的转义去掉
    s = s.replace("\\n", "\n").replace("\\r", "\r").replace("\\t", "\t")
    s = s.replace("\\\"", "\"")
    # 收敛多余空白
    s = re.sub(r"[ \t]+", " ", s)
    # 把多行变成一个段落（你也可以改成保留换行）
    s = re.sub(r"\s*\n\s*", " ", s).strip()
    return s

def _try_parse_json(s: str):
    try:
        return json.loads(s)
    except Exception:
        return None
    
def _extract_inner_from_article(article_val: str) -> dict:
    """
    处理把 JSON 当字符串塞进 article 的情况：
    - 先尝试把 article 当 JSON 解析
    - 解析失败就用正则把 "article": "..." / "moodKeywords": "..." 提取出来
    """
    if not isinstance(article_val, str):
        return {}

    cand = article_val.strip()

    # A) 如果是 { ... } 或 \"{...}\" 形态，直接再解析一次
    maybe = _try_parse_json(cand)
    if isinstance(maybe, dict) and ("article" in maybe or "moodKeywords" in maybe):
        return maybe

    # B) 正则从字符串里“抠出”内嵌的键值
    inner = {}
    # "article": "......"
    m_article = re.search(r'"article"\s*:\s*"(.+?)"', cand, flags=re.S)
    if m_article:
        inner["article"] = _clean_text(m_article.group(1))
    # "moodKeywords": "......"
    m_mood = re.search(r'"moodKeywords"\s*:\s*"(.+?)"', cand, flags=re.S)
    if m_mood:
        inner["moodKeywords"] = _clean_text(m_mood.group(1))
    m_action = re.search(r'"actionKeywords"\s*:\s*"(.+?)"', cand, flags=re.S)
    if m_action:
        inner["actionKeywords"] = _clean_text(m_action.group(1))        
    m_articleTitle = re.search(r'"articleTitle"\s*:\s*"(.+?)"', cand, flags=re.S)
    if m_articleTitle:
        inner["articleTitle"] = _clean_text(m_articleTitle.group(1))                

    return inner

def _as_str(v) -> str:
    """将 dict/list 转成 JSON 字符串；None 变空串；其余直接 str。"""
    if v is None:
        return ""
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)