# OpenAI / DashScope 调用封装

import httpx
from fastapi import HTTPException
from core.config import DASHSCOPE_API_KEY, OPEN_API_KEY, DASH_URL, OPEN_URL, DEFAULT_MODEL
from models.chat_models import ChatRequest

__all__ = ["call_gpt", "call_qwen", "smart_call", "DEFAULT_MODEL"]

def _extract_text_from_choices(choices):
    if isinstance(choices, list) and choices:
        ch0 = choices[0] or {}
        msg = ch0.get("message") or {}
        content = msg.get("content")
        if isinstance(content, str) and content:
            return content
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    t = part.get("text")
                    if isinstance(t, str):
                        parts.append(t)
            if parts:
                return "".join(parts)
        if isinstance(ch0.get("text"), str) and ch0["text"]:
            return ch0["text"]
    return None

def extract_reply(data: dict) -> str:
    if isinstance(data, dict) and "choices" in data:
        text = _extract_text_from_choices(data["choices"])
        if text:
            return text
    if isinstance(data, dict) and "output" in data and isinstance(data["output"], dict):
        out = data["output"]
        text = out.get("text")
        if isinstance(text, str) and text:
            return text
        if "choices" in out:
            text = _extract_text_from_choices(out["choices"])
            if text:
                return text
    return str(data)

async def call_gpt(req: ChatRequest) -> str:
    headers = {"Authorization": f"Bearer {OPEN_API_KEY}"}
    payload = req.to_dict()
    timeout = httpx.Timeout(20.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(OPEN_URL, headers=headers, json=payload)
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=r.text + " err from gpt")
    return extract_reply(r.json())

async def call_qwen(req: ChatRequest) -> str:
    headers = {"Authorization": f"Bearer {DASHSCOPE_API_KEY}"}
    msgs = [{"role": m.role, "content": m.content} for m in req.messages]
    payload = {
        "model": req.model or DEFAULT_MODEL,
        "input": {"messages": msgs},
        "parameters": {"result_format": "text"},
    }
    if req.temperature is not None:
        payload["parameters"]["temperature"] = req.temperature
    if req.max_completion_tokens is not None:
        payload["parameters"]["max_tokens"] = req.max_completion_tokens
    timeout = httpx.Timeout(300.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(DASH_URL, headers=headers, json=payload)
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    return extract_reply(r.json())

async def smart_call(req: ChatRequest) -> str:
    """
    简单策略：
      - 以 'gpt-' 开头 → 走 OpenAI
      - 以 'qwen' / 其他 → 走 DashScope
    """
    model = (req.model or DEFAULT_MODEL).lower()
    if model.startswith("gpt-"):
        return await call_gpt(req)
    return await call_qwen(req)