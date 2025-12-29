# WebSocket 聊天接口

import json
import os
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from services.llm_clients import call_qwen, DEFAULT_MODEL
from models.chat_models import ChatRequest, Message
from typing import Any, Dict, List, Optional

router = APIRouter()
logger = logging.getLogger("uvicorn.error")

@router.websocket("/ws/chat")
async def ws_chat(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            try:
                raw = await ws.receive_text()
            except Exception:
                raw = (await ws.receive_bytes()).decode("utf-8", errors="replace")

            payload = None
            try:
                p = json.loads(raw)
                if isinstance(p, dict):
                    payload = p
            except Exception:
                pass

            try:
                # 每次请求都重新加载配置，或者也可以全局加载一次。
                # 既然是 '直接调用json文件'，为了支持热更新，每次加载虽然有开销，但符合 '修改直接生效' 的直觉。
                # 但考虑到性能，生产环境通常缓存。这里按用户要求，确保修改生效。
                prompts = _load_chat_prompts()
                req_obj = _build_chat_request(payload, raw, prompts)
            except HTTPException as e:
                logger.error(f"HTTPException in chat build: {e.detail}")
                await ws.send_json({"error": True, "status": e.status_code, "detail": e.detail})
                continue
            except Exception as e:
                logger.exception("Error building chat request")
                await ws.send_json({"error": True, "status": 400, "detail": f"bad request: {e!s}"})
                continue

            try:
                reply = await call_qwen(req_obj)
                reply = reply or "（空回复）"
            except HTTPException as e:
                logger.error(f"LLM call failed: {e.detail}")
                await ws.send_json({"error": True, "status": e.status_code, "detail": e.detail})
                continue
            except Exception as e:
                logger.exception("Agent error during LLM call")
                await ws.send_json({"error": True, "status": 500, "detail": f"agent error: {e!s}"})
                continue

            await ws.send_json({"reply": reply})
    except WebSocketDisconnect:
        return
    


def _stringify_value(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, (int, float, bool)):
        return str(v)
    try:
        return json.dumps(v, ensure_ascii=False)
    except Exception:
        return str(v)


def _render_messages_value(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, list):
        lines: List[str] = []
        for item in v:
            if isinstance(item, dict):
                role = _stringify_value(item.get("role", "")).strip()
                content = _stringify_value(item.get("content", "")).strip()
                if role or content:
                    lines.append(f"[{role}] {content}".strip())
            else:
                s = _stringify_value(item).strip()
                if s:
                    lines.append(s)
        return "\n".join(lines)
    return _stringify_value(v)


def _get_payload_value(payload: Any, key: str) -> Any:
    if not isinstance(payload, dict):
        return None
    if key in payload:
        return payload.get(key)
    # backward compatibility: lng/lat may be nested under args
    args = payload.get("args")
    if isinstance(args, dict):
        return args.get(key)
    return None


def build_prompt_messages(prompts: Dict[str, Any], payload: Dict[str, Any]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []

    def _process(template: Dict[str, Any]) -> None:
        role = template.get("role")
        content = template.get("content")
        if not role or content is None:
            return

        needArgs = template.get("needArgs") or []
        # needArgs empty => keep content
        if not needArgs:
            out.append({"role": str(role), "content": str(content)})
            return

        # needArgs non-empty => all must exist and not None
        resolved: Dict[str, Any] = {}
        for k in needArgs:
            v = _get_payload_value(payload, str(k))
            if v is None:
                return  # skip this message
            resolved[str(k)] = v

        rendered = str(content)

        # Special-case: when needArgs include 'messages', allow injecting it into {content}
        if "message" in resolved:
            msg_text = _render_messages_value(resolved["message"])
            if "{content}" in rendered:
                rendered = rendered.replace("{content}", msg_text)
            if "{message}" in rendered:
                rendered = rendered.replace("{message}", msg_text)

        # Standard replacements: {arg}
        for k, v in resolved.items():
            rendered = rendered.replace("{" + str(k) + "}", _stringify_value(v))

        out.append({"role": str(role), "content": rendered})

    for m in (prompts.get("systemMessages") or []):
        if isinstance(m, dict):
            _process(m)

    for m in (prompts.get("userMessages") or []):
        if isinstance(m, dict):
            _process(m)

    return out

def _build_chat_request(payload: dict | None, raw_text: str, prompts: dict) -> ChatRequest:
    b = ChatRequest.builder()

    # model
    model = None
    if isinstance(payload, dict):
        model = payload.get("model")
    b.model(model or DEFAULT_MODEL)

    # params
    if isinstance(payload, dict):
        if "temperature" in payload and isinstance(payload["temperature"], (int, float)):
            b.temperature(float(payload["temperature"]))
        mt = payload.get("max_completion_tokens", payload.get("max_tokens"))
        if isinstance(mt, int):
            b.max_completion_tokens(mt)

    # messages (built from chat_prompts.json)
    if isinstance(payload, dict):
        messages = build_prompt_messages(prompts, payload)
        for m in messages:
            role = m.get("role")
            content = m.get("content")
            if role and content is not None:
                b.addMessage(role, content)

    return b.build()    

def _load_chat_prompts():
    cfg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "chat_prompts.json")
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        logger.exception("Failed to load chat prompts json")
        raise HTTPException(status_code=500, detail="chat prompts json 加载失败")