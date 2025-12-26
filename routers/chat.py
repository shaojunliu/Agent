# WebSocket 聊天接口

import json
import os
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from services.llm_clients import call_qwen, DEFAULT_MODEL
from models.chat_models import ChatRequest, Message

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

    # 1. Default system message
    system_msgs_cfg = prompts.get("systemMessages") or []
    for m in system_msgs_cfg:
        role = m.get("role")
        content = m.get("content")
        if role == "system" and content:
            b.addMessage("system", content)

    if isinstance(payload, dict):
        arr = payload.get("messages")
        preChat = payload.get("preChat")
        preDailySummary = payload.get("preDailySummary")
        args = payload.get("args") or {}
        lng = args.get("lng")
        lat = args.get("lat")

        # 准备数据片段
        location_text = ""
        history_text = ""
        summary_text = ""
        current_msg_text = ""

        # A. Location Context
        loc_cfg = prompts.get("locationContext") or {}
        template = loc_cfg.get("template", "")
        
        if isinstance(lng, (int, float)) and isinstance(lat, (int, float)):
            location_text = template.replace("{lng}", f"{lng:.6f}").replace("{lat}", f"{lat:.6f}")
        elif (isinstance(lng, str) and lng.strip()) and (isinstance(lat, str) and lat.strip()):
            location_text = template.replace("{lng}", lng.strip()).replace("{lat}", lat.strip())

        # B. History & Summary Labels
        hist_labels = prompts.get("historyLabels") or {}
        label_user = hist_labels.get("user", "[历史会话-用户] ")
        label_assistant = hist_labels.get("assistant", "[历史会话-助手] ")
        label_unknown = hist_labels.get("unknown", "[历史会话] ")
        label_summary = hist_labels.get("summary", "[过往摘要记忆] ")

        # C. History Text
        if isinstance(preChat, list):
            lines = []
            for item in preChat:
                role = (item or {}).get("role")
                content = (item or {}).get("content")
                if not (isinstance(content, str) and content.strip()):
                    continue
                text = content.strip()
                if role == "user":
                    lines.append(f"{label_user}{text}")
                elif role == "assistant":
                    lines.append(f"{label_assistant}{text}")
                else:
                    lines.append(f"{label_unknown}{text}")
            if lines:
                history_text = "\n".join(lines)

        # D. Daily Summary Text
        if isinstance(preDailySummary, list):
            lines = []
            for summary in preDailySummary:
                text = (summary or {}).get("summary") or (summary or {}).get("content")
                if isinstance(text, str) and text.strip():
                    lines.append(f"{label_summary}{text.strip()}")
            if lines:
                summary_text = "\n".join(lines)

        # E. Current Messages Text
        if isinstance(arr, list) and arr:
            # 如果是一个list，我们也把它拼接成文本
            lines = []
            for m in arr:
                role = (m or {}).get("role")
                content = (m or {}).get("content")
                if isinstance(content, str) and content.strip():
                    # 这里假设当前对话也用 label_user/assistant 标记?
                    # 用户需求是"用户本次消息"，通常指最后一句。
                    # 如果arr是多条，我们直接拼接内容。
                    lines.append(content.strip())
            if lines:
                current_msg_text = "\n".join(lines)
        else:
            msg = payload.get("message") or payload.get("prompt")
            if isinstance(msg, str) and msg.strip():
                current_msg_text = msg.strip()
            elif raw_text.strip():
                current_msg_text = raw_text.strip()

        # F. 构建 User Messages (从配置读取并替换占位符)
        user_msgs_cfg = prompts.get("userMessages") or []
        for m in user_msgs_cfg:
            role = m.get("role")
            content_tpl = m.get("content")
            if role == "user" and content_tpl:
                # 替换占位符
                final_content = content_tpl.replace("{location}", location_text) \
                                           .replace("{history}", history_text) \
                                           .replace("{summary}", summary_text) \
                                           .replace("{message}", current_msg_text)
                # 清理多余空行 (可选，但通常好一点)
                b.addMessage("user", final_content)

    return b.build()    

def _load_chat_prompts():
    cfg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "chat_prompts.json")
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        logger.exception("Failed to load chat prompts json")
        raise HTTPException(status_code=500, detail="chat prompts json 加载失败")