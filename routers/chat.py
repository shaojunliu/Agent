# WebSocket 聊天接口

import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from services.builder import build_req_from_payload
from services.llm_clients import call_qwen  # 你也可以替换为 smart_call

router = APIRouter()

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
                req_obj = build_req_from_payload(payload, raw)
            except Exception as e:
                await ws.send_json({"error": True, "status": 400, "detail": f"bad request: {e!s}"})
                continue

            try:
                reply = await call_qwen(req_obj)
                reply = reply or "（空回复）"
            except HTTPException as e:
                await ws.send_json({"error": True, "status": e.status_code, "detail": e.detail})
                continue
            except Exception as e:
                await ws.send_json({"error": True, "status": 500, "detail": f"agent error: {e!s}"})
                continue

            await ws.send_json({"reply": reply})
    except WebSocketDisconnect:
        return