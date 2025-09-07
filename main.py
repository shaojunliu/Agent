import os
import httpx
from fastapi import FastAPI, HTTPException, Header, WebSocket, WebSocketDisconnect, Query
from pydantic import BaseModel
from typing import List,Optional
import json

# 从环境变量读取 DashScope API Key

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
# 从环境变量读取 Open API Key
OPEN_API_KEY = os.getenv("OPEN_API_KEY", "")
# 供上游服务调用鉴权（HTTP/WS）
AGENT_API_KEY = os.getenv("AGENT_API_KEY", "")  

if not OPEN_API_KEY:
    raise RuntimeError("请先在环境变量里设置 OPEN_API_KEY")

if not DASHSCOPE_API_KEY:
    raise RuntimeError("请先在环境变量里设置 DASHSCOPE_API_KEY")

app = FastAPI(title="Agent (HTTP + WebSocket)")

# ------------ 数据模型 ------------
class Message(BaseModel):
    role: str            # "user" / "assistant" / "system"
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    model: Optional[str] = "qwen-plus"
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 512

class ChatResponse(BaseModel):
    reply: str

# ------------ 共用调用 ------------
DASH_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
OPEN_URL = "https://api.openai.com/v1/chat/completions"

async def call_qwen(req: ChatRequest) -> str:
    headers = {"Authorization": f"Bearer {DASHSCOPE_API_KEY}"}
    payload = {
        "model": req.model or "qwen-plus",
        "input": {"messages": [m.dict() for m in req.messages]},
        # 需要可加参数："parameters": {"temperature": req.temperature, "max_tokens": req.max_tokens}
    }
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(DASH_URL, headers=headers, json=payload)
    if r.status_code != 200:
        # 将 dashscope 的错误透出，方便排查
        raise HTTPException(status_code=r.status_code, detail=r.text)
    data = r.json()
    return data.get("output", {}).get("text", str(data))


async def call_gpt(req: ChatRequest) -> str:
    headers = {"Authorization": f"Bearer {OPEN_API_KEY}"}
    payload = {
        "model": req.model or "gpt-5",
        "input": {"messages": [m.dict() for m in req.messages]},
        "temperature":1.2,
        "max_tokens":20
        # 需要可加参数："parameters": {"temperature": req.temperature, "max_tokens": req.max_tokens}
    }
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(DASH_URL, headers=headers, json=payload)
    if r.status_code != 200:
        # 将 dashscope 的错误透出，方便排查
        raise HTTPException(status_code=r.status_code, detail=r.text)
    data = r.json()
    return data.get("output", {}).get("text", str(data))


# ------------ 健康检查 ------------
@app.get("/healthz")
def healthz():
    return {"ok": "health !"}

# ------------ HTTP：/chat ------------
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, x_agent_key: str = Header(default="")):
    # 简单鉴权（可选）
    if AGENT_API_KEY and x_agent_key != AGENT_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    reply = await call_qwen(req)
    return ChatResponse(reply=reply)


# ------------ WebSocket：/ws/chat ------------
# 约定：连接时带 ?key=<AGENT_API_KEY>
# 消息格式支持两种：
# A) 直接发纯文本："你好，介绍一下你自己"
# B) 发 JSON：{"messages":[{"role":"user","content":"你好"}], "model":"qwen-plus"}
@app.websocket("/ws/chat")
async def ws_chat(ws: WebSocket, key: str | None = Query(default=None)):
    # 简单鉴权（可选）
    if AGENT_API_KEY and key != AGENT_API_KEY:
        await ws.close(code=4401)  # Unauthorized
        return

    await ws.accept()
    try:
        while True:
            # 优先接收文本帧；若是二进制帧则尝试按 UTF-8 解码
            try:
                raw = await ws.receive_text()              # 只收文本帧
            except Exception:
                # 客户端可能以 binary 发送，尝试转成文本
                bin_data = await ws.receive_bytes()
                raw = bin_data.decode("utf-8", errors="replace")

            # 解析消息：JSON 或纯文本
            try:
                data = json.loads(raw)
                req_obj = ChatRequest(**data)
            except Exception:
                req_obj = ChatRequest(messages=[Message(role="user", content=raw)])

            # 调用通义千问
            try:
                reply = await call_gpt(req_obj)
            except HTTPException as e:
                # 用文本帧返回 JSON
                await ws.send_json({"error": True, "status": e.status_code, "detail": e.detail})
                continue

            # 用文本帧返回 JSON（不要 send_bytes）
            await ws.send_json({"reply": reply})

    except WebSocketDisconnect:
        return

