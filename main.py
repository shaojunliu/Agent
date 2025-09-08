import os
import httpx
from fastapi import FastAPI, HTTPException, Header, WebSocket, WebSocketDisconnect, Query
from pydantic import BaseModel
from typing import List,Optional
import json
from dataclasses import dataclass, field

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
@dataclass
class Message:
    role: str
    content: str

@dataclass
class ChatRequest:
    model: str
    messages: List[Message] = field(default_factory=list)
    temperature: Optional[float] = None
    max_completion_tokens: Optional[int] = None
    def to_dict(self):
        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in self.messages],
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.max_completion_tokens is not None:
            payload["max_completion_tokens"] = self.max_completion_tokens
        return payload

    @classmethod
    def builder(cls):
        return ChatRequestBuilder()


class ChatRequestBuilder:
    def __init__(self):
        self._model = None
        self._messages = []
        self._temperature = None
        self._max_completion_tokens = None

    def model(self, model: str):
        self._model = model
        return self

    def addMessage(self, role: str, content: str):
        self._messages.append(Message(role=role, content=content))
        return self
    def temperature(self, temp: float):
        self._temperature = temp
        return self 
    
    def max_completion_tokens(self, tokens: int):           # ← 方法名可保持不变
        self._max_completion_tokens = tokens                # ← 写入带下划线的属性
        return self

    def build(self):
        return ChatRequest(
            model=self._model,
            messages=self._messages,
            temperature=self._temperature,
            max_completion_tokens=self._max_completion_tokens,
        )
    


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
    payload = req.to_dict()
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(OPEN_URL, headers=headers, json=payload)
    if r.status_code != 200:
        # 将 dashscope 的错误透出，方便排查
        raise HTTPException(status_code=r.status_code, detail=r.text+"err from gpt")
    data = r.json()
    # -------- 1) OpenAI 风格: chat.completions --------
    # 典型返回: {"choices": [{"message": {"role":"assistant","content":"..."} , ...}], "usage": {...}}
    if isinstance(data, dict) and "choices" in data and data["choices"]:
        ch0 = data["choices"][0] or {}
        # 优先取 message.content
        msg = ch0.get("message") or {}
        content = msg.get("content")

        # content 可能是 str 或 富媒体分片(list)
        if isinstance(content, str) and content:
            return content

        if isinstance(content, list):
            # chat 内容为多段结构化分片时：拼接其中的 text
            texts = []
            for part in content:
                # OpenAI 多模态结构: {"type": "text", "text": "..."}
                if isinstance(part, dict) and part.get("type") == "text":
                    t = part.get("text")
                    if isinstance(t, str):
                        texts.append(t)
            if texts:
                return "".join(texts)

        # 某些实现可能放在 choice 的其他字段（极少见）
        if "text" in ch0 and isinstance(ch0["text"], str) and ch0["text"]:
            return ch0["text"]


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
            req_obj = (ChatRequest.builder().model("gpt-4")
                        .addMessage("system","你是一个聊天机器人，但你要把自己当成一个真人，和用户平等话语权的人，负责简单回复,回复需要人性化，口语化，和用户输入风格相似。答复不需要敬语，一次答复只返回到一个标点符号结束，回复不要包含 哈哈等语气词")
                        .addMessage("user", raw).max_completion_tokens(30).build())

            # 调用模型
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

