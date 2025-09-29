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

if not OPEN_API_KEY:
    raise RuntimeError("请先在环境变量里设置 OPEN_API_KEY")

if not DASHSCOPE_API_KEY:
    raise RuntimeError("请先在环境变量里设置 DASHSCOPE_API_KEY")

# ------------ 共用调用 ------------
DASH_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
OPEN_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_MODEL = "qwen-plus"  # 你也可以换成 gpt-4o 等

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


# ------------ 健康检查 ------------
@app.get("/healthz")
def healthz():
    return {"ok": "health !"}


@app.websocket("/ws/chat")
async def ws_chat(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            # 收文本；若客户端发二进制则尝试转成 utf-8 文本
            try:
                raw = await ws.receive_text()
            except Exception:
                raw = (await ws.receive_bytes()).decode("utf-8", errors="replace")

            # 尝试把文本解析为 JSON；失败就当纯文本
            payload = None
            try:
                payload_candidate = json.loads(raw)
                if isinstance(payload_candidate, dict):
                    payload = payload_candidate
            except Exception:
                pass

            # 构建统一的 ChatRequest
            try:
                req_obj = build_req_from_payload(payload, raw)
            except Exception as e:
                await ws.send_json({"error": True, "status": 400, "detail": f"bad request: {e!s}"})
                continue

            # 调模型（这里用 qwen；如果你做了 smart_call，可替换为 smart_call(req_obj)）
            try:
                reply = await call_qwen(req_obj)
                # reply 为空给个兜底
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
    
# -------- OpenAI ----------
async def call_gpt(req: ChatRequest) -> str:
    headers = {"Authorization": f"Bearer {OPEN_API_KEY}"}
    payload = req.to_dict()  # 你已有的 to_dict()

    timeout = httpx.Timeout(20.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(OPEN_URL, headers=headers, json=payload)

    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=r.text + " err from gpt")

    data = r.json()
    return extract_reply(data)

# -------- DashScope (Qwen) ----------】
async def call_qwen(req: ChatRequest) -> str:
    headers = {"Authorization": f"Bearer {DASHSCOPE_API_KEY}"}

    # dataclass -> dict（Message 没有 .dict()，手动取）
    msgs = [{"role": m.role, "content": m.content} for m in req.messages]

    payload = {
        "model": req.model or "qwen-plus",
        "input": {"messages": msgs},
        "parameters": {"result_format": "text"}  # 希望返回纯文本
    }
    if req.temperature is not None:
        payload["parameters"]["temperature"] = req.temperature
    if req.max_completion_tokens is not None:
        payload["parameters"]["max_tokens"] = req.max_completion_tokens  # DashScope 参数名

    timeout = httpx.Timeout(20.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(DASH_URL, headers=headers, json=payload)

    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=r.text)

    data = r.json()
    return extract_reply(data)

def build_req_from_payload(payload: dict | None, raw_text: str) -> ChatRequest:
    """
    兼容以下 JSON 结构（都可选）：
    A) {"message": "你好"}                      # 单条
    B) {"messages":[{"role":"user","content":"hi"}], "model":"...", "temperature":0.7}
    C) {"model":"...", "prompt":"你好"}        # 语法糖
    D) {"max_tokens": 64} / {"max_completion_tokens": 64}
    若 payload 为 None，则用 raw_text 作为 user 消息
    """
    b = ChatRequest.builder()

    # 1) model
    model = None
    if isinstance(payload, dict):
        model = payload.get("model")
    b.model(model or DEFAULT_MODEL)

    # 2) 参数
    if isinstance(payload, dict):
        if "temperature" in payload and isinstance(payload["temperature"], (int, float)):
            b.temperature(float(payload["temperature"]))
        mt = payload.get("max_completion_tokens", payload.get("max_tokens"))
        if isinstance(mt, int):
            b.max_completion_tokens(mt)

    # 3) messages
    b.addMessage("system",
        "你是一个聊天机器人，但你要把自己当成一个真人，和用户平等话语权的人，负责简单回复。"
        "回复要口语化、与用户风格相近；一次答复到一个标点结束；不要包含“哈哈”等语气词。")

    if isinstance(payload, dict):
        if isinstance(payload.get("messages"), list) and payload["messages"]:
            for m in payload["messages"]:
                role = (m or {}).get("role")
                content = (m or {}).get("content")
                if isinstance(role, str) and isinstance(content, str) and content.strip():
                    b.addMessage(role, content)
        else:
            # 优先 message/prompt 字段
            msg = payload.get("message") or payload.get("prompt")
            if isinstance(msg, str) and msg.strip():
                b.addMessage("user", msg.strip())
            elif raw_text.strip():
                b.addMessage("user", raw_text.strip())
    else:
        # 非 JSON：把 raw 当纯文本
        if raw_text.strip():
            b.addMessage("user", raw_text.strip())

    return b.build()

# -------- 通用解析 --------
def _extract_text_from_choices(choices):
    if isinstance(choices, list) and choices:
        ch0 = choices[0] or {}
        msg = ch0.get("message") or {}
        content = msg.get("content")

        # 1) 纯文本
        if isinstance(content, str) and content:
            return content

        # 2) 富媒体分片：拼接 text
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    t = part.get("text")
                    if isinstance(t, str):
                        parts.append(t)
            if parts:
                return "".join(parts)

        # 3) 极少数兼容：choices[0].text
        if isinstance(ch0.get("text"), str) and ch0["text"]:
            return ch0["text"]

    return None

def extract_reply(data: dict) -> str:
    # A) OpenAI 风格：顶层 choices
    if isinstance(data, dict) and "choices" in data:
        text = _extract_text_from_choices(data["choices"])
        if text:
            return text

    # B) DashScope 风格：output.text / output.choices
    if isinstance(data, dict) and "output" in data and isinstance(data["output"], dict):
        out = data["output"]
        # 直接给了纯文本
        text = out.get("text")
        if isinstance(text, str) and text:
            return text
        # 有的模型也会给 choices
        if "choices" in out:
            text = _extract_text_from_choices(out["choices"])
            if text:
                return text

    # 兜底：把原始响应转成字符串返回
    return str(data)