# ChatRequest 构建、通用解析

from .llm_clients import DEFAULT_MODEL
from ..models.chat_models import ChatRequest, Message

def build_req_from_payload(payload: dict | None, raw_text: str) -> ChatRequest:
    """
    兼容：
      - {"message":"你好"} / {"prompt":"..."}
      - {"messages":[{"role":"user","content":"hi"}], "model":"...", "temperature":0.7}
      - {"max_tokens":64} / {"max_completion_tokens":64}
    """
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

    # default system
    b.addMessage("system",
        "你是一个聊天机器人，但你要把自己当成真人，以口语化、简洁的方式回答。一次答复到一个标点结束；不要出现“哈哈”等语气词。")

    # messages
    if isinstance(payload, dict):
        arr = payload.get("messages")
        if isinstance(arr, list) and arr:
            for m in arr:
                role = (m or {}).get("role")
                content = (m or {}).get("content")
                if isinstance(role, str) and isinstance(content, str) and content.strip():
                    b.addMessage(role, content)
        else:
            msg = payload.get("message") or payload.get("prompt")
            if isinstance(msg, str) and msg.strip():
                b.addMessage("user", msg.strip())
            elif raw_text.strip():
                b.addMessage("user", raw_text.strip())
    else:
        if raw_text.strip():
            b.addMessage("user", raw_text.strip())

    return b.build()