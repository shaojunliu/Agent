# WebSocket 聊天接口

import json
import os
import logging
from datetime import datetime
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from services.llm_clients import call_qwen
from core.config import DASHSCOPE_API_KEY, OPEN_API_KEY, DASH_URL, OPEN_URL, DEFAULT_MODEL,DEFAULT_CHAT_MODEL,QIANWEN_MAX
from models.chat_models import ChatRequest
from typing import Any, Dict, List

router = APIRouter()
logger = logging.getLogger("uvicorn.error")

@router.websocket("/ws/chat")
async def ws_chat(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            try:
                msg = await ws.receive()
            except WebSocketDisconnect:
                logger.info("[ws] client disconnected")
                break

            if msg.get("type") == "websocket.disconnect":
                logger.info("[ws] client disconnected: %s", msg)
                break

            if msg.get("text") is not None:
                raw = msg["text"]
            elif msg.get("bytes") is not None:
                raw = msg["bytes"].decode("utf-8", errors="replace")
            else:
                continue

            payload = None
            try:
                p = json.loads(raw)
                if isinstance(p, dict):
                    payload = p
            except Exception:
                pass

            try:
                prompts = _load_chat_prompts()
                req_obj = _build_chat_request(payload, raw, prompts)
            except Exception as e:
                logger.exception("Chat build failed", exc_info=e)
                await ws.send_json({"reply": ""})
                continue

            reply = ""
            try:
                reply = await call_qwen(req_obj)
                if reply is None:
                    reply = ""
            except Exception as e:
                logger.exception("LLM call failed", exc_info=e)
                reply = ""

            await ws.send_json({"reply": reply})
    except WebSocketDisconnect:
        return
    


def _stringify_value(v: Any) -> str:
    """Convert payload values into a compact string for prompt injection."""
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, (int, float, bool)):
        return str(v)
    # lists/dicts -> json
    try:
        return json.dumps(v, ensure_ascii=False)
    except Exception:
        return str(v)


def _parse_dt(x: Any) -> datetime | None:
    """Best-effort parse timestamps/dates."""
    if x is None:
        return None
    if isinstance(x, (int, float)):
        try:
            ts = float(x)
            # ms -> s
            if ts > 1e12:
                ts = ts / 1000.0
            return datetime.fromtimestamp(ts)
        except Exception:
            return None
    if not isinstance(x, str):
        return None
    s = x.strip()
    if not s:
        return None
    s = s.replace("/", "-")
    if s.endswith("Z"):
        s = s[:-1]
    try:
        return datetime.fromisoformat(s)
    except Exception:
        pass
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
        "%Y%m%d",
        "%Y%m%d%H%M%S",
    ):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            continue
    return None


def _extract_item_dt(item: Any) -> datetime | None:
    """Extract a datetime from common fields in a dict item."""
    if not isinstance(item, dict):
        return None
    for k in (
        "time",
        "timestamp",
        "ts",
        "createdAt",
        "created_at",
        "date",
        "datetime",
        "summaryDate",
        "summary_date",
    ):
        if k in item:
            dt = _parse_dt(item.get(k))
            if dt is not None:
                return dt
    return None

def _fmt_dt(dt: datetime, pivot: datetime | None) -> str:
    """Compact datetime prefix for prompts. Uses month-day + HH:MM in pivot's local interpretation."""
    try:
        # Keep it short: MM-DD HH:MM
        return dt.strftime("%m-%d %H:%M")
    except Exception:
        return ""


def _sort_items_closest_to(items: List[Any], pivot: datetime | None) -> List[Any]:
    """Sort by closeness to pivot (smallest abs delta first). If no pivot, sort newest first."""
    indexed = list(enumerate(items))

    def key_fn(t):
        idx, it = t
        dt = _extract_item_dt(it)
        if dt is None:
            # no time info -> push to end, stable order
            return (1, 10**18, idx)
        if pivot is None:
            # newest first
            return (0, -dt.timestamp(), idx)
        return (0, abs((dt - pivot).total_seconds()), idx)

    indexed.sort(key=key_fn)
    return [it for _, it in indexed]


def _render_messages_value(v: Any, pivot: datetime | None = None) -> str:
    """Render list of messages into compact lines. Preserve timestamps when present."""
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
                if not content:
                    continue

                dt = _extract_item_dt(item)
                dt_prefix = _fmt_dt(dt, pivot) if dt is not None else ""
                prefix = f"[{dt_prefix}] " if dt_prefix else ""

                if role == "user":
                    lines.append(f"{prefix}U:{content}")
                elif role == "assistant":
                    lines.append(f"{prefix}A:{content}")
                else:
                    lines.append(f"{prefix}{content}")
            else:
                s = _stringify_value(item).strip()
                if s:
                    lines.append(s)
        return "\n".join(lines)
    return _stringify_value(v)


def _get_payload_value(payload: Any, key: str) -> Any:
    """Prefer payload[key]; if missing/None, fallback to payload['args'][key]."""
    if not isinstance(payload, dict):
        return None
    if key in payload and payload.get(key) is not None:
        return payload.get(key)
    args = payload.get("args")
    if isinstance(args, dict):
        return args.get(key)
    return None


def _render_prechat_value(v: Any, pivot: datetime | None) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, list):
        items = _sort_items_closest_to(v, pivot)
        return _render_messages_value(items, pivot)
    if isinstance(v, dict):
        # best-effort: try common message list fields
        for k in ("messages", "items", "list"):
            if k in v and isinstance(v.get(k), list):
                return _render_messages_value(_sort_items_closest_to(v.get(k), pivot), pivot)
    return _stringify_value(v)


def _render_predaily_summary_value(v: Any, pivot: datetime | None) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, list):
        items = _sort_items_closest_to(v, pivot)
        lines: List[str] = []
        for it in items:
            if not isinstance(it, dict):
                s = _stringify_value(it).strip()
                if s:
                    lines.append(s)
                continue
            dt = _extract_item_dt(it)
            if dt is not None:
                # include time if present
                date_s = dt.strftime("%Y-%m-%d %H:%M")
            else:
                date_s = _stringify_value(it.get("summaryDate") or it.get("summary_date") or "").strip()
            title = _stringify_value(it.get("articleTitle") or it.get("title") or "").strip()
            memory = _stringify_value(it.get("memoryPoint") or "").strip()
            analyze = _stringify_value(it.get("analyzeResult") or "").strip()
            article = _stringify_value(it.get("article") or it.get("summary") or "").strip()

            head_parts: List[str] = []
            if date_s:
                head_parts.append(date_s)
            if title:
                head_parts.append(title)
            head = " ".join(head_parts).strip()

            extra = memory or analyze or article
            if extra:
                # truncate to save tokens
                if len(extra) > 160:
                    extra = extra[:160] + "…"
                lines.append(f"{head}: {extra}" if head else extra)
            elif head:
                lines.append(head)
        return "\n".join([x for x in lines if x])

    if isinstance(v, dict):
        for k in ("items", "list", "summaries"):
            if k in v and isinstance(v.get(k), list):
                return _render_predaily_summary_value(v.get(k), pivot)
    return _stringify_value(v)


def build_prompt_messages(prompts: Dict[str, Any], payload: Dict[str, Any]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    pivot = _parse_dt(payload.get("currentTime")) if isinstance(payload, dict) else None

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

        # For now, do not do special rendering for preChat / preDailySummary.
        # Inject raw payload values (stringify only for non-strings).
        for k, v in resolved.items():
            rendered_v = v if isinstance(v, str) else _stringify_value(v)
            rendered = rendered.replace("{" + str(k) + "}", rendered_v)

        out.append({"role": str(role), "content": rendered})

    # 1) system messages (highest priority)
    for m in (prompts.get("systemMessages") or []):
        if isinstance(m, dict):
            _process(m)

    # 2) context messages (optional, placed between system and user)
    for m in (prompts.get("contextMessages") or []):
        if isinstance(m, dict):
            _process(m)

    # 3) user messages
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
    b.model(model or DEFAULT_CHAT_MODEL)

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
    cfg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "chat_prompts_v2.json")
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        logger.exception("Failed to load chat prompts json")
        raise HTTPException(status_code=500, detail="chat prompts json 加载失败")