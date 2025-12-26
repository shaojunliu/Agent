# ChatRequest 构建、通用解析

from services.llm_clients import DEFAULT_MODEL
from models.chat_models import ChatRequest, Message

def build_req_from_payload(payload: dict | None, raw_text: str) -> ChatRequest:
    b = ChatRequest.builder()

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
        "你是一个日记聊天机器人。但你要把自己当成真人，以口语化、简洁的方式回答。一次答复到一个标点结束；不要出现“哈哈”等语气词。")
    
    
    if isinstance(payload, dict):
        arr = payload.get("messages")
        preChat = payload.get("preChat")
        preDailySummary = payload.get("preDailySummary")
        args = payload.get("args") or {}
        lng = args.get("lng")  # 经度
        lat = args.get("lat")  # 纬度

        # === 根据经纬度 获取位置、当地时间、天气 ===
        # 说明：这里不做真实的地理/天气查询，而是把经纬度作为上下文交给模型：
        # - 让模型基于经纬度推断用户大致位置（城市/区域即可，不必精确到地址）
        # - 推断当地时间与时段（清晨/白天/傍晚/深夜）
        # - 给出当日/次日天气的口语化建议（允许基于常识做保守推断；若无法判断要明确说明不确定）
        # - 若用户发起对话时间较早/较晚，加入人文关怀与作息提醒，并在合适时提示“第二天天气”

        # 将经纬度注入提示词，供模型在回答中参考
        if isinstance(lng, (int, float)) and isinstance(lat, (int, float)):
            b.addMessage(
                "system",
                "[位置上下文] 用户经纬度：lng={:.6f}, lat={:.6f}。\n"
                "请你在回答前先在心里完成：\n"
                "1) 根据经纬度推断用户大致位置（城市/区域层级即可）。\n"
                "2) 推断该地当前当地时间与时段（清晨/上午/下午/傍晚/深夜）。\n"
                "3) 结合时段，用口语化方式给出与天气相关的贴心建议：如穿衣、出行、跑步、带伞等。\n"
                "   若你无法可靠获得实时天气，请明确说明‘我无法实时查询天气’，并给出保守建议。\n"
                "4) 如果判断当前为较早(约05:00-07:00)或较晚(约23:00-02:00)的对话，注意人文关怀：\n"
                "   简短关心用户作息/疲劳，并在合适时提醒第二天的天气与安排。\n"
                "注意：不要编造具体气温/降雨概率等精确数值；更偏向建议型表达。"
            )
        elif (isinstance(lng, str) and lng.strip()) and (isinstance(lat, str) and lat.strip()):
            # 兼容字符串形式的经纬度
            b.addMessage(
                "system",
                f"[位置上下文] 用户经纬度：lng={lng.strip()}, lat={lat.strip()}。\n"
                "请你根据经纬度推断用户大致位置与当地时间时段，并给出天气相关的贴心建议。"
                "若无法实时查询天气，请明确说明并给出保守建议；若对话很早/很晚，加入人文关怀并适当提醒第二天的天气。"
            )


        # === 1. 历史会话：全部用 system + 明确标签 ===
        if isinstance(preChat, list):
            for item in preChat:
                role = (item or {}).get("role")
                content = (item or {}).get("content")
                if not (isinstance(content, str) and content.strip()):
                    continue
                text = content.strip()
                if role == "user":
                    b.addMessage("system", f"[历史会话-用户] {text}")
                elif role == "assistant":
                    b.addMessage("system", f"[历史会话-助手] {text}")
                else:
                    # 兜底：未知角色
                    b.addMessage("system", f"[历史会话] {text}")

        # === 2. 历史摘要：继续用 system，加上记忆标签 ===
        if isinstance(preDailySummary, list):
            for summary in preDailySummary:
                text = (summary or {}).get("summary") or (summary or {}).get("content")
                if isinstance(text, str) and text.strip():
                    b.addMessage("system", f"[过往摘要记忆] {text.strip()}")

        # === 3. 当前这轮 messages / message / prompt ===
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

    return b.build()