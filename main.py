from fastapi import FastAPI
from routers.chat import router as chat_router
from routers.summary import router as summary_router

app = FastAPI(title="Agent (HTTP + WebSocket)")

@app.get("/healthz")
def healthz():
    return {"ok": "health !"}

# 新增: 返回所有 chat 提示词配置
@app.get("/api/prompts/chat")
def allChatPrompts():
    """
    chat prompts 查询入口（仅路由，不做具体实现）
    """
    return {"ok": True}

# 新增: 返回所有 summary 提示词配置
@app.get("/api/prompts/summary")
def allSummaryPrompts():
    """
    summary prompts 查询入口（仅路由，不做具体实现）
    """
    return {"ok": True}

# 注册路由
app.include_router(chat_router)      # /ws/chat
app.include_router(summary_router)   # /api/summary