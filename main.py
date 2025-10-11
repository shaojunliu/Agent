from fastapi import FastAPI
from routers.chat import router as chat_router
from routers.summary import router as summary_router

app = FastAPI(title="Agent (HTTP + WebSocket)")

@app.get("/healthz")
def healthz():
    return {"ok": "health !"}

# 注册路由
app.include_router(chat_router)      # /ws/chat
app.include_router(summary_router)   # /api/summary