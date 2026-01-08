# 环境变量、常量

import os

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
OPEN_API_KEY = os.getenv("OPEN_API_KEY", "")

if not OPEN_API_KEY:
    raise RuntimeError("请先在环境变量里设置 OPEN_API_KEY")
if not DASHSCOPE_API_KEY:
    raise RuntimeError("请先在环境变量里设置 DASHSCOPE_API_KEY")

DASH_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
OPEN_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_CHAT_MODEL = "qwen-plus-character"
DEFAULT_MODEL = "qwen-plus"