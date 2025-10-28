from pydantic import BaseModel
from typing import Literal

class Record(BaseModel):
    """
    对话记录单元
    - role: system / user / assistant / tool
    - content: 文本内容
    """
    role: Literal["system", "user", "assistant", "tool"] = "user"
    content: str


class SummaryReq(BaseModel):
    type: str                     # 必须为 "daily_summary"
    openid: str  # 用户 openId，可为空
    text: str                     # 必填，Memo 拼好的当天聊天内容

class SummarizeResultResp(BaseModel):
    article: str
    moodKeywords: str
    actionKeywords: str
    articleTitle: str
    model: str = "default"
    tokenUsageJson: str = ""