# dataclass：Message / ChatRequest / Builder

from dataclasses import dataclass, field
from typing import List, Optional

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
        self._messages: List[Message] = []
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

    def max_completion_tokens(self, tokens: int):
        self._max_completion_tokens = tokens
        return self

    def build(self) -> "ChatRequest":
        return ChatRequest(
            model=self._model,
            messages=self._messages,
            temperature=self._temperature,
            max_completion_tokens=self._max_completion_tokens,
        )