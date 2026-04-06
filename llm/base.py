from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class LLMRequest:
    messages: List[Dict[str, str]]
    model: str | None = None
    temperature: float = 0.7

@dataclass
class LLMResponse:
    content: str
    provider: str
    model: str

class BaseLLMProvider(ABC):
    @abstractmethod
    def generate(self, request: LLMRequest) -> LLMResponse:
        pass