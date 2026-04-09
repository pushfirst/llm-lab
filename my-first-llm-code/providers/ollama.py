import ollama
from base import BaseLLMProvider, LLMRequest, LLMResponse

class OllamaProvider(BaseLLMProvider):
    def __init__(self):
        self.default_model = "phi3"
    
    def generate(self, request: LLMRequest) -> LLMResponse:
        print("Ollama")
        model = request.model or self.default_model

        res = ollama.chat(
            model = model,
            messages = request.messages,
            options={"temperature": request.temperature}
        )

        return LLMResponse(
            content=res["message"]["content"],
            provider="ollama",
            model=model
        )