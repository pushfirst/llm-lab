import os
from openai import OpenAI
from dotenv import load_dotenv

from llm.base import BaseLLMProvider, LLMRequest, LLMResponse

load_dotenv()

class OpenAIProvider(BaseLLMProvider):
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.default_model = "gpt-5-nano"
    # load environment variables from env and check for key
    def validateApiKey(self):
        load_dotenv(override=True)
        api_key = os.getenv('OPENAI_API_KEY')

        if not api_key:
            print('API key not provided.')
            return False
        elif not api_key.startswith("sk-proj-"):
            print('Invalid API key')
            return False
        elif api_key.strip() != api_key:
            print("API key is contains speces or tabs etc...")
            return False
        else:
            print('Voila! Valid API Key')
            return True
        
    def generate(self, request: LLMRequest) -> LLMResponse:
        print("OpenAI")
        if self.validateApiKey() is False:
            return LLMResponse(
                content="Could not call OpenAI",
                provider="openai",
                model=model
            )
        
        model = request.model or self.default_model

        res = self.client.responses.create(
            model = model,
            input = request.messages,
        )

        return LLMResponse(
            content = res.output_text,
            provider="openai",
            model=model
        )