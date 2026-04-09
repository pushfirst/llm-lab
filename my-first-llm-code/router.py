from base import LLMRequest
from providers.openai import OpenAIProvider
from providers.ollama import OllamaProvider


class LLMRouter:
    def __init__(self):
        self.openai = OpenAIProvider()
        self.ollama = OllamaProvider()

    def route(self, request: LLMRequest):
        local_res = self.ollama.generate(request)
        print('local_response', local_res)

        judge_prompt = """
        Evaluate if this answer is correct and sufficient:
        {local_res.content}

        Answer only YES or NO.
        """

        judge_req = LLMRequest(messages=[{"role": "user", "content": judge_prompt}])
        print('judge_req', judge_req)
        verdict = self.openai.generate(judge_req).content
        print('Verdict', verdict)

        if "NO" in verdict:
            print('Verdict No')
            return self.openai.generate(request)

        return local_res