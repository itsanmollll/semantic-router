import os 
import openai
import unify
from typing import List, Optional
from semantic_router.llms import BaseLLM
from semantic_router.schema import Message
from semantic_router.utils.defaults import EncoderDefault
from semantic_router.utils.logger import logger


class unifyAi(BaseLLM):
    client: Optional[openai.OpenAI]
    async_client: Optional[openai.AsyncOpenAI]
    temperature: Optional[float]
    max_tokens: Optional[int]

    def __init__(
        self,
        name: Optional[str] = None,
        unifyAi_api_key: Optional[str] = None,
        base_url: str = "https://api.unifyai.com/v1",
        temperature: float = 0.01,
        max_tokens: int = 200,      
    ):
        if name is None:
            name = os.getenv(
                "UNIFYAI_CHAT_MODEL_NAME", "mistralai/mistral-7b-instruct"
            )
        super().__init__(name=name)
        self.base_url = base_url
        api_key = unifyAi_api_key or os.getenv("UNIFYAI_API_KEY")
        if api_key is None:
            raise ValueError("UnifyAi API key cannot be 'None'.")
        try:
            self.client = unify.OpenAI(api_key=api_key, base_url=self.base_url)
        except Exception as e:
            raise ValueError(
                f"UnifyAi API client failed to initialize. Error: {e}"
            ) from e
        self.temperature = temperature
        self.max_tokens = max_tokens

    def __call__(self, messages: List[Message]) -> str:
        if self.client is None:
            raise ValueError("UnifyAi client is not initialized.")
        try:
            completion = self.client.chat.completions.create(
                model=self.name,
                messages=[m.to_openai() for m in messages],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            output = completion.choices[0].message.content

            if not output:
                raise Exception("No output generated")
            return output
        except Exception as e:
            logger.error(f"LLM error: {e}")
            raise Exception(f"LLM error: {e}") from e
