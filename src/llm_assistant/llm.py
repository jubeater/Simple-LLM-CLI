import logging
import time

from openai import APIError, APIStatusError, APITimeoutError, OpenAI

from llm_assistant.config import LLMConfig
from llm_assistant.errors import LLMError

logger = logging.getLogger(__name__)


# only support openAI now
class LLM:
    def __init__(
        self, model_name: str, max_output_token: int, temperature: float
    ) -> None:
        self.client = OpenAI()
        self.model_name = model_name
        self.max_output_token = max_output_token
        self.temperature = temperature

    def set_model_name(self, model_name: str) -> None:
        self.model_name = model_name

    def get_config(self) -> LLMConfig:
        return LLMConfig(self.model_name, self.max_output_token, self.temperature)

    def generate(self, messages) -> str:
        try:
            # https://developers.openai.com/api/docs/guides/conversation-state
            # Generate text with messages using different roles -> ["user", "assistant"]
            logger.info("LLM request started", extra={"model": self.model_name})
            start = time.perf_counter()
            response = self.client.responses.create(
                model=self.model_name,
                input=messages,
                max_output_tokens=self.max_output_token,
                temperature=self.temperature,
            )
            duration = time.perf_counter() - start
            logger.info(
                "LLM request completed",
                extra={"model": self.model_name, "duration": duration},
            )
            if not response.output_text or not response.output_text.strip():
                raise LLMError("OpenAI return empty response body")
            return response.output_text
        except (APITimeoutError, APIStatusError, APIError) as error:
            logger.exception("LLM request failed")
            raise LLMError("Unable to get a response from OpenAI") from error
