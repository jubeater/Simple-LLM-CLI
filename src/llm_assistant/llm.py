import logging
import time

from openai import APIError, APIStatusError, APITimeoutError, OpenAI

from llm_assistant.config import LLMConfig
from llm_assistant.errors import LLMError, LLMStreamError
from llm_assistant.models import StreamResult, Usage

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

    def generate_stream(self, messages):
        try:
            # https://developers.openai.com/api/docs/guides/conversation-state
            # Generate text with messages using different roles -> ["user", "assistant"]
            logger.info("LLM request started", extra={"model": self.model_name})
            start = time.perf_counter()
            stream = self.client.responses.create(
                model=self.model_name,
                input=messages,
                max_output_tokens=self.max_output_token,
                temperature=self.temperature,
                stream=True,
            )
            for event in stream:
                match event.type:
                    case "response.output_text.delta":
                        yield event.delta
                    case "response.completed":
                        total_duration = time.perf_counter() - start
                        response_usage = event.response.usage

                        if response_usage is None:
                            usage = Usage(None, None, None)
                        else:
                            usage = Usage(
                                response_usage.input_tokens,
                                response_usage.output_tokens,
                                response_usage.total_tokens,
                            )
                        yield StreamResult(
                            usage=usage,
                            total_duration=total_duration,
                        )
                    case "error":
                        logger.error("LLM streaming error")
                        raise LLMStreamError("LLM Streaming error")
                    case _:
                        continue
        except (APITimeoutError, APIStatusError, APIError) as error:
            logger.exception("Unable to get a response from OpenAI")
            raise LLMError("Unable to get a response from OpenAI") from error
