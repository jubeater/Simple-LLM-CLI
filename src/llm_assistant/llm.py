from openai import APIError, APIStatusError, APITimeoutError, OpenAI


class LLMError(Exception):
    pass


# only support openAI now
class LLM:
    def __init__(self, model_name: str, max_output_token: int, temperature: float) -> None:
        self.client = OpenAI()
        self.model_name = model_name
        self.max_output_token = max_output_token
        self.temperature = temperature

    def get_model_name(self) -> str:
        return self.model_name

    def set_model_name(self, model_name: str) -> None:
        self.model_name = model_name

    def generate(self, conversation) -> str | None:
        try:
            # https://developers.openai.com/api/docs/guides/conversation-state
            # Generate text with messages using different roles -> ["user", "assistant"]
            response = self.client.responses.create(
                model=self.model_name,
                input=conversation,
                max_output_tokens=self.max_output_token,
                temperature=self.temperature
            )
            if not response.output_text or not response.output_text.strip():
                raise LLMError("OpenAI return empty response body")
            return response.output_text
        except (APITimeoutError, APIStatusError, APIError) as error:
            raise LLMError("Unable to get a response from OpenAI") from error
