from openai import APIError, OpenAI


# only support openAI now
class LLM:
    def __init__(self) -> None:
        self.client = OpenAI()

    def generate(self, conversation) -> str:
        try:
            # https://developers.openai.com/api/docs/guides/conversation-state
            # Generate text with messages using different roles -> ["user", "assistant"]
            response = self.client.responses.create(model="gpt-5.6", input=conversation)
            return response.output_text
        except APIError as e:
            return f"Error calling OpenAI API: {e}"
