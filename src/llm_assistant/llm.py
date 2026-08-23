from openai import APIError, OpenAI


def call_llm_assistant(question: str, platform: str = "openai") -> str:
    if platform == "openai":
        try:
            client = OpenAI()
            response = client.responses.create(model="gpt-5.6", input=question)
            return response.output_text
        except APIError as e:
            return f"Error calling OpenAI API: {e}"
    else:
        raise ValueError("Unsupported platform")
