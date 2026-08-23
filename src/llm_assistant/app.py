from llm_assistant.llm import LLMError


class App:
    def __init__(self, llm, conversation):
        self.llm = llm
        self.conversation = conversation

    def ask(self, question: str) -> str:
        self.conversation.add_user_message(question)

        try:
            response = self.llm.generate(self.conversation.get_messages())
        except LLMError:
            self.conversation.remove_last_message()
            raise

        self.conversation.add_assistant_message(response)
        return response

    def clear(self) -> str:
        self.conversation.clear()
        return "conversation clear"
