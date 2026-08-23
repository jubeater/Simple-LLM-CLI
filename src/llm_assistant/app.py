class App:
    def __init__(self, llm, conversation):
        self.llm = llm
        self.conversation = conversation

    def ask(self, question: str) -> str:
        self.conversation.add_user_message(question)

        response = self.llm.generate(self.conversation.get_messages())

        if response.startswith("Error calling OpenAI API: "):
            self.conversation.remove_last_message()
        else:
            self.conversation.add_assistant_message(response)

        print(f"conversation: {self.conversation.messages}")
        return response

    def clear(self) -> str:
        self.conversation.clear()
        return "conversation clear"
