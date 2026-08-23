class Conversation:
    def __init__(self, system_prompt: str | None = None):
        self.messages = []

    def add_user_message(self, content: str) -> None:
        self.messages.append({
            "role": "user",
            "content": content,
        })

    def add_assistant_message(self, content: str) -> None:
        self.messages.append({
            "role": "assistant",
            "content": content,
        })

    def remove_last_message(self) -> None:
        if len(self.messages) > 0:
            self.messages.pop()

    def get_messages(self) -> list[dict[str, str]]:
        return self.messages.copy()

    def clear(self) -> None:
        self.messages.clear()
