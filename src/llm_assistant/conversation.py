class Conversation:
    def __init__(self, system_prompt: str | None = None):
        self.messages = [{"role": "fake", "content": "fake conversation"}]

        if system_prompt:
            self.messages.append({
                "role": "system",
                "content": system_prompt,
            })

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
        self.messages.pop()

    def get_messages(self) -> list[dict[str, str]]:
        return self.messages.copy()

    def clear(self) -> None:
        self.messages.clear()
