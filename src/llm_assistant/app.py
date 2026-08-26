import logging

from llm_assistant.cli import Config
from llm_assistant.conversation import Conversation
from llm_assistant.errors import LLMError
from llm_assistant.llm import LLM

logger = logging.getLogger(__name__)


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
            logger.info(
                "Removed failed question from conversation",
                extra={"question": question},
            )
            raise

        self.conversation.add_assistant_message(response)
        return response

    def clear(self):
        self.conversation.clear()

    def set_model_name(self, model_name) -> None:
        self.llm.set_model_name(model_name)

    def get_config(self) -> Config:
        return self.llm.get_config()


def create_app(config: Config) -> App:
    llm = LLM(config.model, config.max_output_token, config.temperature)
    conversation = Conversation()
    return App(llm, conversation)
