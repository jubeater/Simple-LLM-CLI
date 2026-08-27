import logging

from llm_assistant.config import NETWORK_ERR_RETRY_THRESHOLD, LLMConfig
from llm_assistant.conversation import Conversation
from llm_assistant.errors import LLMError, LLMStreamError
from llm_assistant.llm import LLM
from llm_assistant.models import SessionMetrics, StreamResult, Usage

logger = logging.getLogger(__name__)


class App:
    def __init__(self, llm, conversation):
        self.llm = llm
        self.conversation = conversation
        self.session_metrics = SessionMetrics(0, 0, Usage(0, 0, 0), None, None)

    def ask(self, question: str):
        self.conversation.add_user_message(question)
        retry_count = 0
        while True:
            try:
                response_chunks = []
                messages = self.conversation.get_messages()
                for event in self.llm.generate_stream(messages):
                    if isinstance(event, str):
                        response_chunks.append(event)
                        yield event
                    elif isinstance(event, StreamResult):
                        self.session_metrics.request_count += 1
                        self.conversation.add_assistant_message(
                            "".join(response_chunks)
                        )
                        self.session_metrics.last_latency = event.total_duration
                        if self.session_metrics.total_latency == None:
                            self.session_metrics.total_latency = event.total_duration
                        elif event.total_duration != None:
                            self.session_metrics.total_latency += event.total_duration
                        self.session_metrics.total_usage = event.usage
                break
            except LLMError:
                self.conversation.remove_last_message()
                logger.exception(
                    "Removed failed question from conversation",
                    extra={"question": question},
                )
                self.session_metrics.error_count += 1
                raise
            except LLMStreamError:
                if retry_count >= NETWORK_ERR_RETRY_THRESHOLD:
                    self.conversation.remove_last_message()
                    self.session_metrics.error_count += 1
                    raise

                retry_count += 1
                logger.warning(
                    "LLM request failed, retrying",
                    extra={
                        "attempt": retry_count,
                        "max_retries": NETWORK_ERR_RETRY_THRESHOLD,
                    },
                )

    def clear(self):
        self.conversation.clear()

    def set_model_name(self, model_name) -> None:
        self.llm.set_model_name(model_name)

    def get_config(self) -> LLMConfig:
        return self.llm.get_config()

    def get_stats(self) -> SessionMetrics:
        return self.session_metrics

    def get_num_messages(self) -> int:
        return len(self.conversation.get_messages())


def create_app(config: LLMConfig) -> App:
    llm = LLM(config.model, config.max_output_tokens, config.temperature)
    conversation = Conversation()
    return App(llm, conversation)
