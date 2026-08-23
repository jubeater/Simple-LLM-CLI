from unittest.mock import MagicMock

import pytest

from llm_assistant.app import App
from llm_assistant.conversation import Conversation
from llm_assistant.llm import LLMError


class FakeLLM:
    def __init__(self, response: str = "fake response"):
        self.response = response
        self.calls = []

    def generate(self, messages):
        self.calls.append(messages)
        return self.response


@pytest.fixture
def app() -> App:
    return App(FakeLLM(), Conversation())


def test_ask_calls_llm(app):
    app.ask("What is TCP?")

    assert len(app.llm.calls) == 1


def test_ask_reraises_llm_error_and_removes_user_message():
    llm = MagicMock()
    llm.generate.side_effect = LLMError("Unable to get a response from OpenAI")
    app = App(llm, Conversation())

    with pytest.raises(LLMError, match="Unable to get a response from OpenAI"):
        app.ask("What is TCP?")

    assert app.conversation.get_messages() == []


def test_clear(app):
    app.ask("What is TCP?")
    app.ask("What is TCP?")
    app.clear()
    assert app.conversation.get_messages() == []

