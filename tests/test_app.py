from unittest.mock import Mock

import pytest

from llm_assistant.app import App
from llm_assistant.llm import LLMError


def test_ask_calls_llm():
    llm = Mock()
    conversation = Mock()

    conversation.get_messages.return_value = [{"role": "user", "content": "Hello"}]
    llm.generate.return_value = "Hi!"

    app = App(llm, conversation)

    result = app.ask("Hello")

    assert result == "Hi!"

    llm.generate.assert_called_once_with([{"role": "user", "content": "Hello"}])


def test_ask_reraises_llm_error_and_removes_user_message():
    llm = Mock()
    conversation = Mock()
    llm.generate.side_effect = LLMError("Unable to get a response from OpenAI")
    app = App(llm, conversation)

    with pytest.raises(LLMError, match="Unable to get a response from OpenAI"):
        app.ask("What is TCP?")
        conversation.remove_last_message.assert_called_once_with()


def test_clear():
    llm = Mock()
    conversation = Mock()
    app = App(llm, conversation)
    app.ask("What is TCP?")
    app.ask("What is TCP?")
    app.clear()
    conversation.clear.assert_called_once_with()
