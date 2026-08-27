from unittest.mock import Mock

import pytest

from llm_assistant.app import App
from llm_assistant.conversation import Conversation
from llm_assistant.llm import LLMError, LLMStreamError
from llm_assistant.models import StreamResult, Usage


def test_ask_success():
    mock_llm = Mock()
    mock_llm.generate_stream.return_value = iter(
        [
            "Hello",
            " world",
            StreamResult(
                usage=Usage(10, 5, 15),
                total_duration=0.5,
            ),
        ]
    )

    conversation = Conversation()
    app = App(mock_llm, conversation)

    result = list(app.ask("Hi"))
    mock_llm.generate_stream.assert_called_with([conversation.get_messages()[-2]])

    assert result[:2] == ["Hello", " world"]

    assert conversation.get_messages()[-1] == {
        "role": "assistant",
        "content": "Hello world",
    }

    assert app.session_metrics.request_count == 1
    assert app.session_metrics.error_count == 0
    assert app.session_metrics.last_latency == 0.5
    assert app.session_metrics.total_usage == Usage(10, 5, 15)


def test_ask_retries():
    mock_llm = Mock()

    failed_stream = Mock()
    failed_stream.__iter__ = Mock(side_effect=LLMStreamError("network error"))

    successful_stream = iter(
        [
            "Hello",
            StreamResult(
                usage=Usage(10, 5, 15),
                total_duration=0.5,
            ),
        ]
    )

    mock_llm.generate_stream.side_effect = [
        failed_stream,
        successful_stream,
    ]

    app = App(mock_llm, Conversation())

    result = list(app.ask("Hi"))

    assert result == ["Hello"]
    assert mock_llm.generate_stream.call_count == 2
    assert app.session_metrics.request_count == 1
    assert app.session_metrics.error_count == 0


def test_ask_stream_failure_does_not_commit_partial_response():
    mock_llm = Mock()

    def failed_stream(_):
        yield "Hello"
        yield " world"
        raise LLMStreamError("connection lost")

    mock_llm.generate_stream.side_effect = failed_stream

    conversation = Conversation()
    app = App(mock_llm, conversation)

    with pytest.raises(LLMStreamError):
        list(app.ask("Hi"))

    assert conversation.get_messages() == []

    assert app.session_metrics.request_count == 0
    assert app.session_metrics.error_count == 1


def test_clear():
    llm = Mock()
    conversation = Mock()
    app = App(llm, conversation)
    app.ask("What is TCP?")
    app.ask("What is TCP?")
    app.clear()
    conversation.clear.assert_called_once_with()
