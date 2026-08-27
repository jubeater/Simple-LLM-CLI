from unittest.mock import Mock, patch

import pytest
from openai import APIError

from llm_assistant.errors import LLMError, LLMStreamError
from llm_assistant.llm import LLM
from llm_assistant.models import StreamResult, Usage


@patch("llm_assistant.llm.OpenAI")
def test_set_model_name(_):
    llm = LLM("modelA", 100, 1.0)
    llm.set_model_name("modelB")
    assert llm.model_name == "modelB"


@patch("llm_assistant.llm.OpenAI")
def test_generate(openai):
    events = [
        Mock(type="response.output_text.delta", delta="Hello"),
        Mock(type="response.output_text.delta", delta=" world"),
        Mock(
            type="response.completed",
            response=Mock(
                usage=Mock(
                    input_tokens=10,
                    output_tokens=5,
                    total_tokens=15,
                )
            ),
        ),
    ]
    openai.return_value.responses.create.return_value = events
    llm = LLM("modelA", 100, 1.0)
    results = list(llm.generate_stream([{"role": "user", "content": "Hi"}]))
    assert results[0] == "Hello"
    assert results[1] == " world"

    stream_result = results[2]

    assert isinstance(stream_result, StreamResult)
    assert stream_result.usage.input_tokens == 10
    assert stream_result.usage.output_tokens == 5
    assert stream_result.usage.total_tokens == 15
    assert stream_result.total_duration >= 0  # type: ignore


@patch("llm_assistant.llm.OpenAI")
def test_generate_missing_usage(openai):
    events = [
        Mock(type="response.output_text.delta", delta="Hello"),
        Mock(
            type="response.completed",
            response=Mock(usage=None),
        ),
    ]
    openai.return_value.responses.create.return_value = events
    llm = LLM("modelA", 100, 1.0)
    messages = [{"role": "user", "content": "Hi"}]
    results = list(llm.generate_stream(messages))

    result = results[-1]

    assert result.usage == Usage(None, None, None)  # type: ignore


@patch("llm_assistant.llm.OpenAI")
def test_generate_streaming_llm_err(openai):
    events = [
        Mock(type="response.output_text.delta", delta="Hello"),
        Mock(type="error"),
    ]
    openai.return_value.responses.create.return_value = events
    llm = LLM("modelA", 100, 1.0)
    messages = [{"role": "user", "content": "Hi"}]
    with pytest.raises(LLMStreamError):
        list(llm.generate_stream(messages))


@patch("llm_assistant.llm.OpenAI")
def test_generate_llm_err(openai):
    openai.return_value.responses.create.side_effect = APIError(
        "llm error happened", request=Mock(), body=None
    )
    llm = LLM("modelA", 100, 1.0)
    messages = [{"role": "user", "content": "Hi"}]
    with pytest.raises(LLMError):
        list(llm.generate_stream(messages))
