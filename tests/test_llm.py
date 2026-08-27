from unittest.mock import Mock, patch

import pytest
from openai import APIError, APIStatusError, APITimeoutError

from llm_assistant.errors import LLMError
from llm_assistant.llm import LLM


@patch("llm_assistant.llm.OpenAI")
def test_set_model_name(_):
    llm = LLM("modelA", 100, 1.0)
    llm.set_model_name("modelB")
    assert llm.model_name == "modelB"


@patch("llm_assistant.llm.OpenAI")
def test_generate(openai):
    mock_response = Mock()
    mock_response.output_text = "mock answer from LLM"
    openai.return_value.responses.create.return_value = mock_response
    llm = LLM("modelA", 100, 1.0)
    answer = llm.generate([{"role": "user", "content": "Hello"}])
    assert answer == "mock answer from LLM"

    mock_response.output_text = ""
    with pytest.raises(LLMError, match="OpenAI return empty response body"):
        llm.generate([{"role": "user", "content": "Hello"}])


@patch("llm_assistant.llm.OpenAI")
def test_generate_raises_llm_error_for_openai_errors(openai):
    openai.return_value.responses.create.side_effect = APITimeoutError(request=Mock())
    llm = LLM("modelA", 100, 1.0)
    with pytest.raises(LLMError, match="Unable to get a response from OpenAI"):
        llm.generate([{"role": "user", "content": "Hello"}])

    openai.return_value.responses.create.side_effect = APIStatusError(
        "mock status error",
        response=Mock(),
        body=None,
    )
    with pytest.raises(LLMError, match="Unable to get a response from OpenAI"):
        llm.generate([{"role": "user", "content": "Hello"}])

    openai.return_value.responses.create.side_effect = APIError(
        "mock api error",
        request=Mock(),
        body=None,
    )
    with pytest.raises(LLMError, match="Unable to get a response from OpenAI"):
        llm.generate([{"role": "user", "content": "Hello"}])
