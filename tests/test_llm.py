from unittest.mock import MagicMock, patch

import pytest
from openai import APIError, APIStatusError, APITimeoutError

from llm_assistant.llm import LLM, LLMError


@pytest.fixture
@patch("llm_assistant.llm.OpenAI")
def llm(mock_openai) -> LLM:
    mock_client = MagicMock()
    mock_openai.return_value = mock_client

    return LLM("modelA", 100, 0.3)


def test_get_model_name(llm):
    assert llm.get_model_name() == "modelA"


def test_get_max_output_token(llm):
    assert llm.get_max_output_token() == 100


def test_get_temperature(llm):
    assert llm.get_temperature() == 0.3


def test_set_model_name(llm):
    llm.set_model_name("modelB")
    assert llm.model_name == "modelB"


def test_generate(llm):
    mock_response = MagicMock()
    mock_response.output_text = "mock answer from LLM"
    mocked_client = llm.client
    mocked_client.responses.create.return_value = mock_response

    answer = llm.generate([{"role": "user", "content": "Hello"}])
    assert answer == "mock answer from LLM"

    mock_response.output_text = ""
    with pytest.raises(LLMError, match="OpenAI return empty response body"):
        llm.generate([{"role": "user", "content": "Hello"}])


def test_generate_raises_llm_error_for_openai_errors(llm):
    mocked_client = llm.client

    mocked_client.responses.create.side_effect = APITimeoutError(request=MagicMock())
    with pytest.raises(LLMError, match="Unable to get a response from OpenAI"):
        llm.generate([{"role": "user", "content": "Hello"}])

    mocked_client.responses.create.side_effect = APIStatusError(
        "mock status error",
        response=MagicMock(),
        body=None,
    )
    with pytest.raises(LLMError, match="Unable to get a response from OpenAI"):
        llm.generate([{"role": "user", "content": "Hello"}])

    mocked_client.responses.create.side_effect = APIError(
        "mock api error",
        request=MagicMock(),
        body=None,
    )
    with pytest.raises(LLMError, match="Unable to get a response from OpenAI"):
        llm.generate([{"role": "user", "content": "Hello"}])
