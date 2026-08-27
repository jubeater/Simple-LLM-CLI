import pytest

from llm_assistant.config import (
    DEFAULT_MAX_TOKEN_LIMIT,
    DEFAULT_TEMPERATURE,
    UserConfig,
    load_config,
)
from llm_assistant.errors import ConfigError


def test_no_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY")
    user_config = UserConfig(
        model=None,
        max_output_tokens=None,
        temperature=None,
    )
    with pytest.raises(ConfigError, match="OPENAI_API_KEY is not set"):
        load_config(user_config)


def test_load_config_from_environment(monkeypatch):
    monkeypatch.setenv("AI_MODEL", "env-model")
    monkeypatch.setenv("AI_MAX_OUTPUT_TOKEN", "2000")
    monkeypatch.setenv("AI_TEMPERATURE", "0.8")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")

    user_config = UserConfig(
        model=None,
        max_output_tokens=None,
        temperature=None,
    )

    config = load_config(user_config)

    assert config.model == "env-model"
    assert config.max_output_tokens == 2000
    assert config.temperature == 0.8


def test_cli_overrides_environment(monkeypatch):
    monkeypatch.setenv("AI_MODEL", "env-model")
    monkeypatch.setenv("AI_MAX_OUTPUT_TOKEN", "2000")
    monkeypatch.setenv("AI_TEMPERATURE", "0.8")

    user_config = UserConfig(
        model="cli-model",
        max_output_tokens=1000,
        temperature=0.9,
    )

    config = load_config(user_config)

    assert config.model == "cli-model"
    assert config.max_output_tokens == 1000
    assert config.temperature == 0.9


def test_using_default_with_invalid_environment(monkeypatch):
    monkeypatch.setenv("AI_MODEL", "env-model")
    monkeypatch.setenv("AI_MAX_OUTPUT_TOKEN", "invalidint")
    monkeypatch.setenv("AI_TEMPERATURE", "invalidfloat")

    user_config = UserConfig(
        model="cli-model",
        max_output_tokens=None,
        temperature=None,
    )

    config = load_config(user_config)

    assert config.model == "cli-model"
    assert config.max_output_tokens == DEFAULT_MAX_TOKEN_LIMIT
    assert config.temperature == DEFAULT_TEMPERATURE
