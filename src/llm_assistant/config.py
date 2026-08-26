import logging
import os
from dataclasses import dataclass

from llm_assistant.errors import ConfigError

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_MAX_TOKEN_LIMIT = 1000
DEFAULT_TEMPERATURE = 1.0


@dataclass
class Config:
    model: str
    max_output_tokens: int
    temperature: float


def load_config(config: Config) -> Config:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ConfigError("OPENAI_API_KEY is not set")
    if config.model == None:
        model_name = os.getenv("AI_MODEL", DEFAULT_MODEL)
    else:
        model_name = config.model
    if config.max_output_tokens == None:
        try:
            max_output_tokens = os.getenv(
                "AI_MAX_OUTPUT_TOKEN", DEFAULT_MAX_TOKEN_LIMIT
            )
            max_output_tokens = int(max_output_tokens)
        except ValueError:
            logger.warning(
                "Invalid int for max token limit",
                extra={"value": max_output_tokens, "from": "environment variable"},
            )
            print("Invalid max_output_token get from environment, will use default")
            max_output_tokens = DEFAULT_MAX_TOKEN_LIMIT
    else:
        max_output_tokens = config.max_output_tokens
    if config.temperature == None:
        try:
            temperature = os.getenv("AI_TEMPERATURE", DEFAULT_TEMPERATURE)
            temperature = float(temperature)
        except ValueError:
            logger.warning(
                "Invalid float for temperature",
                extra={"value": temperature, "from": "environment variable"},
            )
            print("Invalid temperature get from environment, will use default")
            temperature = DEFAULT_TEMPERATURE
    else:
        temperature = config.temperature
    return Config(model_name, max_output_tokens, temperature)
