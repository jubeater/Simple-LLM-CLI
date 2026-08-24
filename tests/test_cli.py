import argparse
import logging
from unittest.mock import MagicMock, patch

import pytest

from llm_assistant import cli
from llm_assistant.llm import LLMError


class FakeApp:
    def __init__(self):
        self.questions = []
        self.cleared = False
        self.model = "model-a"
        self.llm = self

    def ask(self, question):
        self.questions.append(question)
        return "fake response"

    def clear(self):
        self.cleared = True

    def get_model_name(self):
        return self.model

    def set_model_name(self, model):
        self.model = model

    def get_max_output_token(self):
        return 1000

    def get_temperature(self):
        return 0.5


class FailingApp:
    def ask(self, _):
        raise LLMError("boom")


@pytest.fixture
def fake_app() -> FakeApp:
    return FakeApp()


@pytest.fixture
def fail_app() -> FailingApp:
    return FailingApp()


def test_main_requires_api_key(monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    cli.main()

    assert "Please set OPENAI API key" in capsys.readouterr().out


def test_parse_question():
    parser = cli.create_parser()

    args = parser.parse_args(["--question", "Hello"])

    assert args.question == "Hello"
    assert args.interactive is False


def test_parse_model_and_generation_options():
    parser = cli.create_parser()

    args = parser.parse_args(
        [
            "--model",
            "model-b",
            "--max_output_token",
            "500",
            "--temperature",
            "0.7",
        ]
    )

    assert args.model == "model-b"
    assert args.max_output_token == 500
    assert args.temperature == 0.7


def test_parse_args():
    parser = MagicMock()
    cli.parse_args(parser)


def test_load_config_from_environment(monkeypatch):
    monkeypatch.setenv("AI_MODEL", "env-model")
    monkeypatch.setenv("AI_MAX_OUTPUT_TOKEN", "2000")
    monkeypatch.setenv("AI_TEMPERATURE", "0.8")

    args = argparse.Namespace(
        model=None,
        max_output_token=None,
        temperature=None,
    )

    config = cli.load_config(args)

    assert config.model == "env-model"
    assert config.max_output_token == 2000
    assert config.temperature == 0.8


def test_cli_overrides_environment(monkeypatch):
    monkeypatch.setenv("AI_MODEL", "env-model")
    monkeypatch.setenv("AI_MAX_OUTPUT_TOKEN", "2000")
    monkeypatch.setenv("AI_TEMPERATURE", "0.8")

    args = argparse.Namespace(
        model="cli-model",
        max_output_token=1000,
        temperature=0.9,
    )

    config = cli.load_config(args)

    assert config.model == "cli-model"
    assert config.max_output_token == 1000
    assert config.temperature == 0.9


def test_using_default_with_invalid_environment(monkeypatch):
    monkeypatch.setenv("AI_MODEL", "env-model")
    monkeypatch.setenv("AI_MAX_OUTPUT_TOKEN", "invalidint")
    monkeypatch.setenv("AI_TEMPERATURE", "invalidfloat")

    args = argparse.Namespace(
        model="cli-model",
        max_output_token=None,
        temperature=None,
    )

    config = cli.load_config(args)

    assert config.model == "cli-model"
    assert config.max_output_token == cli.DEFAULT_MAX_TOKEN_LIMIT
    assert config.temperature == cli.DEFAULT_TEMPERATURE


def test_create_app():
    app = cli.create_app(cli.Config("cli-model", 150, 0.6))
    assert app.llm.get_model_name() == "cli-model"
    assert app.llm.get_max_output_token() == 150
    assert app.llm.get_temperature() == 0.6


def test_run_one_shot(fake_app, capsys):
    cli.run_one_shot(fake_app, "Hello")

    assert fake_app.questions == ["Hello"]

    captured = capsys.readouterr()
    assert "Assistant: fake response" in captured.out

    cli.run_one_shot(fake_app, None)
    captured = capsys.readouterr()
    assert "You need to give an" in captured.out


def test_run_one_shot_exception(fail_app, capsys):
    cli.run_one_shot(fail_app, "World")
    captured = capsys.readouterr()
    assert "Assistant error:" in captured.out


def test_run_interactive_question(monkeypatch, capsys):
    app = FakeApp()

    inputs = iter(
        [
            "What is TCP?",
            "/quit",
        ]
    )

    monkeypatch.setattr(
        "builtins.input",
        lambda _: next(inputs),
    )

    parser = cli.create_parser()

    cli.run_interactive(app, parser)  # type: ignore

    assert app.questions == ["What is TCP?"]

    captured = capsys.readouterr()
    assert "fake response" in captured.out


def test_run_interactive_empty_question(monkeypatch, capsys):
    app = FakeApp()

    inputs = iter(
        [
            "",
            "/quit",
        ]
    )

    monkeypatch.setattr(
        "builtins.input",
        lambda _: next(inputs),
    )

    parser = cli.create_parser()

    cli.run_interactive(app, parser)  # type: ignore

    captured = capsys.readouterr()
    assert "no input" in captured.out


def test_run_interactive_clear(monkeypatch):
    app = FakeApp()

    inputs = iter(
        [
            "/clear",
            "/quit",
        ]
    )

    monkeypatch.setattr(
        "builtins.input",
        lambda _: next(inputs),
    )

    cli.run_interactive(app, cli.create_parser())  # type: ignore

    assert app.cleared is True


def test_run_interactive_show_model(monkeypatch):
    app = FakeApp()

    inputs = iter(
        [
            "/model",
            "/quit",
        ]
    )

    monkeypatch.setattr(
        "builtins.input",
        lambda _: next(inputs),
    )

    cli.run_interactive(app, cli.create_parser())  # type: ignore

    assert app.get_model_name() == "model-a"


def test_run_interactive_switches_model(monkeypatch):
    app = FakeApp()

    inputs = iter(
        [
            "/model model-b",
            "/quit",
        ]
    )

    monkeypatch.setattr(
        "builtins.input",
        lambda _: next(inputs),
    )

    cli.run_interactive(app, cli.create_parser())  # type: ignore

    assert app.get_model_name() == "model-b"


def test_run_interactive_show_config(monkeypatch):
    app = FakeApp()

    inputs = iter(
        [
            "/config",
            "/quit",
        ]
    )

    monkeypatch.setattr(
        "builtins.input",
        lambda _: next(inputs),
    )

    cli.run_interactive(app, cli.create_parser())  # type: ignore

    assert app.get_model_name() == "model-a"


def test_run_interactive_help(monkeypatch, capsys):
    app = FakeApp()

    inputs = iter(
        [
            "/help",
            "/quit",
        ]
    )

    monkeypatch.setattr(
        "builtins.input",
        lambda _: next(inputs),
    )

    cli.run_interactive(app, cli.create_parser())  # type: ignore

    captured = capsys.readouterr()
    assert "usage:" in captured.out


def test_run_interactive_ctrl_c(monkeypatch, caplog):
    app = FakeApp()

    monkeypatch.setattr(
        "builtins.input",
        lambda _: (_ for _ in ()).throw(KeyboardInterrupt),
    )

    with caplog.at_level(logging.INFO):
        cli.run_interactive(app, cli.create_parser())  # type: ignore

    assert "Interactive session interrupted by user" in caplog.text


def test_run_interactive_ctrl_d(monkeypatch, caplog):
    app = FakeApp()

    monkeypatch.setattr(
        "builtins.input",
        lambda _: (_ for _ in ()).throw(EOFError),
    )

    with caplog.at_level(logging.INFO):
        cli.run_interactive(app, cli.create_parser())  # type: ignore

    assert "Interactive session ended by EOF" in caplog.text


def test_run_interactive_exception(monkeypatch, fail_app, capsys):
    inputs = iter(
        [
            "question",
            "/quit",
        ]
    )

    monkeypatch.setattr(
        "builtins.input",
        lambda _: next(inputs),
    )
    cli.run_interactive(fail_app, cli.create_parser())
    captured = capsys.readouterr()
    assert "Assistant error:" in captured.out


@patch("llm_assistant.cli.run_one_shot")
@patch("llm_assistant.cli.run_interactive")
@patch("llm_assistant.cli.create_app")
@patch("llm_assistant.cli.load_config")
@patch("llm_assistant.cli.parse_args")
@patch("llm_assistant.cli.create_parser")
@patch("llm_assistant.cli.configure_logging")
def test_main(
    mocklog,
    mockcreateparse,
    mockparseargs,
    mockloadconfig,
    mockcreateapp,
    mockinteractive,
    mockoneshot,
):
    cli.main()
    mockargs = MagicMock()
    mockargs.interactive = False
    mockparseargs.return_value = mockargs
    cli.main()
