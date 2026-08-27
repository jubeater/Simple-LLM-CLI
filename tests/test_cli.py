import logging
from unittest.mock import Mock, patch

from llm_assistant import cli
from llm_assistant.errors import ConfigError, LLMError


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
            "--max_output_tokens",
            "500",
            "--temperature",
            "0.7",
        ]
    )

    assert args.model == "model-b"
    assert args.max_output_tokens == 500
    assert args.temperature == 0.7


def test_run_one_shot(capsys):
    fake_app = Mock()
    cli.run_one_shot(fake_app, "Hello")

    fake_app.ask.assert_called_once_with("Hello")

    captured = capsys.readouterr()
    assert "Assistant: " in captured.out

    cli.run_one_shot(fake_app, None)
    captured = capsys.readouterr()
    assert "You need to give an input question." in captured.out


def test_run_one_shot_exception(capsys):
    fail_app = Mock()
    fail_app.ask.side_effect = LLMError("Unable to get a response from OpenAI")
    cli.run_one_shot(fail_app, "World")
    captured = capsys.readouterr()
    assert "Assistant error:" in captured.out


def test_run_interactive_question(monkeypatch, capsys):
    app = Mock()
    app.ask.return_value = "fake response"
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

    cli.run_interactive(app, parser)

    app.ask.assert_called_once_with("What is TCP?")

    captured = capsys.readouterr()
    assert "fake response" in captured.out


def test_run_interactive_empty_question(monkeypatch, capsys):
    app = Mock()

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

    cli.run_interactive(app, parser)

    captured = capsys.readouterr()
    assert "no input" in captured.out


def test_run_interactive_clear(monkeypatch):
    app = Mock()

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

    cli.run_interactive(app, cli.create_parser())

    app.clear.assert_called_once_with()


def test_run_interactive_show_model(monkeypatch):
    app = Mock()

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

    cli.run_interactive(app, cli.create_parser())
    app.get_config.assert_called_once_with()


def test_run_interactive_switches_model(monkeypatch):
    app = Mock()

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

    app.set_model_name.assert_called_once_with("model-b")


def test_run_interactive_show_config(monkeypatch):
    app = Mock()

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

    app.get_config.assert_called_once_with()


def test_run_interactive_help(monkeypatch, capsys):
    app = Mock()
    parser = Mock()
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

    cli.run_interactive(app, parser)

    parser.print_help.assert_called_once_with()


def test_run_interactive_ctrl_c(monkeypatch, caplog):
    app = Mock()

    monkeypatch.setattr(
        "builtins.input",
        lambda _: (_ for _ in ()).throw(KeyboardInterrupt),
    )

    with caplog.at_level(logging.INFO):
        cli.run_interactive(app, cli.create_parser())

    assert "Interactive session interrupted by user" in caplog.text


def test_run_interactive_ctrl_d(monkeypatch, caplog):
    app = Mock()

    monkeypatch.setattr(
        "builtins.input",
        lambda _: (_ for _ in ()).throw(EOFError),
    )

    with caplog.at_level(logging.INFO):
        cli.run_interactive(app, cli.create_parser())

    assert "Interactive session ended by EOF" in caplog.text


def test_run_interactive_exception(monkeypatch, capsys):
    fail_app = Mock()
    fail_app.ask.side_effect = LLMError("Unable to get a response from OpenAI")
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
@patch("llm_assistant.cli.create_parser")
def test_main(
    mockcreateparse, mockloadconfig, mockcreateapp, mockinteractive, mockoneshot, capsys
):
    parser = Mock()
    user_args = Mock()
    user_args.model = "fake-modelA"
    user_args.interactive = False
    parser.parse_args.return_value = user_args
    mockcreateparse.return_value = parser

    cli.main()
    captured = capsys.readouterr()
    assert "Using the model" in captured.out
    mockcreateapp.assert_called_once()
    mockoneshot.assert_called_once()

    user_args.interactive = True
    cli.main()
    mockcreateapp.assert_called()
    mockinteractive.assert_called()

    mockloadconfig.side_effect = ConfigError("no key provided!")
    cli.main()
    captured = capsys.readouterr()
    assert "Configuration error:" in captured.out
