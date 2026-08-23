from unittest.mock import MagicMock

from llm_assistant import cli
from llm_assistant.llm import LLMError


def test_main_requires_api_key(monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    cli.main()

    assert "Please set OPENAI API key" in capsys.readouterr().out


def test_main_give_invalid_token_limit(monkeypatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AI_MAX_OUTPUT_TOKENS", "invalidint")
    monkeypatch.setattr("sys.argv", ["ai"])
    monkeypatch.setattr(cli, "LLM", MagicMock())

    cli.main()

    assert "Invalid max_output_token" in capsys.readouterr().out


def test_main_give_invalid_temperature(monkeypatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AI_TEMPERATURE", "invalidfloat")
    monkeypatch.setattr("sys.argv", ["ai"])
    monkeypatch.setattr(cli, "LLM", MagicMock())

    cli.main()

    assert "Invalid temperature" in capsys.readouterr().out


def test_main_requires_question(monkeypatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("sys.argv", ["ai"])
    monkeypatch.setattr(cli, "LLM", MagicMock())

    cli.main()

    assert "you need to give an input question" in capsys.readouterr().out


def test_main_user_provide_config(monkeypatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        "sys.argv",
        ["ai", "--question", "Hello", "-m", "fakeModel", "-mot", "150", "-t", "0.8"],
    )

    mock_llm = MagicMock()
    mock_app = MagicMock()
    mock_app.ask.return_value = "Hi there"
    monkeypatch.setattr(cli, "LLM", MagicMock(return_value=mock_llm))
    monkeypatch.setattr(cli, "App", MagicMock(return_value=mock_app))

    cli.main()

    mock_app.ask.assert_called_once_with("Hello")
    assert "Assistant: Hi there" in capsys.readouterr().out


def test_main_user_provide_invalid_config(monkeypatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        "sys.argv",
        [
            "ai",
            "--question",
            "Hello",
            "-m",
            "fakeModel",
            "-mot",
            "invalidint",
            "-t",
            "invalidfloat",
        ],
    )

    mock_llm = MagicMock()
    mock_app = MagicMock()
    mock_app.ask.return_value = "Hi there"
    monkeypatch.setattr(cli, "LLM", MagicMock(return_value=mock_llm))
    monkeypatch.setattr(cli, "App", MagicMock(return_value=mock_app))

    cli.main()

    mock_app.ask.assert_called_once_with("Hello")
    out = capsys.readouterr().out
    assert "Max token input is not int" in out
    assert "Temperature input is not float" in out
    assert "Assistant: Hi there" in out


def test_main_handles_llm_error(monkeypatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("sys.argv", ["ai", "--question", "Hello"])
    monkeypatch.setattr("builtins.input", lambda _: "/quit")

    mock_app = MagicMock()
    mock_app.ask.side_effect = LLMError("request failed")
    monkeypatch.setattr(cli, "LLM", MagicMock())
    monkeypatch.setattr(cli, "App", MagicMock(return_value=mock_app))

    cli.main()

    assert "Assistant error: request failed" in capsys.readouterr().out


def test_main_interactive_commands(monkeypatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("sys.argv", ["ai", "--interactive"])

    inputs = iter(
        [
            "",
            "/model",
            "/model new-model",
            "/config",
            "/help",
            "/clear",
            "/quit",
        ]
    )
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    mock_llm = MagicMock()
    mock_llm.get_model_name.return_value = "modelA"
    mock_app = MagicMock()
    monkeypatch.setattr(cli, "LLM", MagicMock(return_value=mock_llm))
    monkeypatch.setattr(cli, "App", MagicMock(return_value=mock_app))

    cli.main()

    mock_llm.set_model_name.assert_called_once_with("new-model")
    mock_app.clear.assert_called_once_with()
    output = capsys.readouterr().out
    assert "no input" in output
    assert "Current model is: modelA" in output
    assert "Model switched to new-model" in output
    assert "Configuration" in output
    assert "usage: uv run main.py" in output
    assert "Clear conversation history" in output


def test_main_interactive_question_and_llm_error(monkeypatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("sys.argv", ["ai", "--interactive"])

    inputs = iter(["Hello", "How are you?", "/quit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    mock_llm = MagicMock()
    mock_app = MagicMock()
    mock_app.ask.side_effect = ["Hi there", LLMError("request failed")]
    monkeypatch.setattr(cli, "LLM", MagicMock(return_value=mock_llm))
    monkeypatch.setattr(cli, "App", MagicMock(return_value=mock_app))

    cli.main()

    assert mock_app.ask.call_args_list[0].args == ("Hello",)
    assert mock_app.ask.call_args_list[1].args == ("How are you?",)
    output = capsys.readouterr().out
    assert "Assistant: Hi there" in output
    assert "Assistant error: request failed" in output


def test_main_interactive_ctrl_c_exits(monkeypatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("sys.argv", ["ai", "--interactive"])

    def input_with_ctrl_c(_):
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", input_with_ctrl_c)
    monkeypatch.setattr(cli, "LLM", MagicMock())
    monkeypatch.setattr(cli, "App", MagicMock())

    cli.main()

    assert "Enter your questions" in capsys.readouterr().out


def test_main_interactive_ctrl_d_exits(monkeypatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("sys.argv", ["ai", "--interactive"])

    def input_with_ctrl_d(_):
        raise EOFError

    monkeypatch.setattr("builtins.input", input_with_ctrl_d)
    monkeypatch.setattr(cli, "LLM", MagicMock())
    monkeypatch.setattr(cli, "App", MagicMock())

    cli.main()

    assert "Enter your questions" in capsys.readouterr().out
