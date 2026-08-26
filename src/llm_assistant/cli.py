import argparse
import logging
import sys

from llm_assistant.app import App, create_app
from llm_assistant.config import Config, load_config
from llm_assistant.errors import ConfigError, LLMError


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="uv run ai",
        description="Provide an interactive shell for LLM Assistant.",
    )
    parser.add_argument(
        "-q", "--question", help="Question for LLM, no conversation stored."
    )
    parser.add_argument(
        "-i", "--interactive", action="store_true", help="Enter interactive mode."
    )
    parser.add_argument("-m", "--model", help="Set model of LLM.")
    parser.add_argument(
        "-mot", "--max_output_tokens", type=int, help="Set max output token limit."
    )
    parser.add_argument(
        "-t", "--temperature", type=float, help="Set temperature of the answer."
    )

    return parser


def parse_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    args = parser.parse_args()
    if args.model is not None:
        print(f"Using the model {args.model} to answer now...")
    return args


def run_one_shot(app: App, question: str | None) -> None:
    if not question:
        print("You need to give an input question")
        return

    try:
        answer = app.ask(question)
        print(f"Assistant: {answer}")
    except LLMError as error:
        print(f"Assistant error: {error}")


def run_interactive(app: App, parser: argparse.ArgumentParser) -> None:
    print("Enter your questions. \n'/help' for info, '/clear' to reset conversation")
    print("'/model' show current model id, '/model[space][model_id]' to switch model.")
    print("'/config' show configuration, /quit' to quit")
    while True:
        try:
            user_input = input("You: ").strip()
        except KeyboardInterrupt:
            # Ctrl-c clears the input
            sys.stdout.write("\n")
            logger.info("Interactive session interrupted by user")
            break
        except EOFError:
            # Ctrl-d exits
            logger.info("Interactive session ended by EOF")
            sys.stdout.write("\n")
            break

        if user_input == "":
            print("no input\n")

        elif user_input == "/model":
            cur_config = app.get_config()
            print(f"Current model is: {cur_config.model}")

        elif user_input.startswith("/model "):
            input_model = user_input[len("/model ") :].strip()
            app.set_model_name(input_model)
            print(f"Model switched to {input_model}")

        elif user_input == "/config":
            cur_config = app.get_config()
            print("Configuration")
            print("---------------------------------------")
            print(f"Model:            {cur_config.model}")
            print(f"Max output token: {cur_config.max_output_tokens}")
            print(f"Temperature:      {cur_config.temperature}")

        elif user_input == "/help":
            parser.print_help()

        elif user_input == "/clear":
            app.clear()
            print("Clear conversation history\n")

        elif user_input == "/quit":
            break

        else:
            try:
                answer = app.ask(user_input)
                print(f"Assistant: {answer}\n")
            except LLMError as error:
                print(f"Assistant error: {error}")


def main() -> int:
    configure_logging()
    parser = create_parser()
    args = parse_args(parser)
    user_config = Config(args.model, args.temperature, args.max_output_tokens)
    try:
        app_config = load_config(user_config)
    except ConfigError as e:
        print(f"Configuration error: {e}")
        return 1
    app = create_app(app_config)

    if args.interactive:
        run_interactive(app, parser)
    else:
        run_one_shot(app, args.question)
    return 0


if __name__ == "__main__":
    main()
