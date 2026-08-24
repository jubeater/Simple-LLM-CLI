import argparse
import logging
import os
import sys
from dataclasses import dataclass

from llm_assistant.app import App
from llm_assistant.conversation import Conversation
from llm_assistant.llm import LLM, LLMError

DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_MAX_TOKEN_LIMIT = 1000
DEFAULT_TEMPERATURE = 1.0


@dataclass
class Config:
    model: str
    max_output_token: int
    temperature: float


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
        "-mot", "--max_output_token", type=int, help="Set max output token limit."
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


def load_config(args: argparse.Namespace) -> Config:
    if args.model == None:
        model_name = os.getenv("AI_MODEL", DEFAULT_MODEL)
    else:
        model_name = args.model
    if args.max_output_token == None:
        try:
            max_output_token = os.getenv("AI_MAX_OUTPUT_TOKEN", DEFAULT_MAX_TOKEN_LIMIT)
            max_output_token = int(max_output_token)
        except ValueError:
            logger.warning(
                "Invalid int for max token limit",
                extra={"value": max_output_token, "from": "environment variable"},
            )
            print("Invalid max_output_token get from environment, will use default")
            max_output_token = DEFAULT_MAX_TOKEN_LIMIT
    else:
        max_output_token = args.max_output_token
    if args.temperature == None:
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
        temperature = args.temperature
    return Config(model_name, max_output_token, temperature)


def create_app(config: Config) -> App:
    llm = LLM(config.model, config.max_output_token, config.temperature)
    conversation = Conversation()
    return App(llm, conversation)


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
            print(f"Current model is: {app.llm.get_model_name()}")

        elif user_input.startswith("/model "):
            input_model = user_input[len("/model ") :].strip()
            # little hack on llm
            app.llm.set_model_name(input_model)
            print(f"Model switched to {input_model}")

        elif user_input == "/config":
            print("Configuration")
            print("---------------------------------------")
            print(f"Model:            {app.llm.get_model_name()}")
            print(f"Max output token: {app.llm.get_max_output_token()}")
            print(f"Temperature:      {app.llm.get_temperature()}")

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


def main() -> None:
    if "OPENAI_API_KEY" not in os.environ:
        print("Please set OPENAI API key in your environment first")
        return

    configure_logging()
    parser = create_parser()
    args = parse_args(parser)
    config = load_config(args)
    app = create_app(config)

    if args.interactive:
        run_interactive(app, parser)
    else:
        run_one_shot(app, args.question)


if __name__ == "__main__":
    main()
