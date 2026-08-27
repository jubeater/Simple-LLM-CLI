import argparse
import logging
import sys

from llm_assistant.app import App, create_app
from llm_assistant.config import UserConfig, load_config
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
        "--max_output_tokens", type=int, help="Set max output token limit."
    )
    parser.add_argument(
        "-t", "--temperature", type=float, help="Set temperature of the answer."
    )

    return parser


def print_llm_output(stream_output) -> int:
    try:
        print("Assistant: ", end="")
        for result in stream_output:
            print(result, end="")
        print()
        return 0
    except LLMError as error:
        print(f"Assistant error: {error}")
        return 1


def run_one_shot(app: App, question: str | None) -> int:
    if not question:
        print("You need to give an input question.")
        return 1
    return print_llm_output(app.ask(question))


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

        elif user_input == "/stats":
            stats = app.get_stats()
            cur_config = app.get_config()
            messages = app.get_num_messages()
            avg_latency = (
                None
                if stats.total_latency == None
                else (stats.total_latency / stats.request_count)
            )
            print("Session")
            print("---------------------------------------")
            print(f"Model:            {cur_config.model}")
            print(f"Requests:         {stats.request_count}")
            print(f"Messages:         {messages}")
            print()
            print("Tokens")
            print("---------------------------------------")
            print(f"Input:            {stats.total_usage.input_tokens}")
            print(f"Output:           {stats.total_usage.output_tokens}")
            print(f"Total:            {stats.total_usage.total_tokens}")
            print()
            print("Performance")
            print("---------------------------------------")
            print(f"Last latency:     {stats.last_latency}")
            print(f"Average latency:  {avg_latency}")
            print()
            print("Errors")
            print("---------------------------------------")
            print(f"Failed requests:     {stats.error_count}")

        elif user_input == "/help":
            parser.print_help()

        elif user_input == "/clear":
            app.clear()
            print("Clear conversation history\n")

        elif user_input == "/quit":
            break

        else:
            print_llm_output(app.ask(user_input))


def main() -> int:
    configure_logging()
    parser = create_parser()
    args = parser.parse_args()
    if args.model is not None:
        print(f"Using the model {args.model} to answer now...")

    user_config = UserConfig(
        model=args.model,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
    )
    try:
        app_config = load_config(user_config)
    except ConfigError as error:
        print(f"Configuration error: {error}")
        return 1
    app = create_app(app_config)

    if args.interactive:
        run_interactive(app, parser)
        return 1
    return run_one_shot(app, args.question)


if __name__ == "__main__":
    raise SystemExit(main())
