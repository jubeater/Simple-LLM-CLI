import argparse
import os
import sys

from llm_assistant.app import App
from llm_assistant.conversation import Conversation
from llm_assistant.llm import LLM, LLMError


def main() -> None:
    # 1. Configuration
    # Get model configuration from shell environment or set default
    if "OPENAI_API_KEY" not in os.environ:
        print("Please set OPENAI API key in your environment first")
        return

    model_name = os.getenv("AI_MODEL", "gpt-5.4-mini")
    try:
        max_output_token = os.getenv("AI_MAX_OUTPUT_TOKENS", "1000")
        max_output_token = int(max_output_token)
    except ValueError:
        print("Invalid max_output_token get from environment, will use default")
        max_output_token = 1000
    try:
        temperature = os.getenv("AI_TEMPERATURE", "1.0")
        temperature = float(temperature)
    except ValueError:
        print("Invalid temperature get from environment, will use default")
        temperature = 0.7

    # 2. Create parser for user input
    parser = argparse.ArgumentParser(
        prog="uv run main.py",
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
        "-mot", "--max_output_token", help="Set max output token limit."
    )
    parser.add_argument("-t", "--temperature", help="Set temperature of the answer.")

    args = parser.parse_args()

    # Set user input model configuration
    if args.model is not None:
        print(f"Using the model {args.model} to answer now...")
        model_name = args.model
    if args.max_output_token is not None:
        try:
            int_max_output_token = int(args.max_output_token)
            print(f"Set max output token limit to {int_max_output_token}")
            max_output_token = int_max_output_token
        except ValueError:
            print("Max token input is not int, will keep default.")

    if args.temperature is not None:
        try:
            float_temperature = float(args.temperature)
            print(f"Set temperature to {float_temperature}")
            temperature = float_temperature
        except ValueError:
            print("Temperature input is not float, will keep default.")

    # 3. Initilization
    llm = LLM(model_name, max_output_token, temperature)
    conversation = Conversation()
    app = App(llm, conversation)

    # 4. Return answer for not-interactive mode
    if not args.interactive:
        if args.question is None or args.question == "":
            print("you need to give an input question")
            return
        try:
            answer = app.ask(args.question)
            print(f"Assistant: {answer}")
            return
        except LLMError as error:
            print(f"Assistant error: {error}")

    # --Interactive mode--

    print("Enter your questions. \n'/help' for info, '/clear' to reset conversation")
    print("'/model' show current model id, '/model[space][model_id]' to switch model.")
    print("'/config' show configuration, /quit' to quit")
    while True:
        try:
            user_input = input("You: ").strip()
        except KeyboardInterrupt:
            # Ctrl-c clears the input
            sys.stdout.write("\n")
            break
        except EOFError:
            # Ctrl-d exits
            sys.stdout.write("\n")
            break

        if user_input == None or user_input == "":
            print("no input\n")

        elif user_input == "/model":
            print(f"Current model is: {llm.get_model_name()}")

        elif user_input.startswith("/model "):
            input_model = user_input.rstrip()
            input_model = input_model[len("/model ") :]
            llm.set_model_name(input_model)
            print(f"Model switched to {input_model}")

        elif user_input == "/config":
            print("Configuration")
            print("---------------------------------------")
            print(f"Model:            {model_name}")
            print(f"Max output token: {max_output_token}")
            print(f"Temperature:      {temperature}")

        elif user_input == "/help":
            parser.print_help()

        elif user_input == "/clear":
            answer = app.clear()
            print("Clear conversation history\n")

        elif user_input == "/quit":
            break

        else:
            try:
                answer = app.ask(user_input)
                print(f"Assistant: {answer}\n")
            except LLMError as error:
                print(f"Assistant error: {error}")


if __name__ == "__main__":
    main()
