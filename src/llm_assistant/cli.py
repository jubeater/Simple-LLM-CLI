import argparse
import sys

from .app import App
from .conversation import Conversation
from .llm import LLM


def CLI() -> None:
    # 1. Create parser
    parser = argparse.ArgumentParser(prog="uv run main.py", description="provide a interactive shell for LLM Assistant")
    parser.add_argument("-q", "--question", help="Question for LLM, no conversation stored")
    parser.add_argument("-i", "--interactive", action="store_true", help="Interactive mode with conversation history")


    args = parser.parse_args()

    # 2. init required class, could add different platform, user-specific conversation later.
    llm = LLM()
    conversation = Conversation()
    app = App(llm, conversation)

    # 3. return answer if not interactive
    if not args.interactive:
        answer = app.ask(args.question)
        print(f"Assistant: {answer}")
        return

    # --Interactive mode--

    # First, keep ArgParser from exiting on invalid input
    class InvalidArgs(Exception):
        pass
    def exit(*args, **kwargs):
        raise InvalidArgs
    parser.exit = exit


    print("Enter your questions. \nUse '/help' for info, '/clear' to clear conversation', \n'/quit' to quit, ctrl + c or ctrl + d to leave.")
    while True:
        try:
            question = input("You: ").strip()
        except KeyboardInterrupt:
            # Ctrl-c clears the input
            sys.stdout.write('\n')
            break
        except EOFError:
            # Ctrl-d exits
            sys.stdout.write('\n')
            break

        if question == None or question == "":
            print("Assistant: no input\n")

        elif question == '/help':
            parser.print_help()
            continue

        elif question == '/clear':
            answer = app.clear()
            print("Assistant: clear conversation history\n")

        elif question == '/quit':
            break

        else:
            try:
                answer = app.ask(question)
                print(f"Assistant: {answer}\n")
            except InvalidArgs:
                print("Issue happened when fetching result from the LLM.")
                break

