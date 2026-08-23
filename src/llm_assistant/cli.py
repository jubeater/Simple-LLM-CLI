import argparse
import sys

from .app import ask


def CLI() -> None:
    # 1. Create parser
    parser = argparse.ArgumentParser(prog="uv run main.py", description="provide a interactive shell for LLM Assistant")
    parser.add_argument("-q", "--question", help="the question for LLM assistant to answer")
    parser.add_argument("-i", "--interactive", action="store_true", help="Interactive mode")


    args = parser.parse_args()
    # 2. return answer if not interactive
    if not args.interactive:
        answer = ask(args.question)
        print(f"Assistant: {answer}")
        return

    # --Interactive mode--

    # First, keep ArgParser from exiting on invalid input
    class InvalidArgs(Exception):
        pass
    def exit(*args, **kwargs):
        raise InvalidArgs
    parser.exit = exit


    print("Enter your questions. Use 'help' for info, 'ctrl + c or ctrl + d' to leave.")
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

        if question == 'help':
            parser.print_help()
            continue

        try:
            answer = ask(question)
            print(f"Assistant: {answer}\n")
        except InvalidArgs:
            print("Issue happened when fetching result from the LLM.")
            break

