import argparse

from .app import ask


def CLI() -> None:
    # 1. Create parser
    parser = argparse.ArgumentParser(description="provide a CLI for the LLM Assistant")

    # 2. Add arguments
    parser.add_argument("operation", help="the operation user wants to perform, only one operation for now: 'ask'")
    parser.add_argument("question", help="the question for the operation")

    # 3. Parse arguments
    args = parser.parse_args()

    # 4. call llm assistant function
    answer = ask(args.question)
    # 5. print llm response
    print(
        f"{answer}"
    )
