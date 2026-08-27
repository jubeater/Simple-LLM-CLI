# Simple LLM CLI

A simple command-line LLM application built with Python.

This project is part of my AI engineering learning roadmap. The goal is to build a small but properly structured LLM application while learning practical software engineering concepts such as configuration management, application layering, testing, and dependency isolation.

## Features

- One-shot question answering
- Interactive conversation mode
- Conversation history
- Runtime model switching
- Configurable temperature(support cli arg only)
- Configurable maximum output tokens(support cli arg only)
- Environment-variable based configuration
- Error handling
- Logging
- Unit testing

## Requirements

- Python 3.XX+
- `uv`
- An OpenAI API key

## Installation

Clone the repository:

```bash
git clone https://github.com/jubeater/Simple-LLM-CLI.git
cd Simple-LLM-CLI
```

Install dependencies:
```bash
uv sync
```

Set your OpenAI API key in your environment:
```bash
export OPENAI_API_KEY="your-api-key"
```
## Usage

### One-shot mode

Ask a single question:
```bash
uv run ai -q "What is an LLM?"
```
The question is sent to the LLM and the answer is printed without storing the conversation.

### Interactive mode

Start an interactive session:
```bash
uv run ai -i
```

Example:
```
You: What is an LLM?
Assistant: ...

You: How does attention work?
Assistant: ...
```

### Command-line configuration

You can specify the model:
```bash
uv run ai -q "Hello" --model <model>
```

Set temperature:
```bash
uv run ai -q "Hello" --temperature 0.7
```

Set the maximum output token limit:
```bash
uv run ai -q "Hello" --max-output-tokens 1000
```

Options can also be combined:
```bash
uv run ai \
    -q "Explain attention mechanisms" \
    --model <model> \
    --temperature 0.7 \
    --max-output-tokens 1000
```

## Interactive Commands

When running in interactive mode:

| Command          | Description                    |
| ---------------- | ------------------------------ |
| `/model`         | Show the current model         |
| `/model <model>` | Switch the current model       |
| `/config`        | Show the current configuration |
| `/clear`         | Clear conversation history     |
| `/help`          | Show available CLI options     |
| `/quit`          | Exit the application           |

## Configuration
Configuration is resolved from multiple sources.


The application accepts user-provided configuration through CLI arguments and uses environment variables and defaults to complete the configuration.


The configuration flow is:
```
CLI arguments
      |
      v
  UserConfig
      |
      v
 load_config()
      |
      +---- Environment variables
      |
      +---- Defaults
      |
      v
  LLMConfig
      |
      v
  create_app()
```

`UserConfig` may contain `None` values because the user does not have to specify every option.


`LLMConfig` represents the resolved configuration used to initialize the LLM and contains the required values.


## Architecture

The application follows a simple layered structure:
```
CLI
 |
 v
App
 |
 +----> LLM
 |
 +----> Conversation
```

The CLI is responsible for:

- Parsing command-line arguments
- Handling user interaction
- Formatting output
- Converting application errors into user-facing messages

The `App` layer is responsible for application-level orchestration.

The `LLM` class handles communication with the LLM provider.

The `Conversation` class manages conversation history.

Configuration resolution is handled separately by `config.py`.


## Testing
Run the test suite with:
```bash
uv run pytest
```


