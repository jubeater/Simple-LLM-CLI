# Simple LLM CLI

A simple command-line LLM assistant built with Python and the OpenAI Responses API.

The project is designed as a learning project for understanding how to build an LLM-powered application from scratch, with an emphasis on clean separation of responsibilities, configuration management, streaming responses, error handling, testing, and basic observability.

## Features

* One-shot question answering
* Interactive conversation mode
* Conversation history
* Streaming LLM responses
* Configurable model
* Configurable temperature
* Configurable maximum output tokens
* Runtime model switching
* Session statistics
* Token usage tracking
* Request latency tracking
* Retry handling for streaming/network failures
* Error handling
* Structured application logging
* Unit tests

## Architecture

```text
CLI
 ↓
App
 ├── LLM
 └── Conversation
```

### CLI

Responsible for:

* Parsing command-line arguments
* Interactive user input
* Displaying streamed responses
* Handling CLI commands
* Displaying errors and session statistics

### App

Responsible for application-level orchestration:

* Managing conversation history
* Calling the LLM
* Handling streaming responses
* Retrying retryable streaming failures
* Updating session metrics
* Committing successful assistant responses

### LLM

Responsible for communicating with the OpenAI API:

* Building API requests
* Streaming response events
* Converting provider events into application-level results
* Collecting token usage
* Measuring request latency
* Translating provider exceptions into application exceptions

### Conversation

Responsible for maintaining the conversation history between the user and the assistant.

## Configuration

Configuration follows this flow:

```text
CLI arguments
      ↓
 UserConfig
      ↓
 load_config()
      ↓
Environment variables + Defaults
      ↓
  LLMConfig
      ↓
 create_app()
```

The following LLM settings are supported:

* Model
* Temperature
* Maximum output tokens

## Usage

### One-shot mode

```bash
uv run ai -q "What is TCP?"
```

A single question is sent to the LLM without maintaining a conversation after the request.

### Interactive mode

```bash
uv run ai -i
```

Interactive mode maintains conversation history and supports the following commands:

```text
/model              Show the current model
/model <model_id>   Switch the current model
/config             Show the current configuration
/stats              Show session statistics
/clear              Clear conversation history
/help               Show help
/quit               Exit interactive mode
```

## Streaming

LLM responses are streamed as they are generated rather than waiting for the complete response.

The LLM layer converts provider events into a small application-level interface:

```text
OpenAI response events
        ↓
      LLM
        ↓
 ┌───────────────┐
 │ text chunks   │
 │ StreamResult  │
 │ LLM errors    │
 └───────────────┘
        ↓
       App
        ↓
       CLI
```

The `App` layer consumes the final `StreamResult` to update session metrics and conversation state, while the CLI receives and displays the generated text chunks.

If a streaming/network failure occurs before any text has been received, the application can retry the request up to the configured retry threshold. Once text has already been streamed to the user, the request is not retried to avoid duplicated output.

## Session Statistics

Interactive mode provides `/stats` to display session-level information, including:

```text
Session
---------------------------------------
Model
Requests
Messages

Tokens
---------------------------------------
Input
Output
Total

Performance
---------------------------------------
Last latency
Average latency

Errors
---------------------------------------
Failed requests
```

Successful requests contribute to the request count and token usage statistics. Failed requests are tracked separately.

## Error Handling

The application distinguishes between different types of failures.

```text
OpenAI API exception
        ↓
      LLMError

Streaming error
        ↓
  LLMStreamError
        ↓
      App
        ↓
      Retry
```

Retryable streaming failures are retried by the application layer. If the retry limit is reached, the request is treated as failed.

A partially streamed response is not committed to the conversation if the request ultimately fails.

## Logging

The application uses Python's standard `logging` module.

Logging is used to provide basic operational information such as:

* LLM request start
* LLM request failures
* Retry attempts
* Interactive session termination
* API exceptions

User prompts and sensitive credentials are not intentionally included in normal logging.

## Testing

The project uses `pytest` and `unittest.mock`.

Tests are separated according to application layers:

```text
test_llm.py
    ↓
OpenAI API → LLM

test_app.py
    ↓
LLM → App

test_cli.py
    ↓
App → CLI
```

The tests cover:

* Streaming responses
* Stream completion and usage information
* Missing usage information
* LLM errors
* Streaming errors
* Retry behavior
* Partial-response failure
* Conversation updates
* Session metrics
* CLI commands
* `/stats`
* Interactive session behavior
* CLI error handling

Run the test suite with:

```bash
uv run pytest
```

Run linting with:

```bash
uv run ruff check .
```

## Project Status

### Milestone 1 — Basic CLI

* [x] Basic CLI
* [x] One-shot mode
* [x] Interactive mode
* [x] OpenAI integration

### Milestone 2 — Configuration & Conversation

* [x] Configuration management
* [x] Environment variables
* [x] Conversation history
* [x] Runtime model switching

### Milestone 3 — Refactoring & Testing

* [x] Separate CLI, application, LLM, and conversation responsibilities
* [x] Configuration separation
* [x] Unit testing
* [x] README documentation

### Milestone 4 — Streaming & Observability

* [x] Streaming responses
* [x] Token usage tracking
* [x] Request latency tracking
* [x] Session statistics
* [x] Retry handling
* [x] Exception handling
* [x] Logging
* [x] Unit tests


### Next

No plan yet

## Requirements

* Python
* `uv`
* OpenAI API key

Set the OpenAI API key in your environment before running the application.

## Running the Project

Clone the repository and install the dependencies:

```bash
git clone https://github.com/jubeater/Simple-LLM-CLI.git
cd Simple-LLM-CLI
uv sync
```

Then run:

```bash
uv run ai -i
```

or:

```bash
uv run ai -q "Hello!"
```

## Learning Goals

This project is intentionally kept small. The goal is not to build a production-ready LLM framework, but to learn the fundamental pieces involved in building an LLM application:

* API integration
* Streaming
* Conversation state
* Configuration management
* Error handling
* Retry policies
* Observability
* Unit testing
* Layered application design
