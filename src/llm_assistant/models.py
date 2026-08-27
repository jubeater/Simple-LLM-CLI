from dataclasses import dataclass


@dataclass
class Usage:
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None


@dataclass
class StreamResult:
    usage: Usage
    total_duration: float | None


@dataclass
class SessionMetrics:
    request_count: int
    error_count: int
    total_usage: Usage
    total_latency: float | None
    last_latency: float | None
