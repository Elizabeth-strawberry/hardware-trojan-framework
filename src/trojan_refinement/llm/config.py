"""Explicit provider settings; credentials are read from the named environment variable."""

from dataclasses import asdict, dataclass, field
import json
import math
import os
from pathlib import Path
from urllib.parse import urlsplit


@dataclass(frozen=True)
class ModelConfig:
    api: str
    model: str
    base_url: str
    api_key_env: str
    reasoning_effort: str | None = None
    enable_thinking: bool | None = None
    timeout_seconds: float = 1200
    max_output_tokens: int = 32768
    max_rag_calls: int = 3
    max_result_calls: int = 6
    max_result_bytes: int = 131072
    request_options: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.api not in ("responses", "chat_completions", "messages"):
            raise ValueError("Unknown API format")
        if not self.model.strip() or not self.api_key_env.isidentifier():
            raise ValueError("Specify a model and credential environment variable")
        parsed = urlsplit(self.base_url)
        if (
            parsed.scheme != "https"
            or not parsed.netloc
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "Use an HTTPS API base URL without embedded credentials or query strings"
            )
        if (
            type(self.timeout_seconds) not in (int, float)
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("Timeout must be positive and finite")
        for name in ("max_output_tokens", "max_rag_calls", "max_result_calls", "max_result_bytes"):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name.endswith("calls") else 1):
                raise ValueError("Invalid model budget: " + name)
        if self.reasoning_effort not in (None, "none", "low", "medium", "high", "max"):
            raise ValueError(
                "Unknown reasoning effort; check the chosen model's supported settings"
            )
        if self.enable_thinking is not None and type(self.enable_thinking) is not bool:
            raise ValueError("enable_thinking must be boolean or omitted")
        if not isinstance(self.request_options, dict) or not set(self.request_options) <= {
            "thinking",
            "output_config",
        }:
            raise ValueError("Only native Messages thinking/output_config options are supported")
        if self.request_options and self.api != "messages":
            raise ValueError("Native request_options require the Messages API")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))

    def public_dict(self):
        return asdict(self)

    def read_key(self):
        key = os.environ.get(self.api_key_env, "").strip()
        if not key or any(c.isspace() for c in key):
            raise ValueError("Set the credential environment variable " + self.api_key_env)
        return key
