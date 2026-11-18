"""OpenAI-compatible Chat Completions transport with complete SSE validation and reasoning-content replay."""

import json
import re
from types import SimpleNamespace
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

MAX_BYTES = 16 * 1024 * 1024


def redact(text, key=""):
    if key:
        text = text.replace(key, "[REDACTED]")
    return re.sub("sk-[A-Za-z0-9_-]{10,}", "[REDACTED]", text)


class NoRedirects(HTTPRedirectHandler):

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class APIError(RuntimeError):

    def __init__(self, status, body, headers, key=""):
        self.status_code, self.body = (status, body)
        self.response = SimpleNamespace(headers=headers)
        super().__init__(
            redact(f"Chat API HTTP {status}: {json.dumps(body, ensure_ascii=False)}", key)
        )


def read_stream(response):
    """Assemble one complete streamed answer, including tool-call arguments.

    Reasoning is preserved for the model's tool continuation, never printed.
    Truncated streams are errors rather than partial experiment results.
    """
    result = {"choices": [], "usage": None}
    message = {"role": "assistant", "content": ""}
    calls, finish, done, total, event = ({}, None, False, 0, [])

    def consume(lines):
        nonlocal finish, done
        payload = "\n".join((line[5:].lstrip() for line in lines if line.startswith("data:")))
        if not payload:
            return
        if payload == "[DONE]":
            done = True
            return
        if done:
            raise ValueError("Chat stream contains data after [DONE]")
        data = json.loads(payload)
        if data.get("error"):
            raise ValueError(
                "Chat API streaming error: " + json.dumps(data["error"], ensure_ascii=False)
            )
        for key in ("id", "model", "created", "system_fingerprint"):
            if key in data:
                if key in ("id", "model") and key in result and (result[key] != data[key]):
                    raise ValueError("Chat response identity changed within stream")
                result[key] = data[key]
        if data.get("usage") is not None:
            result["usage"] = data["usage"]
        for choice in data.get("choices", []):
            if choice.get("index", 0) != 0:
                raise ValueError("Expected one Chat choice")
            delta = choice.get("delta") or {}
            if delta.get("role") not in (None, "assistant"):
                raise ValueError("Expected an assistant Chat delta")
            for key in ("content", "reasoning_content", "refusal"):
                if delta.get(key) is not None:
                    if not isinstance(delta[key], str):
                        raise ValueError("Non-text Chat delta")
                    message[key] = message.get(key, "") + delta[key]
            for call in delta.get("tool_calls") or []:
                index = call.get("index")
                if type(index) is not int or index < 0:
                    raise ValueError("Invalid streamed tool index")
                current = calls.setdefault(
                    index, {"type": "function", "function": {"name": "", "arguments": ""}}
                )
                if call.get("id"):
                    if current.get("id") not in (None, call["id"]):
                        raise ValueError("Streamed tool ID changed")
                    current["id"] = call["id"]
                if call.get("type") not in (None, "function"):
                    raise ValueError("Unsupported Chat tool type")
                for key in ("name", "arguments"):
                    value = (call.get("function") or {}).get(key)
                    if value is not None:
                        current["function"][key] += value
            if choice.get("finish_reason") is not None:
                if finish is not None:
                    raise ValueError("Duplicate Chat finish reason")
                finish = choice["finish_reason"]

    while True:
        raw = response.readline(MAX_BYTES + 1)
        if not raw:
            if event:
                consume(event)
            break
        total += len(raw)
        if total > MAX_BYTES:
            raise ValueError("Chat API stream exceeds 16 MiB")
        line = raw.decode("utf-8").rstrip("\r\n")
        if line:
            event.append(line)
        else:
            consume(event)
            event = []
    if not done or finish is None:
        raise ValueError("Incomplete Chat stream; missing finish_reason or [DONE]")
    if sorted(calls) != list(range(len(calls))):
        raise ValueError("Noncontiguous streamed tool indices")
    if calls:
        message["tool_calls"] = [calls[i] for i in sorted(calls)]
    result["choices"] = [{"index": 0, "message": message, "finish_reason": finish}]
    return result


class ChatClient:

    def __init__(self, key, base_url, timeout=1200, opener=None):
        self.key, self.endpoint, self.timeout = (
            key,
            base_url.rstrip("/") + "/chat/completions",
            timeout,
        )
        self.opener = opener or build_opener(NoRedirects())
        self.responses = SimpleNamespace(create=self.create)

    def create(self, **body):
        request = Request(
            self.endpoint,
            data=json.dumps(body, ensure_ascii=False).encode(),
            method="POST",
            headers={
                "Authorization": "Bearer " + self.key,
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
                "Accept-Encoding": "identity",
            },
        )
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                if "text/event-stream" in response.headers.get("Content-Type", "").lower():
                    return read_stream(response)
                raw = response.read(MAX_BYTES + 1)
        except HTTPError as exc:
            raw = redact(exc.read(8192).decode(errors="replace"), self.key)
            try:
                data = json.loads(raw)
            except ValueError:
                data = {"error": {"message": raw}}
            raise APIError(exc.code, data, dict(exc.headers), self.key) from None
        except URLError as exc:
            raise RuntimeError(
                redact(f"Chat API connection failed：{exc.reason}", self.key)
            ) from None
        if len(raw) > MAX_BYTES:
            raise ValueError("Chat API response exceeds 16 MiB")
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("Expected an Chat API Chat object")
        if data.get("error"):
            raise APIError(200, data, {}, self.key)
        return data


def check_response(data, model):
    if data.get("model") != model:
        raise ValueError(
            "Returned model differs from the requested frozen model: " + str(data.get("model"))
        )
    choices = data.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("Expected one Chat choice")
    choice = choices[0]
    message = choice.get("message") or {}
    if message.get("role") != "assistant" or message.get("refusal"):
        raise ValueError("Missing assistant answer or model refusal")
    calls = message.get("tool_calls") or []
    finish = choice.get("finish_reason")
    if finish not in ("stop", "tool_calls") or (finish == "tool_calls") != bool(calls):
        raise ValueError("Incomplete/refused Chat response: " + str(finish))
    if calls and (not isinstance(calls, list)):
        raise ValueError("Invalid Chat tool-call list")
    if not calls and (not isinstance(message.get("content"), str)):
        raise ValueError("Missing final Chat text")
    return (message, calls)


def aggregate_usage(responses):
    usages = [response.get("usage") or {} for response in responses]

    def total(key):
        values = [usage.get(key) for usage in usages]
        return (
            sum(values)
            if values and all((type(value) is int and value >= 0 for value in values))
            else None
        )

    details = [
        (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") for usage in usages
    ]
    return {
        "input_tokens": total("prompt_tokens"),
        "output_tokens": total("completion_tokens"),
        "total_tokens": total("total_tokens"),
        "reasoning_tokens": (
            sum(details) if details and all((type(v) is int and v >= 0 for v in details)) else None
        ),
    }
