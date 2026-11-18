"""Messages API transport with signed thinking-block preservation."""

import json
from types import SimpleNamespace
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener
from .chat_completions import MAX_BYTES as MAX_RESPONSE_BYTES, NoRedirects, redact

API_VERSION = "2023-06-01"


def read_stream(response):
    """Assemble native SSE without exposing thinking text in console output."""
    message, blocks, stopped, event_lines, total = (None, {}, False, [], 0)

    def consume(lines):
        nonlocal message, stopped
        payload = "\n".join((line[5:].lstrip() for line in lines if line.startswith("data:")))
        if not payload:
            return
        data = json.loads(payload)
        typ = data.get("type")
        if typ == "error":
            raise ValueError(
                "Native streaming error: " + json.dumps(data.get("error"), ensure_ascii=False)
            )
        if typ == "message_start":
            if message is not None:
                raise ValueError("Duplicate native message_start")
            message = data["message"]
        elif typ == "content_block_start":
            i = data["index"]
            if i in blocks:
                raise ValueError("Duplicate native block index")
            blocks[i] = {"block": data["content_block"], "json_parts": [], "stopped": False}
        elif typ == "content_block_delta":
            item, delta = (blocks[data["index"]], data["delta"])
            block = item["block"]
            kind = delta.get("type")
            if kind in ("text_delta", "thinking_delta", "signature_delta"):
                field = {
                    "text_delta": "text",
                    "thinking_delta": "thinking",
                    "signature_delta": "signature",
                }[kind]
                block[field] = block.get(field, "") + delta[field]
            elif kind == "input_json_delta":
                item["json_parts"].append(delta["partial_json"])
            else:
                raise ValueError("Unsupported native content delta: " + str(kind))
        elif typ == "content_block_stop":
            item = blocks[data["index"]]
            if item["json_parts"]:
                item["block"]["input"] = json.loads("".join(item["json_parts"]))
            item["stopped"] = True
        elif typ == "message_delta":
            if message is None:
                raise ValueError("message_delta before message_start")
            message.update(data.get("delta", {}))
            message.setdefault("usage", {}).update(data.get("usage", {}))
        elif typ == "message_stop":
            stopped = True
        elif typ not in ("ping", None):
            raise ValueError("Unsupported native SSE event: " + str(typ))

    while True:
        raw = response.readline(MAX_RESPONSE_BYTES + 1)
        if not raw:
            if event_lines:
                consume(event_lines)
            break
        total += len(raw)
        if total > MAX_RESPONSE_BYTES:
            raise ValueError("Messages API stream exceeds the response byte limit")
        line = raw.decode("utf-8").rstrip("\r\n")
        if line:
            event_lines.append(line)
        else:
            consume(event_lines)
            event_lines = []
    if not stopped or message is None or any((not b["stopped"] for b in blocks.values())):
        raise ValueError("Native SSE ended before message_stop; incomplete output is not scored")
    if sorted(blocks) != list(range(len(blocks))):
        raise ValueError("Noncontiguous native content block indices")
    message["content"] = [blocks[i]["block"] for i in sorted(blocks)]
    return message


class GatewayError(RuntimeError):

    def __init__(self, status, body, headers, key=""):
        self.status_code = status
        self.body = body
        self.response = SimpleNamespace(headers=headers)
        super().__init__(
            redact(f"Messages API HTTP {status}: {json.dumps(body, ensure_ascii=False)}", key)
        )


class MessagesClient:

    def __init__(self, key, base_url, timeout=1200, opener=None):
        self.endpoint = base_url.rstrip("/") + "/messages"
        self.key, self.timeout = (key, timeout)
        self.opener = opener or build_opener(NoRedirects())
        self.responses = SimpleNamespace(create=self.create)

    def create(self, **body):
        request = Request(
            self.endpoint,
            data=json.dumps(body, ensure_ascii=False).encode(),
            headers={
                "Authorization": "Bearer " + self.key,
                "x-api-key": self.key,
                "anthropic-version": API_VERSION,
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Accept-Encoding": "identity",
            },
            method="POST",
        )
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                headers = getattr(response, "headers", {})
                if "text/event-stream" in headers.get("Content-Type", "").lower():
                    return read_stream(response)
                raw = response.read(MAX_RESPONSE_BYTES + 1)
        except HTTPError as exc:
            raw = redact(exc.read(8192).decode(errors="replace"), self.key)
            try:
                data = json.loads(raw)
            except ValueError:
                data = {"error": {"message": raw}}
            raise GatewayError(exc.code, data, dict(exc.headers), self.key) from None
        except URLError as exc:
            raise RuntimeError(
                redact(f"Messages API connection failed：{exc.reason}", self.key)
            ) from None
        if len(raw) > MAX_RESPONSE_BYTES:
            raise ValueError("Messages API response exceeds the response byte limit")
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("Expected a native Messages response object")
        if data.get("error"):
            raise GatewayError(200, data, {}, self.key)
        return data


def check_response(data, model):
    if data.get("model") != model:
        raise ValueError("Returned model does not match the frozen requested model")
    if data.get("type") != "message" or data.get("role") != "assistant":
        raise ValueError("Expected a native assistant Message; compatibility output is not scored")
    if not isinstance(data.get("content"), list):
        raise ValueError("Missing native content blocks")
    if data.get("stop_reason") not in ("tool_use", "end_turn"):
        raise ValueError("Incomplete/refused native response: " + str(data.get("stop_reason")))
    calls = [b for b in data["content"] if b.get("type") == "tool_use"]
    if (data["stop_reason"] == "tool_use") != bool(calls):
        raise ValueError("Tool blocks do not match the native stop reason")
    return calls


def aggregate_usage(responses):
    """Count native billable input, including cache reads/writes, on every turn.

    Messages output_tokens includes thinking; no separate reasoning count is
    invented when the gateway does not return that field.
    """
    usages = [r.get("usage") or {} for r in responses]

    def total(field, missing=None):
        values = [u.get(field, missing) for u in usages]
        return (
            sum(values) if values and all((isinstance(v, int) and v >= 0 for v in values)) else None
        )

    raw_input = total("input_tokens")
    cache_read, cache_write = (
        total("cache_read_input_tokens", 0),
        total("cache_creation_input_tokens", 0),
    )
    inputs = raw_input + cache_read + cache_write if raw_input is not None else None
    outputs = total("output_tokens")
    reasoning = total("reasoning_tokens")
    return {
        "input_tokens": inputs,
        "output_tokens": outputs,
        "total_tokens": inputs + outputs if inputs is not None and outputs is not None else None,
        "uncached_input_tokens": raw_input,
        "cache_creation_input_tokens": cache_write,
        "input_tokens_details": {"cached_tokens": cache_read},
        "output_tokens_details": {"reasoning_tokens": reasoning},
        "usage_source": "native_messages; output_tokens includes thinking",
    }
