"""Normalize three API formats for one provider-independent Generator loop."""

from copy import deepcopy

from .chat_completions import ChatClient, check_response as check_chat
from .config import ModelConfig
from .messages import MessagesClient, check_response as check_messages
from ..generator.contracts import SCHEMA


class ModelClient:
    def __init__(self, config: ModelConfig):
        self.config = config
        key = config.read_key()
        if config.api == "responses":
            try:
                from openai import OpenAI
            except ImportError:
                raise RuntimeError("Responses API requires: pip install '.[api]'") from None
            self.client = OpenAI(
                api_key=key,
                base_url=config.base_url,
                timeout=config.timeout_seconds,
                max_retries=0,
                default_headers={"Accept-Encoding": "identity"},
            )
        elif config.api == "chat_completions":
            self.client = ChatClient(key, config.base_url, config.timeout_seconds)
        else:
            self.client = MessagesClient(key, config.base_url, config.timeout_seconds)

    def close(self):
        if self.config.api == "responses":
            self.client.close()

    def create(self, history, tools):
        """Return append-only continuation items, function calls, and final text."""
        cfg = self.config
        if cfg.api == "responses":
            body = {
                "model": cfg.model,
                "input": deepcopy(history),
                "store": False,
                "max_output_tokens": cfg.max_output_tokens,
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "trojan_review",
                        "strict": True,
                        "schema": SCHEMA,
                    }
                },
            }
            if cfg.reasoning_effort is not None:
                body["reasoning"] = {"effort": cfg.reasoning_effort}
            body.update(
                tools=tools,
                tool_choice="auto" if tools else "none",
                parallel_tool_calls=False,
                include=["reasoning.encrypted_content"],
            )
            response = self.client.responses.create(**body)
            raw = response.model_dump(mode="json", by_alias=True, exclude_none=True)
            if raw.get("status") != "completed" or raw.get("model") != cfg.model:
                raise ValueError("Incomplete response or unexpected model")
            output = raw.get("output", [])
            if any(
                part.get("type") == "refusal" for item in output for part in item.get("content", [])
            ):
                raise ValueError("Model refused the request")
            calls = [
                {"id": item["call_id"], "name": item["name"], "arguments": item["arguments"]}
                for item in output
                if item.get("type") == "function_call"
            ]
            text = "".join(
                part["text"]
                for item in output
                if item.get("type") == "message"
                for part in item.get("content", [])
                if part.get("type") == "output_text"
            )
            return {"continuation": output, "calls": calls, "text": text, "raw": raw}
        if cfg.api == "chat_completions":
            body = {
                "model": cfg.model,
                "messages": deepcopy(history),
                "stream": True,
                "stream_options": {"include_usage": True},
                "max_completion_tokens": cfg.max_output_tokens,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "trojan_review", "strict": True, "schema": SCHEMA},
                },
            }
            if cfg.reasoning_effort is not None:
                body["reasoning_effort"] = cfg.reasoning_effort
            if cfg.enable_thinking is not None:
                body["enable_thinking"] = cfg.enable_thinking
            if tools:
                body.update(
                    tools=[
                        {
                            "type": "function",
                            "function": {
                                k: t[k] for k in ("name", "description", "parameters", "strict")
                            },
                        }
                        for t in tools
                    ],
                    tool_choice="auto",
                    parallel_tool_calls=False,
                )
            raw = self.client.create(**body)
            message, pending = check_chat(raw, cfg.model)
            calls = [
                {
                    "id": call["id"],
                    "name": call["function"]["name"],
                    "arguments": call["function"]["arguments"],
                }
                for call in pending
            ]
            return {
                "continuation": [message],
                "calls": calls,
                "text": message.get("content") or "",
                "raw": raw,
            }
        body = {
            "model": cfg.model,
            "system": history[0]["content"],
            "messages": deepcopy(history[1:]),
            "max_tokens": cfg.max_output_tokens,
            "stream": True,
            **deepcopy(cfg.request_options),
        }
        if tools:
            # Enums/local validators enforce numeric bounds for gateways that
            # do not support minimum/maximum in their native tool grammar.
            from copy import deepcopy as clone

            native = []
            for tool in tools:
                schema = clone(tool["parameters"])
                for field in schema.get("properties", {}).values():
                    if field.get("type") == "integer" and "minimum" in field and "maximum" in field:
                        field["enum"] = list(range(field.pop("minimum"), field.pop("maximum") + 1))
                    if field.get("type") == "array":
                        field.pop("maxItems", None)
                native.append(
                    {
                        "name": tool["name"],
                        "description": tool["description"],
                        "input_schema": schema,
                    }
                )
            body.update(
                tools=native, tool_choice={"type": "auto", "disable_parallel_tool_use": True}
            )
        raw = self.client.create(**body)
        pending = check_messages(raw, cfg.model)
        import json

        calls = [
            {"id": call["id"], "name": call["name"], "arguments": json.dumps(call["input"])}
            for call in pending
        ]
        return {
            "continuation": [{"role": "assistant", "content": raw["content"]}],
            "calls": calls,
            "text": "".join(b["text"] for b in raw["content"] if b["type"] == "text"),
            "raw": raw,
        }

    def tool_messages(self, results):
        if self.config.api == "responses":
            return [
                {"type": "function_call_output", "call_id": r["id"], "output": r["content"]}
                for r in results
            ]
        if self.config.api == "chat_completions":
            return [
                {"role": "tool", "tool_call_id": r["id"], "content": r["content"]} for r in results
            ]
        return [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": r["id"], "content": r["content"]}
                    for r in results
                ],
            }
        ]
