from copy import deepcopy
import io
import json
from types import SimpleNamespace
import unittest

from trojan_refinement.llm.chat_completions import read_stream as read_chat_stream
from trojan_refinement.llm.client import ModelClient
from trojan_refinement.llm.config import ModelConfig
from trojan_refinement.llm.messages import read_stream as read_messages_stream
from trojan_refinement.generator import stage_results


class StubTransport:
    def __init__(self, response):
        self.response, self.requests = response, []
        self.responses = self

    def create(self, **body):
        self.requests.append(body)
        return self.response


def model_client(api, response, **options):
    client = ModelClient.__new__(ModelClient)
    client.config = ModelConfig(
        api, "fixture-model", "https://example.test/v1", "FIXTURE_KEY", **options
    )
    client.client = StubTransport(response)
    return client


class ModelClientTests(unittest.TestCase):
    def setUp(self):
        self.history = [
            {"role": "system", "content": "fixture instructions"},
            {"role": "user", "content": "{}"},
        ]

    def test_chat_preserves_reasoning_and_tool_id_for_continuation(self):
        message = {
            "role": "assistant",
            "content": "",
            "reasoning_content": "fixture internal content",
            "tool_calls": [
                {
                    "id": "call1",
                    "type": "function",
                    "function": {"name": stage_results.TOOL_NAME, "arguments": "{}"},
                }
            ],
        }
        raw = {
            "model": "fixture-model",
            "choices": [{"message": message, "finish_reason": "tool_calls"}],
        }
        client = model_client(
            "chat_completions", raw, enable_thinking=True, reasoning_effort="high"
        )
        reply = client.create(self.history, [stage_results.TOOL])
        self.assertEqual(reply["continuation"][0], message)
        self.assertEqual(reply["calls"][0]["id"], "call1")
        request = client.client.requests[0]
        self.assertTrue(request["enable_thinking"])
        self.assertEqual(request["reasoning_effort"], "high")
        self.assertEqual(request["tools"][0]["function"]["name"], stage_results.TOOL_NAME)
        self.assertEqual(
            client.tool_messages([{"id": "call1", "content": "result"}])[0]["tool_call_id"], "call1"
        )

    def test_kimi_config_does_not_invent_reasoning_effort(self):
        raw = {
            "model": "fixture-model",
            "choices": [
                {"message": {"role": "assistant", "content": "{}"}, "finish_reason": "stop"}
            ],
        }
        client = model_client("chat_completions", raw, enable_thinking=True)
        client.create(self.history, [])
        self.assertNotIn("reasoning_effort", client.client.requests[0])

    def test_native_messages_preserve_signed_thinking_blocks(self):
        content = [
            {"type": "thinking", "thinking": "fixture", "signature": "signed-fixture"},
            {"type": "tool_use", "id": "call1", "name": stage_results.TOOL_NAME, "input": {}},
        ]
        raw = {
            "model": "fixture-model",
            "type": "message",
            "role": "assistant",
            "stop_reason": "tool_use",
            "content": content,
        }
        client = model_client(
            "messages",
            raw,
            request_options={"thinking": {"type": "adaptive"}, "output_config": {"effort": "low"}},
        )
        reply = client.create(self.history, [stage_results.TOOL])
        self.assertEqual(reply["continuation"][0]["content"], content)
        self.assertEqual(client.client.requests[0]["thinking"], {"type": "adaptive"})
        self.assertEqual(client.client.requests[0]["messages"], self.history[1:])
        result = client.tool_messages([{"id": "call1", "content": "result"}])
        self.assertEqual(result[0]["content"][0]["tool_use_id"], "call1")

    def test_responses_preserve_encrypted_reasoning_and_aliases(self):
        raw = {
            "model": "fixture-model",
            "status": "completed",
            "output": [
                {"type": "reasoning", "encrypted_content": "fixture-encrypted"},
                {
                    "type": "function_call",
                    "name": stage_results.TOOL_NAME,
                    "call_id": "call1",
                    "arguments": "{}",
                },
            ],
        }
        dumped = []

        def model_dump(**kwargs):
            dumped.append(kwargs)
            return deepcopy(raw)

        client = model_client(
            "responses", SimpleNamespace(model_dump=model_dump), reasoning_effort="none"
        )
        reply = client.create(self.history, [stage_results.TOOL])
        self.assertEqual(reply["continuation"], raw["output"])
        self.assertEqual(client.client.requests[0]["reasoning"], {"effort": "none"})
        self.assertTrue(dumped[0]["by_alias"])
        self.assertTrue(dumped[0]["exclude_none"])

    def test_config_rejects_credentials_in_urls_or_arbitrary_body_options(self):
        for url in (
            "http://example.test/v1",
            "https://user:password@example.test/v1",
            "https://example.test/v1?key=hidden",
        ):
            with self.assertRaises(ValueError):
                ModelConfig("chat_completions", "fixture", url, "FIXTURE_KEY")
        with self.assertRaises(ValueError):
            ModelConfig(
                "messages",
                "fixture",
                "https://example.test/v1",
                "FIXTURE_KEY",
                request_options={"model": "override"},
            )

    def test_incomplete_chat_stream_is_invalid(self):
        stream = io.BytesIO(
            b'data: {"model":"fixture-model","choices":[{"index":0,"delta":{"role":"assistant","content":"partial"}}]}\n\n'
        )
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            read_chat_stream(stream)

    def test_incomplete_native_stream_is_invalid(self):
        stream = io.BytesIO(
            b'data: {"type":"message_start","message":{"model":"fixture-model","content":[]}}\n\n'
        )
        with self.assertRaisesRegex(ValueError, "incomplete"):
            read_messages_stream(stream)


if __name__ == "__main__":
    unittest.main()
