from copy import deepcopy
import json
from types import SimpleNamespace
import unittest

from trojan_refinement.generator.controller import parse_decision, refine_candidates
from trojan_refinement.generator import contracts, stage_results
from trojan_refinement.llm.config import ModelConfig
from trojan_refinement.rag.retriever import StructuralRetriever
from test_retrieval import bank
from test_stage_results import fixture


def decision_row(group, kept, removed):
    return {
        "group_id": group,
        "decision": "refine" if kept and removed else "keep" if kept else "reject",
        "retained_instances": kept,
        "removed_instances": removed,
        "reason": "Synthetic fixture: retain inspection mechanism, exclude separate host-data implementation.",
        "evidence_instances": kept[:1] or removed[:1],
        "reference_ids": [],
        "removal_evidence": (
            [
                {
                    "instances": removed,
                    "category": "host_data_branch",
                    "reason": "Synthetic host context; payload input connection remains a boundary.",
                    "evidence_instances": kept[:1] or removed[:1],
                }
            ]
            if removed
            else []
        ),
    }


class FakeClient:
    def __init__(self, replies, **config):
        self.config = ModelConfig(
            "chat_completions", "fixture-model", "https://example.test/v1", "FIXTURE_KEY", **config
        )
        self.replies = iter(replies)
        self.requests = []

    def create(self, history, tools):
        self.requests.append(deepcopy((history, tools)))
        return next(self.replies)

    def tool_messages(self, results):
        return [{"role": "tool", "tool_call_id": r["id"], "content": r["content"]} for r in results]


def reply(text="", calls=None):
    return {
        "text": text,
        "calls": calls or [],
        "continuation": [{"role": "assistant", "content": text}],
        "raw": {"model": "fixture-model", "usage": {"total_tokens": 10}},
    }


class GeneratorTests(unittest.TestCase):
    def setUp(self):
        self.payload, self.identities, self.module, self.snapshot = fixture()
        self.decision = {
            "decisions": [decision_row("g1", ["n1", "n2"], ["n3"])],
            "summary": "Synthetic test only",
        }

    def test_read_only_result_lookup_then_refine(self):
        pending = [
            {
                "id": "call1",
                "name": stage_results.TOOL_NAME,
                "arguments": json.dumps({"group_id": "g1", "stage": "all", "instance_ids": []}),
            }
        ]
        client = FakeClient([reply(calls=pending), reply(json.dumps(self.decision))])
        original = deepcopy(self.snapshot)
        result = refine_candidates(self.payload, self.snapshot, None, client)
        self.assertEqual(result["retained_anonymous_instances"], ["n1", "n2"])
        self.assertEqual(len(result["stage_result_calls"]), 1)
        self.assertEqual(self.snapshot, original)
        serialized = json.dumps(client.requests)
        for forbidden in ("private_trigger", '"labels"', '"nt"', '"node_id"'):
            self.assertNotIn(forbidden, serialized)
        self.assertIn("read_pipeline_results", serialized)

    def test_rag_and_result_budgets_are_independent(self):
        rag = StructuralRetriever(bank(), self.payload, "c3540", "TRIT-TC")
        call = {
            "id": "rag1",
            "name": "search_trojan_rag",
            "arguments": json.dumps(
                {"group_id": "g1", "query": "XOR branch", "trojan_type": "any", "top_k": 1}
            ),
        }
        decision = deepcopy(self.decision)
        decision["decisions"][0]["reference_ids"] = ["rag_ref0001"]
        client = FakeClient(
            [reply(calls=[call]), reply(json.dumps(decision))], max_rag_calls=1, max_result_calls=1
        )
        result = refine_candidates(self.payload, self.snapshot, rag, client)
        offered = {tool["name"] for tool in client.requests[1][1]}
        self.assertEqual(offered, {"read_pipeline_results"})
        self.assertEqual(len(result["rag_calls"]), 1)

    def test_exhausted_budget_cannot_be_bypassed(self):
        pending = [
            {
                "id": "call1",
                "name": stage_results.TOOL_NAME,
                "arguments": json.dumps({"group_id": "g1", "stage": "all", "instance_ids": []}),
            }
        ]
        client = FakeClient([reply(calls=pending)], max_result_calls=0)
        with self.assertRaisesRegex(ValueError, "budget"):
            refine_candidates(self.payload, self.snapshot, None, client)

    def test_no_missing_summary_repair_or_unjustified_removal(self):
        malformed = deepcopy(self.decision)
        malformed.pop("summary")
        with self.assertRaises(ValueError):
            parse_decision(json.dumps(malformed))
        malformed = deepcopy(self.decision)
        malformed["decisions"][0]["removal_evidence"] = []
        with self.assertRaises(ValueError):
            refine_candidates(
                self.payload, self.snapshot, None, FakeClient([reply(json.dumps(malformed))])
            )

    def test_overlap_is_retained_if_any_group_keeps_it(self):
        payload = deepcopy(self.payload)
        first = payload["candidate_groups"][0]
        first["graph"]["nodes"] = first["graph"]["nodes"][:2]
        second = deepcopy(first)
        second["group_id"] = "g2"
        payload["candidate_groups"].append(second)
        choice = {
            "summary": "Overlap fixture",
            "decisions": [decision_row("g1", ["n1"], ["n2"]), decision_row("g2", ["n2"], ["n1"])],
        }
        selected = contracts.validate(choice, payload)
        self.assertEqual(set().union(*selected.values()), {"n1", "n2"})

    def test_no_api_needed_when_no_candidates(self):
        client = FakeClient([])
        result = refine_candidates(
            {"candidate_groups": [], "references": []},
            {"schema": "generator_pipeline_results_v1", "groups": {}},
            None,
            client,
        )
        self.assertEqual(result["status"], "valid")
        self.assertFalse(client.requests)

    def test_summary_only_prose_and_extra_fields_are_invalid(self):
        for text in ("I would keep all nodes", json.dumps({**self.decision, "extra": 1})):
            with self.assertRaises(ValueError):
                parse_decision(text)

    def test_decoder_register_interfaces_are_serialized_without_signal_names(self):
        names = list(self.identities.values())
        interface = {
            "source_register": names[0],
            "target_register": names[1],
            "role": "missing_decoder_output",
        }
        payload_result = {
            "groups": [
                {
                    "group_id": "p1",
                    "seed_instances": [names[0]],
                    "expanded_instances": names,
                    "accepted": True,
                    "expansion_steps": [],
                    "budget_truncated": False,
                }
            ],
            "candidate_instances": names,
            "settings": {},
        }
        recovery = [
            {
                "source_payload_group": "p1",
                "scope_instances": names,
                "new_scope_instances": [],
                "accepted_new_instances": [],
                "rejected_new_scope_instances": [],
                "interfaces": [],
                "proposals": [
                    {
                        "instance_name": names[1],
                        "interface": interface,
                        "reason": "bounded_decoder_to_output_register_xor",
                    }
                ],
                "boundaries": [],
                "budget_truncated": False,
                "settings": {},
            }
        ]
        samples = [
            {
                "node_id": i,
                "trojan_score": 0.5,
                "drivers": [{"instance_name": name, "cell_type": "fixture", "output_port": "Q"}],
            }
            for i, name in enumerate(names)
        ]
        snapshot = stage_results.snapshot(
            self.payload,
            self.module,
            self.identities,
            payload_result,
            recovery,
            [0],
            [0, 1, 2],
            samples,
        )
        serialized = snapshot["groups"]["g1"]["modulechecker"]["recovery"][0]["proposals"][0][
            "interface"
        ]
        self.assertEqual(serialized["source_register"], "n1")
        self.assertEqual(serialized["target_register"], "n2")
        self.assertNotIn("private_", json.dumps(snapshot))


if __name__ == "__main__":
    unittest.main()
