import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from trojan_refinement.rag.graph_similarity import circuit_graph
from trojan_refinement.rag.retriever import (
    StructuralRetriever as GeneratorRAG,
    connected_payload,
    execute_call,
    freeze_knowledge_base as snapshot,
)


def graph(names=("n1", "n2"), first_type="and2s1"):
    return circuit_graph(
        [
            {
                "instance_name": names[0],
                "cell_type": first_type,
                "port_connections": {"DIN1": "a", "DIN2": "b", "Q": "trigger"},
            },
            {
                "instance_name": names[1],
                "cell_type": "xor2s1",
                "port_connections": {"DIN1": "data", "DIN2": "trigger", "Q": "out"},
            },
        ]
    )


def bank():
    return {
        "schema": "generator_structural_rag_v1",
        "entries": [
            {
                "reference_id": "same_host",
                "host": "c3540",
                "suite": "TRIT-TC",
                "categories": ["combinational"],
                "graph": graph(),
            },
            {
                "reference_id": "wrong_suite",
                "host": "s13207",
                "suite": "TRIT-TS",
                "categories": ["fsm"],
                "graph": graph(),
            },
            {
                "reference_id": "rag_ref0001",
                "host": "c5315",
                "suite": "TRIT-TC",
                "categories": ["combinational"],
                "graph": graph(("kr1n1", "kr1n2")),
            },
        ],
    }


class RetrievalTests(unittest.TestCase):

    def setUp(self):
        self.payload = {
            "candidate_groups": [{"group_id": "g1", "graph": graph()}],
            "references": [],
        }
        self.args = {
            "group_id": "g1",
            "query": "XOR trigger branch",
            "trojan_type": "any",
            "top_k": 3,
        }

    def test_held_out_host_and_suite_filter_and_anonymous_output(self):
        rag = GeneratorRAG(bank(), self.payload, "c3540", "TRIT-TC")
        result = rag.search(self.args)
        self.assertEqual([m["reference_id"] for m in result["matches"]], ["rag_ref0001"])
        self.assertEqual(result["eligible_reference_count"], 1)
        encoded = json.dumps(result)
        for private in ("c3540", "c5315", "same_host", "wrong_suite", '"host"'):
            self.assertNotIn(private, encoded)
        self.assertNotIn("source", result["matches"][0])
        self.assertGreater(result["matches"][0]["graph_similarity"], 0.99)

    def test_return_budget_keeps_whole_graphs_and_reports_omissions(self):
        rag = GeneratorRAG(bank(), self.payload, "c3540", "TRIT-TC", max_result_bytes=1024)
        result = rag.search(self.args)
        self.assertLessEqual(len(json.dumps(result, ensure_ascii=False).encode()), 1024)
        self.assertEqual(result["matches"], [])
        self.assertEqual(result["omitted_due_to_budget"], 1)

    def test_missing_edge_sources_are_recovered_from_anonymous_ports(self):
        original = self.payload["candidate_groups"][0]["graph"]
        del original["edges"][0]["source"]
        normalized = connected_payload(self.payload)
        rebuilt = normalized["candidate_groups"][0]["graph"]
        self.assertEqual(rebuilt["edges"][0]["source"], "n1")
        self.assertNotIn("source", original["edges"][0])
        self.assertEqual([n["id"] for n in rebuilt["nodes"]], [n["id"] for n in original["nodes"]])
        rag = GeneratorRAG(bank(), self.payload, "c3540", "TRIT-TC")
        self.assertEqual(rag.search(self.args)["matches"][0]["graph_similarity"], 1.0)

    def test_bad_queries_are_tool_errors_and_cannot_choose_files_or_hosts(self):
        rag = GeneratorRAG(bank(), self.payload, "c3540", "TRIT-TC")
        for change in (
            {"group_id": "unknown"},
            {"host": "c3540"},
            {"top_k": 4},
            {"top_k": True},
            {"query": ""},
            {"query": "x" * 1001},
        ):
            with self.subTest(change=change):
                args = {**self.args, **change}
                call = execute_call(rag, "search_trojan_rag", json.dumps(args))
                self.assertEqual(call["result"]["status"], "error")
                self.assertEqual(call["result"]["matches"], [])

    def test_snapshot_verifies_sources_anonymizes_identifiers_and_preserves_constants(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = Path(tmp)
            source = kb / "TRIT-TC/c5315_T002/log.txt"
            source.parent.mkdir(parents=True)
            source.write_text("reference source")
            original = graph(("troj0_gate", "trojan0_payload"))
            original["nodes"][0]["ports"]["DIN1"] = "1'b0"
            entry = {
                "id": "TRIT-TC/c5315_T002",
                "variant": "c5315_T002",
                "suite": "TRIT-TC",
                "source": "TRIT-TC/c5315_T002/log.txt",
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "type": "combinational",
                "graph": original,
            }
            index = kb / "trojan_graphs/index.json"
            index.parent.mkdir()
            index.write_text(json.dumps({"entries": [entry]}))
            frozen = snapshot(kb)
            encoded = json.dumps(frozen["entries"][0]["graph"])
            self.assertNotIn("troj0_gate", encoded)
            self.assertNotIn("trojan0_payload", encoded)
            self.assertIn("1'b0", encoded)
            source.write_text("modified")
            with self.assertRaisesRegex(ValueError, "stale"):
                snapshot(kb)
