import os
from pathlib import Path
import tempfile
import unittest

from trojan_refinement.rag.graph_similarity import circuit_graph
from trojan_refinement.rag.neo4j_store import Neo4jConfig, Neo4jGraphStore


class Neo4jQueryTests(unittest.TestCase):
    def test_remote_cleartext_credentials_and_unsafe_database_names_are_rejected(self):
        for kwargs in (
            {"http_uri": "http://example.com"},
            {"database": "neo4j/other"},
            {"http_uri": "https://user:secret@example.com"},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Neo4jConfig(**kwargs)

    def test_query_anchors_are_parameters_and_limits_are_validated(self):
        captured = []
        store = Neo4jGraphStore.__new__(Neo4jGraphStore)
        store.query = lambda query, parameters: captured.append((query, parameters)) or []
        name = "a') DETACH DELETE n //"
        store.bounded_paths("key", sources=[name], max_hops=2)
        query, parameters = captured[0]
        self.assertNotIn(name, query)
        self.assertEqual(parameters["sources"], [name])
        self.assertIn("*1..2", query)
        for depth in (0, 33, True):
            with self.assertRaises(ValueError):
                store.bounded_paths("key", sources=["a"], max_hops=depth)
        with self.assertRaises(ValueError):
            store.bounded_paths("key")

    def test_result_limit_is_reported_as_truncation(self):
        store = Neo4jGraphStore.__new__(Neo4jGraphStore)
        store.query = lambda *_: [
            {"nodes": ["a", "b"], "edges": []},
            {"nodes": ["a", "c"], "edges": []},
        ]
        result = store.bounded_paths("key", sources=["a"], limit=1)
        self.assertTrue(result["truncated"])
        self.assertEqual(len(result["paths"]), 1)


@unittest.skipUnless(
    os.getenv("NEO4J_TEST_URI"), "Set NEO4J_TEST_URI for live Neo4j integration tests"
)
class LiveNeo4jTests(unittest.TestCase):
    def setUp(self):
        self.store = Neo4jGraphStore(
            Neo4jConfig(http_uri=os.environ["NEO4J_TEST_URI"], password_env="NEO4J_TEST_PASSWORD")
        )
        self.keys = []

    def tearDown(self):
        for key in self.keys:
            self.store.delete_graph(key)

    def test_live_import_direction_port_filters_and_graph_isolation(self):
        graph = circuit_graph(
            [
                {
                    "instance_name": "a",
                    "cell_type": "and2s1",
                    "port_connections": {"DIN1": "i", "DIN2": "j", "Q": "x"},
                },
                {
                    "instance_name": "b",
                    "cell_type": "xor2s1",
                    "port_connections": {"DIN1": "x", "DIN2": "host", "Q": "out"},
                },
            ]
        )
        first = self.store.import_graph(graph, "test_graph_a")
        self.keys.append(first)
        second = self.store.import_graph(graph, "test_graph_b")
        self.keys.append(second)
        result = self.store.bounded_paths(
            first, sources=["a"], targets=["b"], target_port="DIN1", max_hops=1
        )
        self.assertEqual([p["nodes"] for p in result["paths"]], [["a", "b"]])
        self.assertEqual(
            self.store.bounded_paths(first, sources=["b"], targets=["a"], max_hops=1)["paths"], []
        )
        self.assertEqual(
            self.store.bounded_paths(first, targets=["b"], target_port="DIN2", max_hops=1)["paths"],
            [],
        )
        self.store.delete_graph(first)
        self.keys.remove(first)
        self.assertTrue(self.store.bounded_paths(second, sources=["a"], max_hops=1)["paths"])

    def test_live_module_recovery_uses_retrieved_missing_interface_paths(self):
        from test_pipeline import demo
        from trojan_refinement.pipeline import StructureConfig, prepare_candidates

        with tempfile.TemporaryDirectory() as tmp:
            report = demo.create_demo(tmp)
            settings = StructureConfig(upstream_depth=0, downstream_depth=0)
            result = prepare_candidates(
                report, "synthetic_demo", 4, "combinational", settings, graph_store=self.store
            )
            self.keys.append(result["metadata"]["graph_key"])
            audits = result["payload_result"]["module_recovery"]
            self.assertTrue(any(a["graph_queries"] for a in audits))
            self.assertTrue(any("collect_b" in a["accepted_new_instances"] for a in audits))
