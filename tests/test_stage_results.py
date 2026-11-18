import copy
import hashlib
import json
from types import SimpleNamespace
import unittest
from trojan_refinement.rag.graph_similarity import circuit_graph
from trojan_refinement.generator import stage_results as stages

runner = SimpleNamespace(
    digest=lambda value: hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
)


def fixture():
    graph = circuit_graph(
        [
            {
                "instance_name": "n1",
                "cell_type": "and2s1",
                "port_connections": {"DIN1": "a", "DIN2": "b", "Q": "trigger"},
            },
            {
                "instance_name": "n2",
                "cell_type": "xor2s1",
                "port_connections": {"DIN1": "data", "DIN2": "trigger", "Q": "out"},
            },
            {
                "instance_name": "n3",
                "cell_type": "nor2s1",
                "port_connections": {"DIN1": "x", "DIN2": "y", "Q": "data"},
            },
        ]
    )
    payload = {
        "candidate_groups": [
            {
                "group_id": "g1",
                "kind": "combinational",
                "graph": graph,
                "payload_clues": [{"instance_name": "n2"}],
                "module_evidence": {"hypothesis": ["n1", "n2", "n3"]},
            }
        ],
        "references": [],
        "retrieval_matches": {},
        "library": [],
        "limitations": "Fixture",
    }
    identities = {"n1": "private_trigger", "n2": "private_payload", "n3": "private_host"}
    module = {
        "groups": [
            {
                "group_id": "g1",
                "kind": "combinational",
                "source_payload_group": "p1",
                "candidate_instances": list(identities.values()),
            }
        ]
    }
    samples = [
        {
            "node_id": i,
            "trojan_score": score,
            "secret_label": 1000,
            "netlist_path": "/private/path",
            "drivers": [{"instance_name": name, "cell_type": cell, "output_port": "Q"}],
        }
        for i, name, cell, score in [
            (0, "private_trigger", "and2s1", 0.4),
            (1, "private_host", "nor2s1", 0.1),
            (2, "private_payload", "xor2s1", 0.8),
            (3, "private_other", "and2s1", 0.05),
        ]
    ]
    payload_result = {
        "candidate_instances": ["private_trigger", "private_payload"],
        "settings": {"upstream_depth": 2, "downstream_depth": 2},
        "groups": [
            {
                "group_id": "p1",
                "seed_instances": ["private_payload"],
                "expanded_instances": ["private_trigger", "private_payload"],
                "accepted": True,
                "budget_truncated": False,
                "expansion_steps": [
                    {
                        "hop": 1,
                        "active_directions": ["upstream"],
                        "added_instances": ["private_trigger"],
                    }
                ],
            }
        ],
    }
    interface = {
        "target": "private_payload",
        "port": "DIN1",
        "role": "module_fanin",
        "signal": "private_signal",
    }
    recovery = [
        {
            "source_payload_group": "p1",
            "scope_instances": list(identities.values()),
            "new_scope_instances": ["private_host"],
            "accepted_new_instances": ["private_host"],
            "rejected_new_scope_instances": [],
            "interfaces": [interface],
            "proposals": [
                {
                    "instance_name": "private_host",
                    "interface": interface,
                    "reason": "missing_module_private_fanin",
                }
            ],
            "boundaries": [
                {
                    "signal": "private_signal",
                    "interface": interface,
                    "reason": "primary_input_or_multiple_drivers",
                }
            ],
            "budget_truncated": False,
            "settings": {"external_combinational_depth": 2},
        }
    ]
    frozen = stages.snapshot(
        payload, module, identities, payload_result, recovery, [2], [2, 0, 1, 3], samples
    )
    return (payload, identities, module, frozen)


class SnapshotTests(unittest.TestCase):

    def setUp(self):
        self.payload, self.identities, self.module, self.frozen = fixture()
        self.reader = stages.PipelineResults(self.frozen, self.payload)
        self.args = {"group_id": "g1", "stage": "all", "instance_ids": []}

    def test_real_stage_provenance_is_exposed_without_private_or_evaluation_fields(self):
        result = self.reader.lookup(self.args)
        rows = {r["instance_id"]: r for r in result["instances"]}
        self.assertEqual(
            [rows[n]["origin_stage"] for n in ("n1", "n2", "n3")],
            ["payloadchecker", "nhtd_gl", "modulechecker"],
        )
        self.assertEqual(rows["n3"]["predictions"][0]["rank"], 3)
        self.assertFalse(rows["n3"]["selected_initially"])
        self.assertEqual(result["modulechecker"]["recovery"][0]["accepted_new_instances"], ["n3"])
        encoded = json.dumps(result)
        for forbidden in (
            *self.identities.values(),
            "private_signal",
            "/private",
            '"node_id"',
            '"labels"',
            '"nt"',
            '"recall"',
            "secret_label",
            "ranked_node_ids",
        ):
            self.assertNotIn(forbidden, encoded)
        self.assertTrue(result["read_only"])

    def test_query_cannot_choose_files_labels_external_instances_or_unknown_groups(self):
        for delta in (
            {"path": "/private/path"},
            {"labels": True},
            {"group_id": "g2"},
            {"stage": "ground_truth"},
            {"instance_ids": ["context_00001"]},
            {"instance_ids": ["n1", "n1"]},
            {"instance_ids": [4]},
        ):
            with self.subTest(delta=delta):
                call = stages.execute_call(self.reader, json.dumps({**self.args, **delta}))
                self.assertEqual(call["result"]["status"], "error")

    def test_queries_leave_frozen_inputs_unchanged_and_obey_stage_and_byte_limits(self):
        before = runner.digest(self.frozen)
        result = self.reader.lookup({**self.args, "stage": "nhtd_gl", "instance_ids": ["n3"]})
        self.assertEqual([r["instance_id"] for r in result["instances"]], ["n3"])
        self.assertEqual(result["nhtd_gl"]["initial_predictions"], [])
        self.assertNotIn("modulechecker", result)
        result["instances"].clear()
        self.assertEqual(runner.digest(self.frozen), before)
        tiny = stages.PipelineResults(self.frozen, self.payload, max_result_bytes=100)
        error = stages.execute_call(tiny, json.dumps(self.args))["result"]
        self.assertEqual(error["status"], "error")
        self.assertIn("budget", error["error"])
