import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from trojan_refinement.detection import read_predictions
from trojan_refinement.evaluation import evaluate_nodes
from trojan_refinement.generator import stage_results
from trojan_refinement.pipeline import StructureConfig, anonymize_candidates, prepare_candidates
from trojan_refinement.storage import load_prepared, write_prepared

spec = importlib.util.spec_from_file_location(
    "create_demo", Path(__file__).parents[1] / "examples/create_demo.py"
)
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


class PipelineTests(unittest.TestCase):
    def test_detector_candidates_seed_recovery_and_labels_stay_offline(self):
        with tempfile.TemporaryDirectory() as temp:
            report = demo.create_demo(temp)
            raw = json.loads(report.read_text())
            raw["ranked_samples"][0]["private_label"] = "DO_NOT_EXPOSE"
            report.write_text(json.dumps(raw))
            result = prepare_candidates(report, "synthetic_demo", 3, "combinational")
            self.assertEqual(result["metadata"]["initial_node_ids"], [0, 1, 2])
            self.assertTrue(result["module_result"]["groups"])
            self.assertNotIn("consumer", result["identity_map"].values())
            self.assertEqual(
                set(result["identity_map"].values()),
                {"host", "trigger", "payload", "collect_a", "collect_b"},
            )
            public = json.dumps([result["generator_input"], result["pipeline_results"]])
            for forbidden in (
                "DO_NOT_EXPOSE",
                temp,
                '"node_id"',
                '"labels"',
                '"nt"',
                '"host_data"',
                '"collect_a"',
            ):
                self.assertNotIn(forbidden, public)
            reader = stage_results.PipelineResults(
                result["pipeline_results"], result["generator_input"]
            )
            group = result["generator_input"]["candidate_groups"][0]
            frozen = reader.lookup(
                {"group_id": group["group_id"], "stage": "all", "instance_ids": []}
            )
            self.assertTrue(frozen["read_only"])
            self.assertEqual(
                {r["origin_stage"] for r in frozen["instances"]}, {"nhtd_gl", "payloadchecker"}
            )

    def test_mapped_results_and_frozen_inputs_are_verified(self):
        with tempfile.TemporaryDirectory() as temp:
            report = demo.create_demo(temp)
            result = prepare_candidates(report, "synthetic_demo", 3, "combinational")
            folder = Path(temp) / "prepared"
            write_prepared(folder, result)
            self.assertEqual(load_prepared(folder), json.loads(json.dumps(result)))
            with self.assertRaises(ValueError):
                write_prepared(folder, result)
            (folder / "generator_input.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "changed"):
                load_prepared(folder)
            path = Path(temp) / "synthetic_demo.v"
            path.write_text(path.read_text() + "// changed")
            with self.assertRaises(ValueError):
                read_predictions(report, "synthetic_demo", 3)

    def test_stable_ties_and_invalid_scores(self):
        with tempfile.TemporaryDirectory() as temp:
            report = demo.create_demo(temp)
            raw = json.loads(report.read_text())
            for row in raw["ranked_samples"]:
                row["trojan_score"] = 0.5
            report.write_text(json.dumps(raw))
            self.assertEqual(
                read_predictions(report, "synthetic_demo", 2)["initial_node_ids"], [0, 1]
            )
            raw["ranked_samples"][0]["trojan_score"] = float("nan")
            report.write_text(json.dumps(raw))
            with self.assertRaises(ValueError):
                read_predictions(report, "synthetic_demo", 2)

    def test_anonymization_preserves_source_edges_and_constant_semantics(self):
        cells = {
            "first": {
                "cell_type": "and2s1",
                "port_connections": {"DIN1": "1'b0", "DIN2": "in", "Q": "q"},
            },
            "last": {
                "cell_type": "xor2s1",
                "port_connections": {"DIN1": "q", "DIN2": "host", "Q": "out"},
            },
        }
        groups = [
            {
                "group_id": "g1",
                "kind": "combinational",
                "candidate_instances": ["first", "last"],
                "payload_clues": [],
                "module_evidence": {"source": "/private/path", "members": ["first"]},
            }
        ]
        payload, identities = anonymize_candidates(cells, groups, {})
        graph = payload["candidate_groups"][0]["graph"]
        self.assertEqual(
            graph["edges"][0]["source"], next(k for k, v in identities.items() if v == "first")
        )
        self.assertIn("1'b0", json.dumps(graph))
        self.assertNotIn("/private/path", json.dumps(payload))

    def test_metrics_use_exact_candidate_size_and_report_lost_true_positives(self):
        labels = {0: 1, 1: 0, 2: 1, 3: 0}
        full = evaluate_nodes([0, 1, 2], [1, 3, 0, 2], labels)
        pruned = evaluate_nodes([0], [1, 3, 0, 2], labels)
        self.assertEqual(full["recall"], 1)
        self.assertEqual(pruned["recall"], 0.5)
        self.assertEqual(full["same_cost_nhtd_recall"], 0.5)
        self.assertEqual(pruned["inspection_cost"], 0.5)
        with self.assertRaises(ValueError):
            evaluate_nodes([5], [1, 3, 0, 2], labels)

    def test_invalid_structural_budgets(self):
        for kwargs in ({"upstream_depth": -1}, {"instance_budget": 0}, {"downstream_depth": True}):
            with self.assertRaises(ValueError):
                StructureConfig(**kwargs)

    def test_modulechecker_receives_anonymous_training_references(self):
        from test_retrieval import bank

        with tempfile.TemporaryDirectory() as temp:
            report = demo.create_demo(temp)
            result = prepare_candidates(
                report, "synthetic_demo", 3, "combinational", rag_snapshot=bank()
            )
            self.assertTrue(result["generator_input"]["references"])
            self.assertTrue(result["module_result"]["references"])
            public = json.dumps(result["generator_input"])
            self.assertNotIn("c5315", public)
            self.assertNotIn("wrong_suite", public)


if __name__ == "__main__":
    unittest.main()
