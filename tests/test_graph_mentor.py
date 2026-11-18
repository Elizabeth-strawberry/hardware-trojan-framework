import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from trojan_refinement.graph_mentor.data import load_corpus, validate_graph
from trojan_refinement.rag.graph_similarity import circuit_graph
from trojan_refinement.storage import write_json

HAS_TORCH = (
    importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("torch_geometric") is not None
)
HAS_FAISS = HAS_TORCH and importlib.util.find_spec("faiss") is not None


def annotated_corpus(path):
    graph = circuit_graph(
        [
            {
                "instance_name": "a",
                "cell_type": "and2s1",
                "port_connections": {"DIN1": "i", "DIN2": "j", "Q": "x"},
            },
            {
                "instance_name": "b",
                "cell_type": "nor2s1",
                "port_connections": {"DIN1": "k", "DIN2": "l", "Q": "y"},
            },
            {
                "instance_name": "t",
                "cell_type": "and2s1",
                "port_connections": {"DIN1": "x", "DIN2": "y", "Q": "condition"},
            },
            {
                "instance_name": "p",
                "cell_type": "xor2s1",
                "port_connections": {"DIN1": "condition", "DIN2": "host", "Q": "out"},
            },
        ]
    )
    entries = [
        {
            "id": f"reference_{i}",
            "host": f"reference_host_{i}",
            "suite": "TRIT-TC",
            "split": "train",
            "trojan_type": "combinational",
            "graph": graph,
            "module_labels": {"a": "collection", "b": "collection", "t": "trigger", "p": "payload"},
        }
        for i in (1, 2)
    ]
    write_json(path, {"schema": "trojan_module_corpus_v1", "entries": entries})
    return entries


class CorpusTests(unittest.TestCase):
    def test_module_labels_are_required_and_not_inferred_from_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "corpus.json"
            entries = annotated_corpus(path)
            del entries[0]["module_labels"]["a"]
            write_json(path, {"schema": "trojan_module_corpus_v1", "entries": entries})
            with self.assertRaisesRegex(ValueError, "explicit functional module label"):
                load_corpus(path)

    def test_variants_of_one_host_cannot_cross_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "corpus.json"
            entries = annotated_corpus(path)
            entries[1].update(host=entries[0]["host"], split="test")
            write_json(path, {"schema": "trojan_module_corpus_v1", "entries": entries})
            with self.assertRaisesRegex(ValueError, "same split"):
                load_corpus(path)

    def test_edges_must_match_the_port_level_graph(self):
        with tempfile.TemporaryDirectory() as tmp:
            graph = annotated_corpus(Path(tmp) / "corpus.json")[0]["graph"]
            graph["edges"][0]["source"] = "p"
            with self.assertRaisesRegex(ValueError, "port connections"):
                validate_graph(graph)


@unittest.skipUnless(HAS_TORCH, "Install the research extra to test GraphMentor")
class GraphMentorTests(unittest.TestCase):
    def test_metric_training_requires_positive_and_negative_pairs_within_one_type(self):
        from trojan_refinement.graph_mentor.training import train_mentor

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "corpus.json"
            entries = annotated_corpus(path)
            entries[0]["module_labels"] = dict.fromkeys(entries[0]["module_labels"], "collection")
            entries[1]["trojan_type"] = "fsm"
            entries[1]["module_labels"] = {name: name for name in entries[1]["module_labels"]}
            write_json(path, {"schema": "trojan_module_corpus_v1", "entries": entries})
            with self.assertRaisesRegex(ValueError, "within a type"):
                train_mentor(path, Path(tmp) / "mentor.pt", exclude_hosts=["target"], epochs=1)

    def test_fanin_fanout_weights_are_independent_and_direction_matters(self):
        import torch
        from trojan_refinement.graph_mentor.data import graph_tensor
        from trojan_refinement.graph_mentor.model import GraphMentor

        torch.manual_seed(2)
        with tempfile.TemporaryDirectory() as tmp:
            graph = annotated_corpus(Path(tmp) / "corpus.json")[0]["graph"]
            data, _ = graph_tensor(graph)
            model = GraphMentor(hidden_dim=8, num_layers=1, dropout=0).eval()
            self.assertIsNot(model.fanin[0].lin_l.weight, model.fanout[0].lin_l.weight)
            nodes, pooled = model(data.x, data.edge_index)
            reversed_nodes, _ = model(data.x, data.edge_index.flip(0))
            self.assertEqual(tuple(nodes.shape), (4, 16))
            self.assertEqual(tuple(pooled.shape), (1, 16))
            self.assertFalse(torch.allclose(nodes, reversed_nodes))

    def test_contrastive_loss_respects_module_labels_and_type_boundaries(self):
        import torch
        from trojan_refinement.graph_mentor.model import module_contrastive_loss

        embeddings = torch.tensor(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], requires_grad=True
        )
        types = torch.tensor([0, 0, 0, 1])
        labels = torch.tensor([0, 0, 1, 2])
        good = module_contrastive_loss(embeddings, labels, types)
        bad = module_contrastive_loss(embeddings, torch.tensor([0, 1, 0, 2]), types)
        self.assertAlmostEqual(float(good.detach()), 0.0)
        self.assertGreater(float(bad.detach()), 0.0)
        bad.backward()
        self.assertGreater(float(embeddings.grad.abs().sum()), 0.0)


@unittest.skipUnless(HAS_FAISS, "Install the research extra to test learned retrieval")
class LearnedRetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from trojan_refinement.graph_mentor.training import train_mentor
        from trojan_refinement.rag.learned_retriever import build_embedding_bank

        torch.set_num_threads(1)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.dataset = cls.root / "corpus.json"
        cls.entries = annotated_corpus(cls.dataset)
        cls.checkpoint = cls.root / "mentor.pt"
        cls.training = train_mentor(
            cls.dataset,
            cls.checkpoint,
            exclude_hosts=["target_host", "synthetic_demo"],
            epochs=3,
            hidden_dim=8,
            num_layers=1,
        )
        cls.bank = cls.root / "bank"
        build_embedding_bank(cls.dataset, cls.checkpoint, cls.bank)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_faiss_filters_before_selecting_neighbors(self):
        from trojan_refinement.rag.vector_index import CosineIndex

        index = CosineIndex([[1, 0], [0.9, 0.1], [0, 1]])
        self.assertEqual(index.search([[1, 0]], 1, [2])[0][0][0], 2)
        with self.assertRaisesRegex(ValueError, "Zero query"):
            index.search([[0, 0]], 1)

    def test_checkpoint_and_index_are_usable_without_training_labels_at_query_time(self):
        from trojan_refinement.rag.learned_retriever import EmbeddingBank

        bank = EmbeddingBank(self.bank, "target_host", "TRIT-TC")
        graph = self.entries[0]["graph"]
        hit = bank.retrieve(graph, 1)[0]
        self.assertGreater(hit["score"], 0.999)
        self.assertEqual(len(hit["module_assignments"]), 4)
        self.assertEqual(
            {a["role"] for a in hit["module_assignments"]}, {"collection", "trigger", "payload"}
        )
        self.assertNotIn("module_labels", json.dumps(bank.entries))

    def test_trained_or_undeclared_test_hosts_are_rejected(self):
        from trojan_refinement.rag.learned_retriever import EmbeddingBank

        for host in ("reference_host_1", "undeclared_target"):
            with self.subTest(host=host), self.assertRaisesRegex(ValueError, "explicitly excluded"):
                EmbeddingBank(self.bank, host, "TRIT-TC")
        with self.assertRaisesRegex(ValueError, "same-suite"):
            EmbeddingBank(self.bank, "target_host", "TRIT-TS")

    def test_modified_index_is_rejected_before_loading(self):
        import shutil
        from trojan_refinement.rag.learned_retriever import EmbeddingBank

        copy_path = self.root / "tampered_bank"
        shutil.copytree(self.bank, copy_path)
        with (copy_path / "graphs.faiss").open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "changed"):
            EmbeddingBank(copy_path, "target_host", "TRIT-TC")

    def test_auto_type_retrieval_is_connected_to_candidate_preparation(self):
        from test_pipeline import demo
        from trojan_refinement.pipeline import prepare_candidates
        from trojan_refinement.rag.learned_retriever import LearnedRetriever

        with tempfile.TemporaryDirectory() as tmp:
            report = demo.create_demo(tmp)
            prepared = prepare_candidates(
                report, "synthetic_demo", 3, "auto", mentor_index=self.bank, suite="TRIT-TC"
            )
            self.assertTrue(prepared["module_result"]["type_retrieval"])
            payload = prepared["generator_input"]
            self.assertTrue(payload["references"])
            self.assertIn("module_assignments", payload["references"][0])
            self.assertNotIn("reference_host", json.dumps(payload))
            retriever = LearnedRetriever(self.bank, payload, "synthetic_demo", "TRIT-TC")
            result = retriever.search(
                {
                    "group_id": payload["candidate_groups"][0]["group_id"],
                    "query": "trigger payload",
                    "trojan_type": "any",
                    "top_k": 1,
                }
            )
            self.assertEqual(result["status"], "success")
            self.assertIn("faiss", result["method"])
