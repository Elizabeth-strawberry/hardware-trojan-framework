"""GraphMentor graph retrieval and per-node functional module assignment."""

from pathlib import Path
import shutil

from .retriever import StructuralRetriever, _anonymous_graph
from .vector_index import CosineIndex
from ..storage import file_hash, read_json, write_json


def build_embedding_bank(dataset, checkpoint, output):
    import numpy as np
    import torch
    from torch.nn import functional as F
    from ..graph_mentor.data import graph_tensor, load_corpus
    from ..graph_mentor.training import load_mentor

    model, training = load_mentor(checkpoint)
    entries = [e for e in load_corpus(dataset) if e["id"] in training["training_ids"]]
    if file_hash(dataset) != training["dataset_sha256"] or len(entries) != len(
        training["training_ids"]
    ):
        raise ValueError("Build the bank from the checkpoint's exact training corpus")
    directory = Path(output)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Use an empty embedding-bank directory")
    directory.mkdir(parents=True, exist_ok=True)
    references, graph_vectors, module_vectors, modules = [], [], [], []
    with torch.no_grad():
        for number, entry in enumerate(entries, 1):
            data, ids = graph_tensor(entry["graph"])
            nodes, graph = model(data.x, data.edge_index)
            ref_id = f"rag_ref{number:04d}"
            anonymous = _anonymous_graph(entry["graph"], f"kr{number:04d}")
            references.append(
                {
                    "reference_id": ref_id,
                    "host": entry["host"],
                    "suite": entry["suite"],
                    "categories": [entry["trojan_type"]],
                    "graph": anonymous,
                }
            )
            graph_vectors.append(graph[0].numpy())
            for role in sorted(set(entry["module_labels"].values())):
                positions = [i for i, n in enumerate(ids) if entry["module_labels"][n] == role]
                module_vectors.append(F.normalize(nodes[positions].mean(dim=0), dim=0).numpy())
                modules.append({"reference_id": ref_id, "role": role})
    CosineIndex(np.stack(graph_vectors)).save(directory / "graphs.faiss")
    CosineIndex(np.stack(module_vectors)).save(directory / "modules.faiss")
    shutil.copyfile(checkpoint, directory / "mentor.pt")
    write_json(directory / "references.json", {"entries": references, "modules": modules})
    manifest = {
        "schema": "trojan_embedding_bank_v1",
        "training": training,
        "sha256": {
            name: file_hash(directory / name)
            for name in ("graphs.faiss", "modules.faiss", "mentor.pt", "references.json")
        },
    }
    write_json(directory / "manifest.json", manifest)
    return manifest


class EmbeddingBank:
    def __init__(self, directory, held_out_host, suite):
        from ..graph_mentor.training import load_mentor

        directory = Path(directory)
        manifest = read_json(directory / "manifest.json")
        names = {"graphs.faiss", "modules.faiss", "mentor.pt", "references.json"}
        if manifest.get("schema") != "trojan_embedding_bank_v1" or set(manifest["sha256"]) != names:
            raise ValueError("Unknown embedding-bank manifest")
        for name in names:
            if file_hash(directory / name) != manifest["sha256"][name]:
                raise ValueError("Embedding bank changed: " + name)
        self.fingerprint = file_hash(directory / "manifest.json")
        self.model, training = load_mentor(directory / "mentor.pt")
        if training != manifest["training"]:
            raise ValueError("Checkpoint training manifest mismatch")
        held_out_host = held_out_host.lower()
        if (
            held_out_host in training["training_hosts"]
            or held_out_host not in training["excluded_hosts"]
        ):
            raise ValueError(
                "The tested host must be explicitly excluded from GraphMentor training"
            )
        metadata = read_json(directory / "references.json")
        self.entries, self.modules = metadata["entries"], metadata["modules"]
        self.graph_index = CosineIndex.load(directory / "graphs.faiss")
        self.module_index = CosineIndex.load(directory / "modules.faiss")
        if (
            len(self.entries) != self.graph_index.index.ntotal
            or len(self.modules) != self.module_index.index.ntotal
        ):
            raise ValueError("Embedding rows do not match reference metadata")
        self.eligible = [
            i
            for i, e in enumerate(self.entries)
            if e["host"] != held_out_host and e["suite"] == suite
        ]
        if not self.eligible:
            raise ValueError("No same-suite training references remain for the tested host")

    def encode(self, graph):
        import torch
        from ..graph_mentor.data import graph_tensor

        data, ids = graph_tensor(graph)
        with torch.no_grad():
            nodes, pooled = self.model(data.x, data.edge_index)
        return ids, nodes.numpy(), pooled.numpy()

    def retrieve(self, graph, top_k=3, trojan_type="any"):
        ids, nodes, pooled = self.encode(graph)
        allowed = [
            i
            for i in self.eligible
            if trojan_type == "any" or trojan_type in self.entries[i]["categories"]
        ]
        result = []
        for row, similarity in self.graph_index.search(pooled, top_k, allowed)[0]:
            entry = self.entries[row]
            module_rows = [
                i for i, m in enumerate(self.modules) if m["reference_id"] == entry["reference_id"]
            ]
            assignments = self.module_index.search(nodes, 1, module_rows)
            result.append(
                {
                    "entry": entry,
                    "score": similarity,
                    "module_assignments": [
                        {
                            "instance_id": name,
                            "role": self.modules[hits[0][0]]["role"],
                            "score": hits[0][1],
                        }
                        for name, hits in zip(ids, assignments)
                    ],
                }
            )
        return result


class LearnedRetriever(StructuralRetriever):
    method = "graph_mentor_faiss_cosine_plus_structural_keywords"

    def __init__(self, directory, payload, held_out_host, suite, max_result_bytes=131072):
        self.bank = EmbeddingBank(directory, held_out_host, suite)
        super().__init__(
            {"schema": "generator_structural_rag_v1", "entries": self.bank.entries},
            payload,
            held_out_host,
            suite,
            max_result_bytes,
        )
        self.learned = {}

    def compare(self, group_id, entry):
        if group_id not in self.learned:
            self.learned[group_id] = {
                hit["entry"]["reference_id"]: {
                    "score": hit["score"],
                    "module_assignments": hit["module_assignments"],
                }
                for hit in self.bank.retrieve(self.groups[group_id], len(self.entries))
            }
        return self.learned[group_id][entry["reference_id"]]
