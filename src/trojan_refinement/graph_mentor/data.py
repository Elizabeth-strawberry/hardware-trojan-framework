"""Annotated training graphs and identity-independent gate features."""

from collections import Counter
import json
from pathlib import Path

from ..rag.graph_similarity import cell_family, circuit_graph

TYPES = ("combinational", "fsm", "counter")
FAMILIES = ("and", "nand", "or", "nor", "xor", "xnor", "inv", "buf", "mux", "dff", "latch", "other")
FEATURE_NAMES = (*FAMILIES, "log_fanin", "log_fanout", "input_ports", "enable_ports")


def validate_graph(graph):
    nodes = graph.get("nodes", [])
    ids = [node["id"] for node in nodes]
    if not ids or any(not isinstance(n, str) or not n for n in ids) or len(set(ids)) != len(ids):
        raise ValueError("Graphs require distinct nonempty node IDs")
    rebuilt = circuit_graph(
        [
            {"instance_name": n["id"], "cell_type": n["cell_type"], "port_connections": n["ports"]}
            for n in nodes
        ]
    )
    if "edges" in graph and graph["edges"] != rebuilt["edges"]:
        raise ValueError("Graph edges do not agree with gate port connections")
    return rebuilt


def load_corpus(path):
    corpus = json.loads(Path(path).read_text())
    if corpus.get("schema") != "trojan_module_corpus_v1" or not corpus.get("entries"):
        raise ValueError("Expected a nonempty trojan_module_corpus_v1 dataset")
    entries, seen, host_splits = [], set(), {}
    for item in corpus["entries"]:
        entry = dict(item)
        for field in ("id", "host", "suite", "trojan_type"):
            if not isinstance(entry.get(field), str) or not entry[field]:
                raise ValueError("Missing corpus field: " + field)
        if entry["id"] in seen or entry["trojan_type"] not in TYPES:
            raise ValueError("Duplicate reference ID or unsupported Trojan type")
        seen.add(entry["id"])
        entry["host"] = entry["host"].lower()
        entry["split"] = entry.get("split", "train")
        if entry["split"] not in ("train", "validation", "test"):
            raise ValueError("Unknown corpus split")
        key = entry["host"]
        if key in host_splits and host_splits[key] != entry["split"]:
            raise ValueError("All variants of a host must belong to the same split")
        host_splits[key] = entry["split"]
        entry["graph"] = validate_graph(entry["graph"])
        labels = entry.get("module_labels", {})
        if set(labels) != {n["id"] for n in entry["graph"]["nodes"]} or any(
            not isinstance(label, str) or not label.strip() for label in labels.values()
        ):
            raise ValueError("Provide an explicit functional module label for every training node")
        entries.append(entry)
    return sorted(entries, key=lambda e: e["id"])


def gate_family(kind):
    family = cell_family(kind)
    if family.startswith(("mx", "mux")):
        return "mux"
    if family.startswith(("dff", "sdff", "ff")):
        return "dff"
    for prefix in ("xnor", "xor", "nand", "nor", "and", "or", "inv", "buf", "latch"):
        if family.startswith(prefix):
            return prefix
    return "other"


def graph_tensor(graph):
    import torch
    from torch_geometric.data import Data

    graph = validate_graph(graph)
    ids = [n["id"] for n in graph["nodes"]]
    lookup = {name: i for i, name in enumerate(ids)}
    incoming = Counter(
        e["target"]
        for e in graph["edges"]
        if e["target_port"].upper() not in ("CLK", "CK", "CLOCK")
    )
    outgoing = Counter(
        e["source"]
        for e in graph["edges"]
        if e["target_port"].upper() not in ("CLK", "CK", "CLOCK")
    )
    features = []
    for node in graph["nodes"]:
        family = gate_family(node["cell_type"])
        ports = [p.upper() for p in node["ports"]]
        features.append(
            [float(family == f) for f in FAMILIES]
            + [
                float(incoming[node["id"]]),
                float(outgoing[node["id"]]),
                float(
                    sum(
                        p not in ("Q", "QN", "Y", "Z", "ZN", "O", "OUT", "CLK", "CK", "CLOCK")
                        for p in ports
                    )
                ),
                float(sum(p in ("EB", "EN", "E", "CE", "ENABLE") for p in ports)),
            ]
        )
    x = torch.tensor(features, dtype=torch.float32)
    x[:, len(FAMILIES) :] = torch.log1p(x[:, len(FAMILIES) :])
    pairs = [
        (lookup[e["source"]], lookup[e["target"]])
        for e in graph["edges"]
        if e["target_port"].upper() not in ("CLK", "CK", "CLOCK")
    ]
    edge_index = (
        torch.tensor(pairs, dtype=torch.long).t().contiguous()
        if pairs
        else torch.empty((2, 0), dtype=torch.long)
    )
    return Data(x=x, edge_index=edge_index), ids
