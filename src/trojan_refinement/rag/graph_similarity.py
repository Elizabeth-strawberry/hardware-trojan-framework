"""Directed Weisfeiler-Lehman graph features and gate-family similarity for structural references."""

from collections import Counter, defaultdict
import hashlib
import json
import math
import re

OUTPUTS = {"Q", "QN", "Y", "Z", "ZN", "O", "OUT"}


def parse_instances(text):
    text = re.sub("/\\*.*?\\*/|//[^\\n]*", "", text, flags=re.S)
    instances = []
    for match in re.finditer(
        "^\\s*(\\w+)\\s+(\\\\[^\\s(]+|[A-Za-z_][\\w$]*)\\s*\\((.*?)\\)\\s*;", text, re.M | re.S
    ):
        instances.append(
            {
                "instance_name": match[2],
                "cell_type": match[1],
                "verilog_instance": match.group().strip(),
                "port_connections": dict(
                    (
                        (p, s.strip())
                        for p, s in re.findall("\\.([\\w$]+)\\s*\\(\\s*([^()]*)\\s*\\)", match[3])
                    )
                ),
            }
        )
    return instances


def cell_family(cell):
    family = re.sub("s\\d+$", "", cell.lower())
    return {
        "i1": "inv",
        "ib1": "inv",
        "hi1": "inv",
        "nnd2": "nand2",
        "nnd3": "nand3",
        "nnd4": "nand4",
        "nnd5": "nand5",
        "xnr2": "xnor2",
    }.get(family, family)


def circuit_graph(instances):
    nodes, drivers, consumers = ([], defaultdict(list), defaultdict(list))
    for instance in instances:
        name = instance["instance_name"]
        ports = instance["port_connections"]
        nodes.append(
            {
                "id": name,
                "cell_type": instance["cell_type"],
                "family": cell_family(instance["cell_type"]),
                "ports": ports,
            }
        )
        for port, signal in ports.items():
            (drivers if port.upper() in OUTPUTS else consumers)[signal].append((name, port))
    edges, inputs, outputs = ([], [], [])
    for signal in sorted(drivers.keys() | consumers.keys()):
        for a, ap in drivers[signal]:
            for b, bp in consumers[signal]:
                edges.append(
                    {
                        "source": a,
                        "target": b,
                        "source_port": ap,
                        "target_port": bp,
                        "signal": signal,
                    }
                )
        if not drivers[signal]:
            inputs.append({"signal": signal, "consumers": consumers[signal]})
        if not consumers[signal]:
            outputs.append({"signal": signal, "drivers": drivers[signal]})
    return {
        "directed": True,
        "nodes": nodes,
        "edges": edges,
        "boundary_inputs": inputs,
        "boundary_outputs": outputs,
    }


def _features(graph):
    """Compare directed WL features without instance names, net names, or effect labels."""
    labels = {n["id"]: n["family"] for n in graph["nodes"]}
    incoming, outgoing = (defaultdict(list), defaultdict(list))
    for e in graph["edges"]:
        role = re.sub("^DIN\\d*$", "DATA", e["target_port"].upper())
        edge_role = e["source_port"].upper() + ":" + role
        incoming[e["target"]].append((e["source"], edge_role))
        outgoing[e["source"]].append((e["target"], edge_role))
    features = Counter(("gate:" + v for v in labels.values()))
    for iteration in range(2):
        updated = {}
        for name, label in labels.items():
            signature = (
                label,
                sorted(((p, labels[n]) for n, p in incoming[name])),
                sorted(((p, labels[n]) for n, p in outgoing[name])),
            )
            value = hashlib.sha256(json.dumps(signature).encode()).hexdigest()
            updated[name] = value
            if incoming[name] or outgoing[name]:
                features[f"wl{iteration + 1}:" + value] += 1
        labels = updated
    return features


def _cosine(q, r):
    dot = sum((v * r.get(k, 0) for k, v in q.items()))
    norm = math.sqrt(sum((v * v for v in q.values())) * sum((v * v for v in r.values())))
    return dot / norm if norm else 0.0


def graph_similarity(query, reference):
    q, r = (_features(query), _features(reference))

    def part(features, prefix):
        return {k: v for k, v in features.items() if k.startswith(prefix)}

    gate = _cosine(part(q, "gate:"), part(r, "gate:"))
    wl1 = _cosine(part(q, "wl1:"), part(r, "wl1:"))
    wl2 = _cosine(part(q, "wl2:"), part(r, "wl2:"))
    structure = 0.5 * wl1 + 0.5 * wl2
    score = 0.2 * gate + 0.8 * structure
    qtypes = Counter((n["family"] for n in query["nodes"]))
    rtypes = Counter((n["family"] for n in reference["nodes"]))
    return {
        "score": score,
        "gate_similarity": gate,
        "structure_similarity": structure,
        "structure_components": {"wl1_similarity": wl1, "wl2_similarity": wl2},
        "weights": {
            "gate": 0.2,
            "structure": 0.8,
            "within_structure_wl1": 0.5,
            "within_structure_wl2": 0.5,
        },
        "structure_evidence": (
            "matching_neighborhood_features"
            if structure > 0
            else "no_matching_neighborhood_features"
        ),
        "method": "separate_gate_and_directed_WL_h2_weighted_v2",
        "query_nodes": len(query["nodes"]),
        "reference_nodes": len(reference["nodes"]),
        "query_edges": len(query["edges"]),
        "reference_edges": len(reference["edges"]),
        "shared_gate_counts": dict(qtypes & rtypes),
        "query_only_gate_counts": dict(qtypes - rtypes),
        "reference_only_gate_counts": dict(rtypes - qtypes),
    }
