"""Prepare structural candidates and anonymous, frozen Generator inputs."""

from dataclasses import asdict, dataclass
from pathlib import Path
import re

from .detection import read_predictions
from .generator import stage_results
from .module_checker import check_modules
from .payload_checker import check_payloads
from .ports import OUTPUT_PORTS
from .rag.graph_similarity import circuit_graph
from .rag.retriever import StructuralRetriever
from .sequential import load_cell_library


@dataclass(frozen=True)
class StructureConfig:
    upstream_depth: int = 2
    downstream_depth: int = 2
    output_tolerance: int = 1
    instance_budget: int = 100
    module_recovery_depth: int = 2
    max_new_instances: int = 100

    def __post_init__(self):
        for name, value in asdict(self).items():
            if type(value) is not int or value < (1 if name == "instance_budget" else 0):
                raise ValueError(f"Invalid structural budget: {name}")


def anonymize_candidates(cells, groups, library):
    """Replace instance/net identities while preserving edges and constants.

    Identity maps stay offline. Source paths, Verilog source statements, and
    line locations are excluded from the model-facing evidence.
    """
    names = sorted(cells)
    ids = {name: f"n{i:05d}" for i, name in enumerate(names, 1)}
    signals = sorted({s for cell in cells.values() for s in cell["port_connections"].values()})
    nets = {signal: f"s{i:05d}" for i, signal in enumerate(signals, 1)}
    literal = re.compile(r"(?:\d+)?'[sS]?[bBoOdDhH][0-9a-fA-FxXzZ?_]+|[01]")
    nets = {signal: signal if literal.fullmatch(signal) else anon for signal, anon in nets.items()}
    private_fields = {"source", "source_line", "source_end_line", "verilog_instance"}

    def sanitize(value):
        if isinstance(value, str):
            return ids.get(value, nets.get(value, value))
        if isinstance(value, list):
            return [sanitize(item) for item in value]
        if isinstance(value, dict):
            return {
                ids.get(key, nets.get(key, key)): sanitize(item)
                for key, item in value.items()
                if key not in private_fields
            }
        return value

    clean = []
    for group in groups:
        graph = circuit_graph(
            [
                {
                    "instance_name": ids[name],
                    "cell_type": cells[name]["cell_type"],
                    "port_connections": {
                        port: nets[signal]
                        for port, signal in cells[name]["port_connections"].items()
                    },
                }
                for name in group["candidate_instances"]
            ]
        )
        clean.append(
            {
                "group_id": group["group_id"],
                "kind": group["kind"],
                "graph": graph,
                "payload_clues": sanitize(group["payload_clues"]),
                "module_evidence": sanitize(group["module_evidence"]),
            }
        )
    members = {name for group in groups for name in group["candidate_instances"]}
    payload = {
        "candidate_groups": clean,
        "references": [],
        "retrieval_matches": {},
        "library": [
            {
                "cell_type": kind,
                "definition": library[kind].get("definition", ""),
                "control_semantics": library[kind].get("control_semantics", {}),
                "primitive_definitions": library[kind].get("primitive_definitions", {}),
            }
            for kind in sorted({cells[n]["cell_type"] for n in members})
            if kind in library
        ],
        "limitations": "Bounded structural hypotheses only; unresolved interfaces do not prove benignity.",
    }
    return payload, {ids[name]: name for name in members}


def prepare_candidates(
    report,
    variant,
    top_k,
    trojan_type,
    config=None,
    *,
    netlist=None,
    library=None,
    rag_snapshot=None,
    mentor_index=None,
    suite=None,
    graph_store=None,
):
    if trojan_type not in ("combinational", "fsm", "counter", "auto"):
        raise ValueError("Use combinational, fsm, counter, or auto")
    if rag_snapshot is not None and mentor_index is not None:
        raise ValueError("Choose either structural references or a GraphMentor index")
    if trojan_type == "auto" and (mentor_index is None or suite is None):
        raise ValueError("Automatic type retrieval requires a GraphMentor index and explicit suite")
    config = config or StructureConfig()
    host = variant.split("_T")[0].lower()
    suite = suite or ("TRIT-TC" if trojan_type == "combinational" else "TRIT-TS")
    detection = read_predictions(report, variant, top_k, netlist)
    cells, seeds = detection["cells"], detection["seeds"]
    if library:
        specs = load_cell_library(Path(library), {cell["cell_type"] for cell in cells.values()})
    elif trojan_type in ("combinational", "auto"):
        specs = {
            cell["cell_type"]: {
                "directions": {
                    p: "output" if p.upper() in OUTPUT_PORTS else "input"
                    for p in cell["port_connections"]
                },
                "sequential": False,
                "definition": "",
                "control_semantics": {},
                "primitive_definitions": {},
            }
            for cell in cells.values()
        }
        if any("dff" in kind.lower() or "latch" in kind.lower() for kind in specs):
            raise ValueError("Supply a cell library for netlists containing sequential cells")
    else:
        raise ValueError("FSM and Counter analysis requires --library")
    payload_result = check_payloads(
        cells,
        seeds,
        config.upstream_depth,
        config.downstream_depth,
        config.output_tolerance,
        config.instance_budget,
        stop_at_registers=trojan_type == "combinational",
    )
    type_hypotheses, type_evidence, bank = {}, {}, None
    if mentor_index is not None:
        from .rag.learned_retriever import EmbeddingBank

        bank = EmbeddingBank(mentor_index, host, suite)
    if trojan_type == "auto":
        for component in payload_result["groups"]:
            if not component["accepted"]:
                continue
            graph = circuit_graph(
                [
                    {
                        "instance_name": n,
                        "cell_type": cells[n]["cell_type"],
                        "port_connections": cells[n]["port_connections"],
                    }
                    for n in component["expanded_instances"]
                ]
            )
            hit = bank.retrieve(graph, 1)[0]
            category = hit["entry"]["categories"][0]
            type_hypotheses[component["group_id"]] = {
                "combinational": "Comb.",
                "fsm": "FSM",
                "counter": "Counter",
            }[category]
            type_evidence[component["group_id"]] = {
                "type": category,
                "reference_id": hit["entry"]["reference_id"],
                "score": hit["score"],
            }
    kind = {"combinational": "Comb.", "fsm": "FSM", "counter": "Counter", "auto": "Comb."}[
        trojan_type
    ]
    graph_key = None
    if graph_store is not None:
        full_graph = circuit_graph(
            [
                {
                    "instance_name": name,
                    "cell_type": cell["cell_type"],
                    "port_connections": cell["port_connections"],
                }
                for name, cell in cells.items()
            ]
        )
        graph_key = graph_store.import_graph(full_graph, variant)
    groups = check_modules(
        cells,
        seeds,
        payload_result,
        kind,
        specs,
        config.module_recovery_depth,
        config.max_new_instances,
        graph_store=graph_store,
        graph_key=graph_key,
        type_hypotheses=type_hypotheses,
    )
    module = {"groups": groups, "type_retrieval": type_evidence}
    payload, identities = anonymize_candidates(cells, groups, specs)
    # References enrich ModuleChecker evidence; they are not an automatic
    # match-based acceptance/rejection rule or behavioral labels.
    if rag_snapshot is not None or mentor_index is not None:
        if mentor_index is not None:
            from .rag.learned_retriever import LearnedRetriever

            retriever = LearnedRetriever(mentor_index, payload, host, suite)
        else:
            retriever = StructuralRetriever(rag_snapshot, payload, host, suite)
        references, matches = {}, {}
        module["references"] = {}
        for group in payload["candidate_groups"]:
            result = retriever.search(
                {
                    "group_id": group["group_id"],
                    "query": "payload trigger control state feedback enable decoder module completeness",
                    "trojan_type": group["kind"],
                    "top_k": 3,
                }
            )
            module["references"][group["group_id"]] = result["matches"]
            matches[group["group_id"]] = []
            for reference in result["matches"]:
                references[reference["reference_id"]] = reference
                matches[group["group_id"]].append(
                    {
                        "reference_id": reference["reference_id"],
                        "score": reference["score"],
                        "method": result["method"],
                    }
                )
        payload["references"] = list(references.values())
        payload["retrieval_matches"] = matches
    frozen = stage_results.snapshot(
        payload,
        module,
        identities,
        payload_result,
        payload_result["module_recovery"],
        detection["initial_node_ids"],
        detection["ranked_node_ids"],
        detection["samples"],
    )
    instance_nodes = {
        name: sorted(
            {
                row["node_id"]
                for row in detection["samples"]
                for driver in row["drivers"]
                if driver["instance_name"] == name
            }
        )
        for name in identities.values()
    }
    return {
        "generator_input": payload,
        "pipeline_results": frozen,
        "payload_result": payload_result,
        "module_result": module,
        "identity_map": identities,
        "node_mapping": instance_nodes,
        "metadata": {
            "variant": variant,
            "host": host,
            "suite": suite,
            "trojan_type": trojan_type,
            "requested_top_k": top_k,
            "initial_node_ids": detection["initial_node_ids"],
            "ranked_node_ids": detection["ranked_node_ids"],
            "unmapped_initial_node_ids": detection["unmapped_initial_node_ids"],
            "netlist_sha256": detection["netlist_sha256"],
            "structure_config": asdict(config),
            "mentor_index_sha256": bank.fingerprint if bank is not None else None,
            "graph_key": graph_key,
        },
    }
