"""Command-line preparation, read-only Generator review, and offline scoring."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

from .evaluation import evaluate_nodes
from .generator.contracts import validate_identity_map
from .generator.controller import refine_candidates
from .llm.client import ModelClient
from .llm.config import ModelConfig
from .pipeline import StructureConfig, prepare_candidates
from .rag.retriever import StructuralRetriever, freeze_knowledge_base
from .storage import file_hash, load_prepared, read_json, write_json, write_prepared


def prepare(args):
    graph_store = None
    if args.neo4j:
        from .rag.neo4j_store import Neo4jGraphStore

        graph_store = Neo4jGraphStore()
    config = StructureConfig(
        args.upstream_depth,
        args.downstream_depth,
        args.output_tolerance,
        args.instance_budget,
        args.module_recovery_depth,
        args.max_new_instances,
    )
    result = prepare_candidates(
        args.report,
        args.variant,
        args.top_k,
        args.trojan_type,
        config,
        netlist=args.netlist,
        library=args.library,
        rag_snapshot=read_json(args.rag) if args.rag else None,
        mentor_index=args.mentor_index,
        suite=args.suite,
        graph_store=graph_store,
    )
    write_prepared(args.output, result)
    print(f"Prepared {len(result['module_result']['groups'])} candidate groups. No API request.")


def review(args):
    output = Path(args.output)
    if output.exists() or output.with_suffix(output.suffix + ".progress.json").exists():
        raise ValueError("Use a new output path to preserve prior decisions and failed attempts")
    prepared = load_prepared(args.input)
    payload, identities = prepared["generator_input"], prepared["identity_map"]
    validate_identity_map(payload, prepared["module_result"], identities)
    config = ModelConfig.load(args.config)
    retriever = None
    if args.rag and args.mentor_index:
        raise ValueError("Choose either structural references or a GraphMentor index")
    if args.rag:
        metadata = prepared["metadata"]
        retriever = StructuralRetriever(
            read_json(args.rag),
            payload,
            metadata["host"],
            metadata["suite"],
            config.max_result_bytes,
        )
    if args.mentor_index:
        from .rag.learned_retriever import LearnedRetriever

        metadata = prepared["metadata"]
        retriever = LearnedRetriever(
            args.mentor_index, payload, metadata["host"], metadata["suite"], config.max_result_bytes
        )
        if metadata.get("mentor_index_sha256") not in (None, retriever.bank.fingerprint):
            raise ValueError("Review must use the GraphMentor bank recorded during preparation")
    audit = {
        "prepared_manifest_sha256": file_hash(Path(args.input) / "manifest.json"),
        "rag_sha256": file_hash(args.rag) if args.rag else None,
        "mentor_index_sha256": retriever.bank.fingerprint if args.mentor_index else None,
        "model_config": config.public_dict(),
    }
    client = ModelClient(config) if payload["candidate_groups"] else None
    if client is None:
        # Empty structural outputs need no credentials or network connection.
        from types import SimpleNamespace

        client = SimpleNamespace(config=config, close=lambda: None)
    started = time.monotonic()
    latest = {}

    def save(event):
        latest.clear()
        latest.update(event)
        write_json(output.with_suffix(output.suffix + ".progress.json"), {**audit, **event})

    try:
        result = refine_candidates(
            payload, prepared["pipeline_results"], retriever, client, on_event=save
        )
        selected = {identities[anon] for anon in result["retained_anonymous_instances"]}
        nodes = sorted({node for name in selected for node in prepared["node_mapping"][name]})
        all_names = set(identities.values())
        write_json(
            output,
            {
                **audit,
                **result,
                "elapsed_seconds": time.monotonic() - started,
                "retained_instances": sorted(selected),
                "removed_instances": sorted(all_names - selected),
                "candidate_node_ids": nodes,
                "inspection_nodes": len(nodes),
                "baseline_inspection_nodes": len(
                    {n for ns in prepared["node_mapping"].values() for n in ns}
                ),
            },
        )
        print(f"Valid decision: {len(selected)} instances / {len(nodes)} detector nodes.")
    except Exception as exc:
        # Preserve raw responses and tool traces; invalid output never becomes
        # an experiment score, and no label-informed retry is attempted.
        write_json(
            output.with_suffix(output.suffix + ".failed.json"),
            {
                **audit,
                **latest,
                "status": "invalid",
                "error_type": type(exc).__name__,
                "error": "Review failed; inspect local response trace. No result was scored.",
                "elapsed_seconds": time.monotonic() - started,
            },
        )
        raise
    finally:
        client.close()


def score(args):
    prepared = load_prepared(args.input)
    labels = {int(node): value for node, value in read_json(args.labels).items()}
    ranking = prepared["metadata"]["ranked_node_ids"]
    stages = {"nhtd_gl": prepared["metadata"]["initial_node_ids"]}
    name_nodes = prepared["node_mapping"]
    # Payload expansion can contain instances omitted by ModuleChecker. Score
    # only stages for which this prepared bundle has a complete node mapping.
    stages["modulechecker"] = sorted({n for nodes in name_nodes.values() for n in nodes})
    if args.result:
        result = read_json(args.result)
        if result.get("status") != "valid" or result.get("prepared_manifest_sha256") != file_hash(
            Path(args.input) / "manifest.json"
        ):
            raise ValueError("Result does not belong to these prepared inputs")
        stages["generator"] = result["candidate_node_ids"]
    write_json(
        args.output,
        {stage: evaluate_nodes(nodes, ranking, labels) for stage, nodes in stages.items()},
    )
    print("Offline metrics saved; labels were not supplied to any model.")


def train_graph_mentor(args):
    from .graph_mentor.training import train_mentor

    manifest = train_mentor(
        args.dataset,
        args.output,
        exclude_hosts=args.exclude_host,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        seed=args.seed,
        device=args.device,
        max_batch_nodes=args.max_batch_nodes,
    )
    print(f"GraphMentor trained on {len(manifest['training_ids'])} reference graphs.")


def build_vector_index(args):
    from .rag.learned_retriever import build_embedding_bank

    build_embedding_bank(args.dataset, args.checkpoint, args.output)
    print("Graph and module FAISS indexes saved with the checkpoint manifest.")


def parser():
    root = argparse.ArgumentParser(description="TrojanRefinement structural refinement pipeline")
    sub = root.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser(
        "prepare", help="Load detector predictions and run structural tools locally"
    )
    prepare_parser.add_argument("--report", required=True)
    prepare_parser.add_argument("--variant", required=True)
    prepare_parser.add_argument("--top-k", type=int, required=True)
    prepare_parser.add_argument(
        "--trojan-type", choices=["combinational", "fsm", "counter", "auto"], required=True
    )
    prepare_parser.add_argument(
        "--netlist", help="Override the report's netlist path; SHA-256 must still match"
    )
    prepare_parser.add_argument(
        "--library", help="Standard-cell Verilog definitions; required for FSM/Counter"
    )
    prepare_parser.add_argument("--upstream-depth", type=int, default=2)
    prepare_parser.add_argument(
        "--mentor-index", help="GraphMentor checkpoint and FAISS reference bank"
    )
    prepare_parser.add_argument(
        "--suite",
        choices=["TRIT-TC", "TRIT-TS"],
        help="Required for automatic Trojan type retrieval",
    )
    prepare_parser.add_argument(
        "--neo4j", action="store_true", help="Use Neo4j/Cypher paths for module recovery"
    )
    prepare_parser.add_argument(
        "--rag", help="Optional frozen training references for ModuleChecker evidence"
    )
    prepare_parser.add_argument("--downstream-depth", type=int, default=2)
    prepare_parser.add_argument("--output-tolerance", type=int, default=1)
    prepare_parser.add_argument("--instance-budget", type=int, default=100)
    prepare_parser.add_argument("--module-recovery-depth", type=int, default=2)
    prepare_parser.add_argument("--max-new-instances", type=int, default=100)
    prepare_parser.add_argument("--output", required=True)
    prepare_parser.set_defaults(func=prepare)
    review_parser = sub.add_parser("review", help="Review frozen results using the chosen API")
    review_parser.add_argument("--input", required=True)
    review_parser.add_argument("--config", required=True)
    review_parser.add_argument("--rag", help="Optional anonymous frozen training-reference JSON")
    review_parser.add_argument("--mentor-index", help="Optional GraphMentor/FAISS reference bank")
    review_parser.add_argument("--output", required=True)
    review_parser.set_defaults(func=review)
    rag_parser = sub.add_parser(
        "freeze-rag", help="Verify and anonymize an external training knowledge base"
    )
    rag_parser.add_argument("--knowledge-base", required=True)
    rag_parser.add_argument("--output", required=True)
    rag_parser.set_defaults(
        func=lambda args: write_json(args.output, freeze_knowledge_base(args.knowledge_base))
    )
    score_parser = sub.add_parser("score", help="Evaluate saved decisions offline")
    score_parser.add_argument("--input", required=True)
    score_parser.add_argument(
        "--labels", required=True, help="JSON map of detector node IDs to 0/1"
    )
    score_parser.add_argument("--result")
    score_parser.add_argument("--output", required=True)
    score_parser.set_defaults(func=score)
    train_parser = sub.add_parser(
        "train-mentor", help="Train GraphMentor with host-isolated module annotations"
    )
    train_parser.add_argument("--dataset", required=True)
    train_parser.add_argument("--output", required=True)
    train_parser.add_argument("--exclude-host", nargs="+", required=True)
    train_parser.add_argument("--epochs", type=int, default=100)
    train_parser.add_argument("--batch-size", type=int, default=8)
    train_parser.add_argument("--learning-rate", type=float, default=0.001)
    train_parser.add_argument("--hidden-dim", type=int, default=64)
    train_parser.add_argument("--num-layers", type=int, default=2)
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument("--device", default="cpu")
    train_parser.add_argument("--max-batch-nodes", type=int, default=4096)
    train_parser.set_defaults(func=train_graph_mentor)
    index_parser = sub.add_parser(
        "build-vector-index", help="Build graph/module FAISS indexes from trained embeddings"
    )
    index_parser.add_argument("--dataset", required=True)
    index_parser.add_argument("--checkpoint", required=True)
    index_parser.add_argument("--output", required=True)
    index_parser.set_defaults(func=build_vector_index)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        args.func(args)
    except ImportError:
        print(
            "Error: Install optional dependencies with pip install -e '.[research,api]'.",
            file=sys.stderr,
        )
        return 1
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as exc:
        # Do not print HTTP error bodies or client objects containing secrets.
        if isinstance(exc, (ValueError, KeyError, TypeError)):
            print(f"Error: {exc}", file=sys.stderr)
        else:
            print(
                f"Error: {type(exc).__name__}. Check configuration and local traces.",
                file=sys.stderr,
            )
        return 1
    return 0
