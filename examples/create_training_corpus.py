"""Create explicitly annotated synthetic reference graphs for the training example."""

from pathlib import Path

from create_demo import create_demo
from trojan_refinement.graph_mentor.data import load_corpus
from trojan_refinement.netlist import parse_netlist
from trojan_refinement.rag.graph_similarity import circuit_graph
from trojan_refinement.storage import read_json, write_json


def create_training_corpus(directory):
    directory = Path(directory)
    report_path = create_demo(directory)
    report = read_json(report_path)
    cells = parse_netlist(
        directory / "synthetic_demo.v", report["ranked_samples"][0]["netlist_sha256"]
    )
    labels = {
        "collect_a": "signal_acquisition",
        "collect_b": "signal_acquisition",
        "trigger": "trigger_generation",
        "payload": "payload",
    }
    graph = circuit_graph([{"instance_name": name, **cells[name]} for name in labels])
    entries = [
        {
            "id": f"reference_{i}",
            "host": f"training_host_{i}",
            "suite": "TRIT-TC",
            "split": "train",
            "trojan_type": "combinational",
            "graph": graph,
            "module_labels": labels,
        }
        for i in (1, 2)
    ]
    path = directory / "module_graphs.json"
    write_json(path, {"schema": "trojan_module_corpus_v1", "entries": entries})
    load_corpus(path)
    return path


if __name__ == "__main__":
    print(create_training_corpus("outputs/training_demo"))
