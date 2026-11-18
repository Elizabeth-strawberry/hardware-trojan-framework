"""Create a synthetic circuit fixture for the offline pipeline example.

Run from the repository root: python examples/create_demo.py
The files are placed in outputs/demo/ and are ignored by Git.
"""

import hashlib
import json
from pathlib import Path


def create_demo(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    netlist = """module synthetic_demo(a, b, c, d, host_a, host_b, en, result);
  input a, b, c, d, host_a, host_b, en;
  output result;
  and2s1 collect_a (.DIN1(a), .DIN2(b), .Q(collected_a));
  nor2s1 collect_b (.DIN1(c), .DIN2(d), .Q(collected_b));
  and2s1 trigger (.DIN1(collected_a), .DIN2(collected_b), .Q(condition));
  or2s1 host (.DIN1(host_a), .DIN2(host_b), .Q(host_data));
  xor2s1 payload (.DIN1(condition), .DIN2(host_data), .Q(modified));
  and2s1 consumer (.DIN1(modified), .DIN2(en), .Q(result));
endmodule
"""
    (directory / "synthetic_demo.v").write_text(netlist)
    fingerprint = hashlib.sha256(netlist.encode()).hexdigest()
    cells = [
        ("trigger", "and2s1", "condition", 0.95),
        ("payload", "xor2s1", "modified", 0.90),
        ("host", "or2s1", "host_data", 0.85),
        ("collect_a", "and2s1", "collected_a", 0.20),
        ("collect_b", "nor2s1", "collected_b", 0.15),
        ("consumer", "and2s1", "result", 0.10),
    ]
    rows = [
        {
            "netlist": "synthetic_demo",
            "node_id": i,
            "trojan_score": score,
            "netlist_path": "synthetic_demo.v",
            "netlist_sha256": fingerprint,
            "signal_name": signal,
            "drivers": [{"instance_name": name, "cell_type": kind, "output_port": "Q"}],
        }
        for i, (name, kind, signal, score) in enumerate(cells)
    ]
    report = directory / "predictions.json"
    report.write_text(json.dumps({"ranked_samples": rows}, indent=2) + "\n")
    (directory / "labels.json").write_text(
        json.dumps(
            {i: int(row[0] not in ("host", "consumer")) for i, row in enumerate(cells)}, indent=2
        )
        + "\n"
    )
    return report


if __name__ == "__main__":
    print(create_demo(Path("outputs/demo")))
