"""Load ranked NHTD-GL predictions and map selected nodes to driving cells.

The baseline detector is an external input, not re-trained by this package.
Only the supplied top-k predictions seed structural recovery; the full netlist
provides connectivity context and is never used as the initial candidate set.
"""

import hashlib
import json
import math
from pathlib import Path

from .netlist import parse_netlist


def read_predictions(report_path, variant, top_k, netlist_path=None):
    """Read a mapped report with ranked_samples; preserve score ties in input order."""
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    report_path = Path(report_path).resolve()
    report = json.loads(report_path.read_text())
    samples = [row for row in report["ranked_samples"] if row["netlist"] == variant]
    if not samples:
        raise ValueError("No detector predictions for the requested variant")
    ids = [row["node_id"] for row in samples]
    if any(type(n) is not int or n < 0 for n in ids) or len(set(ids)) != len(ids):
        raise ValueError("Detector node IDs must be unique nonnegative integers")
    for row in samples:
        score = row["trojan_score"]
        if type(score) not in (int, float) or not math.isfinite(score):
            raise ValueError("Detector scores must be finite numbers")
    samples.sort(key=lambda row: -row["trojan_score"])
    hashes = {row["netlist_sha256"] for row in samples}
    if len(hashes) != 1:
        raise ValueError("Predictions refer to different netlist fingerprints")
    path = Path(netlist_path) if netlist_path else Path(samples[0]["netlist_path"])
    if not path.is_absolute():
        path = report_path.parent / path
    cells = parse_netlist(path, next(iter(hashes)))
    if not cells:
        raise ValueError("No supported named-port instances were parsed")
    # Rebuild an allowlisted view. Labels, statistics, and other report fields
    # never enter model inputs or the read-only result tools.
    clean = []
    for row in samples:
        drivers = []
        for driver in row.get("drivers", []):
            name = driver["instance_name"]
            cell = cells.get(name)
            if cell is None or cell["cell_type"] != driver["cell_type"]:
                raise ValueError("Detector-to-instance mapping does not match the netlist")
            port = driver["output_port"]
            if port not in cell["port_connections"]:
                raise ValueError("Unknown mapped output port")
            if "signal_name" in row and cell["port_connections"][port] != row["signal_name"]:
                raise ValueError("Detector node signal differs from its mapped driver output")
            drivers.append(
                {"instance_name": name, "cell_type": cell["cell_type"], "output_port": port}
            )
        clean.append(
            {"node_id": row["node_id"], "trojan_score": row["trojan_score"], "drivers": drivers}
        )
    initial = clean[:top_k]
    seeds = {d["instance_name"] for row in initial for d in row["drivers"]}
    if not seeds:
        raise ValueError("None of the selected predictions maps to a driving cell")
    return {
        "cells": cells,
        "seeds": seeds,
        "samples": clean,
        "ranked_node_ids": [row["node_id"] for row in clean],
        "initial_node_ids": [row["node_id"] for row in initial],
        "netlist_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "unmapped_initial_node_ids": [row["node_id"] for row in initial if not row["drivers"]],
    }
