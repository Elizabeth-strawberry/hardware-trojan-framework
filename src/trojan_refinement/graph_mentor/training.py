"""Train GraphMentor using host-isolated, explicitly annotated reference graphs."""

from collections import Counter
import hashlib
from pathlib import Path
import random

import torch
from torch.nn import functional as F
from torch_geometric.data import Batch

from .data import FEATURE_NAMES, TYPES, graph_tensor, load_corpus
from .model import GraphMentor, module_contrastive_loss
from ..storage import write_json


def train_mentor(
    dataset,
    output,
    *,
    exclude_hosts,
    epochs=100,
    batch_size=8,
    learning_rate=0.001,
    hidden_dim=64,
    num_layers=2,
    seed=0,
    device="cpu",
    max_batch_nodes=4096
):
    if not exclude_hosts:
        raise ValueError("Declare held-out hosts before training")
    if (
        any(type(v) is not int or v < 1 for v in (epochs, batch_size, max_batch_nodes))
        or not 0 < learning_rate < 1
    ):
        raise ValueError("Invalid training budget")
    output = Path(output)
    if output.exists():
        raise ValueError("Use a new checkpoint path")
    excluded = {host.lower() for host in exclude_hosts}
    entries = [
        e for e in load_corpus(dataset) if e["split"] == "train" and e["host"] not in excluded
    ]
    if not entries:
        raise ValueError("No training graphs remain after host isolation")
    counts = Counter(
        (e["trojan_type"], label) for e in entries for label in e["module_labels"].values()
    )
    if not any(
        len({role for typ, role in counts if typ == t}) > 1
        and any(count > 1 for (typ, _), count in counts.items() if typ == t)
        for t in TYPES
    ):
        raise ValueError(
            "Module metric learning needs repeated labels and different modules within a type"
        )
    roles = {role: i for i, role in enumerate(sorted(counts))}
    samples = []
    for entry in entries:
        data, ids = graph_tensor(entry["graph"])
        if len(ids) > max_batch_nodes:
            raise ValueError("A training graph exceeds max_batch_nodes")
        data.y_module = torch.tensor(
            [roles[(entry["trojan_type"], entry["module_labels"][n])] for n in ids]
        )
        data.y_type = torch.full((len(ids),), TYPES.index(entry["trojan_type"]), dtype=torch.long)
        data.graph_type = torch.tensor([TYPES.index(entry["trojan_type"])])
        samples.append(data)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = GraphMentor(hidden_dim, num_layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    history = []
    for epoch in range(epochs):
        order = list(range(len(samples)))
        rng.shuffle(order)
        batches, current, size = [], [], 0
        for index in order:
            sample = samples[index]
            if current and (
                len(current) >= batch_size or size + sample.num_nodes > max_batch_nodes
            ):
                batches.append(current)
                current, size = [], 0
            current.append(sample)
            size += sample.num_nodes
        if current:
            batches.append(current)
        model.train()
        total = 0.0
        for rows in batches:
            batch = Batch.from_data_list(rows).to(device)
            optimizer.zero_grad()
            nodes, graphs = model(batch.x, batch.edge_index, batch.batch)
            metric = module_contrastive_loss(nodes, batch.y_module, batch.y_type)
            loss = metric + 0.25 * F.cross_entropy(model.type_head(graphs), batch.graph_type)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite GraphMentor training loss")
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
        history.append({"epoch": epoch + 1, "loss": total / len(batches)})
    manifest = {
        "schema": "graph_mentor_checkpoint_v1",
        "architecture": model.config,
        "feature_names": list(FEATURE_NAMES),
        "training_hosts": sorted({e["host"] for e in entries}),
        "excluded_hosts": sorted(excluded),
        "training_ids": [e["id"] for e in entries],
        "dataset_sha256": hashlib.sha256(Path(dataset).read_bytes()).hexdigest(),
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "max_batch_nodes": max_batch_nodes,
        "history": history,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"manifest": manifest, "state_dict": model.cpu().state_dict()}, output)
    write_json(output.with_suffix(".json"), manifest)
    return manifest


def load_mentor(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    manifest = checkpoint["manifest"]
    if manifest.get("schema") != "graph_mentor_checkpoint_v1" or manifest.get(
        "feature_names"
    ) != list(FEATURE_NAMES):
        raise ValueError("Unsupported GraphMentor checkpoint or feature schema")
    model = GraphMentor(**manifest["architecture"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model, manifest
