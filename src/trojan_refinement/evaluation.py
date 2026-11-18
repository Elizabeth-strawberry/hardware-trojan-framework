"""Offline metrics; neither these functions nor labels are exposed as LLM tools."""


def evaluate_nodes(selected, ranked_nodes, labels):
    """Recall, normalized inspection size, and same-size baseline recall.

    labels maps detector node IDs to 0/1. nt is used only here, after decisions.
    Instances with multiple detector output nodes are scored as multiple nodes.
    """
    selected, ranked_nodes = set(selected), list(ranked_nodes)
    if len(set(ranked_nodes)) != len(ranked_nodes) or not set(ranked_nodes) <= labels.keys():
        raise ValueError("Ranking must contain distinct labeled node IDs")
    if set(ranked_nodes) != set(labels) or any(
        type(x) is not int or x not in (0, 1) for x in labels.values()
    ):
        raise ValueError("Labels must cover exactly the ranked evaluation universe")
    if not selected <= labels.keys():
        raise ValueError("Unknown selected node IDs")
    nt = sum(labels.values())
    if not nt:
        raise ValueError("Recall is undefined without Trojan nodes")
    hits = sum(labels[n] for n in selected)
    baseline_hits = sum(labels[n] for n in ranked_nodes[: len(selected)])
    return {
        "trojan_nodes": nt,
        "inspection_nodes": len(selected),
        "true_positives": hits,
        "recall": hits / nt,
        "inspection_cost": len(selected) / nt,
        "same_cost_nhtd_recall": baseline_hits / nt,
        "recall_gain_pp": 100 * (hits - baseline_hits) / nt,
    }
