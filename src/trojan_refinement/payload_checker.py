"""Construct bounded candidate components and locate output-proximal XOR/XNOR payload clues."""

from collections import deque
from .graph import build_graph_index, weakly_connected_components
from .ports import OUTPUT_PORTS


def find_payload_clues(group, cells, forward, reverse, tolerance=1):
    """At least one XOR/XNOR at shortest directed distance <= tau to a group sink."""
    boundary = {n for n in group if not forward[n] & group}
    distance = {n: 0 for n in boundary}
    queue = deque(sorted(boundary))
    while queue:
        n = queue.popleft()
        if distance[n] >= tolerance:
            continue
        for parent in sorted(reverse[n] & group):
            if parent not in distance:
                distance[parent] = distance[n] + 1
                queue.append(parent)
    return [
        {
            "instance_name": n,
            "cell_type": cells[n]["cell_type"],
            "distance_to_boundary": distance[n],
            "position": "output_proximity",
        }
        for n in sorted(distance)
        if cells[n]["cell_type"].lower().startswith(("xor", "xnr", "xnor"))
    ]


def check_payloads(
    cells,
    seeds,
    upstream_depth=2,
    downstream_depth=2,
    tolerance=1,
    instance_budget=100,
    stop_at_registers=True,
):
    if min(upstream_depth, downstream_depth, tolerance) < 0 or instance_budget < 1:
        raise ValueError("Invalid PayloadChecker budget")
    seeds = set(seeds)
    if not seeds or not seeds <= cells.keys():
        raise ValueError("Invalid seeds")
    forward, reverse, drivers, consumers = build_graph_index(cells)
    results = []
    for seed_group in weakly_connected_components(seeds, forward, reverse):
        expanded = set(seed_group)
        up = set(seed_group)
        down = set(seed_group)
        steps = []
        clues = find_payload_clues(expanded, cells, forward, reverse, tolerance)
        found = bool(clues)
        budget_truncated = False
        for hop in range(1, max(upstream_depth, downstream_depth) + 1):
            reached = {}
            if hop <= upstream_depth:
                reached["upstream"] = {m for n in up for m in reverse[n]} - up
            if hop <= downstream_depth and (not found):
                reached["downstream"] = {m for n in down for m in forward[n]} - down
            if not reached:
                break
            if stop_at_registers:
                reached = {
                    d: {
                        n
                        for n in ns
                        if not any((t in cells[n]["cell_type"].lower() for t in ("dff", "latch")))
                    }
                    for d, ns in reached.items()
                }
            eligible = set().union(*reached.values()) - expanded
            room = max(0, instance_budget - len(expanded))
            added = set(sorted(eligible)[:room])
            expanded |= added
            up |= reached.get("upstream", set()) & expanded
            down |= reached.get("downstream", set()) & expanded
            steps.append(
                {"hop": hop, "active_directions": list(reached), "added_instances": sorted(added)}
            )
            clues = find_payload_clues(expanded, cells, forward, reverse, tolerance)
            found |= bool(clues)
            if len(eligible) > room:
                budget_truncated = True
                break
        clues = find_payload_clues(expanded, cells, forward, reverse, tolerance)
        for clue in clues:
            ports = cells[clue["instance_name"]]["port_connections"]
            clue["internal_inputs"] = {
                p: s
                for p, s in ports.items()
                if p.upper() not in OUTPUT_PORTS and drivers[s] & expanded - {clue["instance_name"]}
            }
        results.append(
            {
                "group_id": f"payload_{len(results) + 1}",
                "seed_instances": sorted(seed_group),
                "expanded_instances": sorted(expanded),
                "payload_clues": clues,
                "accepted": bool(clues),
                "expansion_steps": steps,
                "budget_truncated": budget_truncated,
            }
        )
    return {
        "groups": results,
        "candidate_instances": sorted(
            {n for g in results if g["accepted"] for n in g["expanded_instances"]}
        ),
        "settings": dict(
            upstream_depth=upstream_depth,
            downstream_depth=downstream_depth,
            tolerance=tolerance,
            instance_budget=instance_budget,
            stop_at_registers=stop_at_registers,
        ),
    }
