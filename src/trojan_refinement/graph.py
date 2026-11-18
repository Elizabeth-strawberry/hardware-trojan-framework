"""Directed instance connectivity, weak components, and combinational module evidence."""

from collections import defaultdict
from .ports import OUTPUT_PORTS


def build_graph_index(cells):
    drivers, consumers = (defaultdict(set), defaultdict(set))
    for name, cell in cells.items():
        for port, signal in cell["port_connections"].items():
            if port.upper() in OUTPUT_PORTS:
                drivers[signal].add(name)
            elif port.upper() not in {"CLK", "CK"}:
                consumers[signal].add(name)
    forward, reverse = (defaultdict(set), defaultdict(set))
    for signal in drivers.keys() & consumers.keys():
        for a in drivers[signal]:
            for b in consumers[signal]:
                if a != b:
                    forward[a].add(b)
                    reverse[b].add(a)
    return (forward, reverse, drivers, consumers)


def weakly_connected_components(seeds, forward, reverse):
    remaining, result = (set(seeds), [])
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        group, frontier = ({start}, [start])
        while frontier:
            current = frontier.pop()
            for name in sorted((forward[current] | reverse[current]) & remaining):
                remaining.remove(name)
                group.add(name)
                frontier.append(name)
        result.append(group)
    return result


def check_combinational_modules(group, clues, cells, drivers):
    """Screen combinational control cones independently of payload expansion."""
    partitions = []
    for clue in clues:
        payload = clue["instance_name"]
        for port, signal in clue["internal_inputs"].items():
            for root in sorted(drivers[signal] & group - {payload}):
                parents, pending = ({}, [root])
                valid = True
                while pending:
                    node = pending.pop()
                    if node in parents:
                        continue
                    if any((t in cells[node]["cell_type"].lower() for t in ("dff", "latch"))):
                        valid = False
                        break
                    upstream = {
                        d
                        for p, sig in cells[node]["port_connections"].items()
                        if p.upper() not in OUTPUT_PORTS
                        for d in drivers[sig] & group
                    }
                    if payload in upstream:
                        valid = False
                        break
                    parents[node] = upstream
                    pending.extend(upstream - parents.keys())
                if not valid:
                    continue
                remaining = set(parents)
                while remaining:
                    ready = {n for n in remaining if not parents[n] & remaining}
                    if not ready:
                        valid = False
                        break
                    remaining -= ready
                if not valid:
                    continue
                collection = {n for n in parents if not parents[n]} - {root}
                aggregation = set(parents) - collection
                if collection and aggregation:
                    partitions.append(
                        {
                            "payload_instance": payload,
                            "control_port": port,
                            "signal_collection": sorted(collection),
                            "intermediate_aggregation": sorted(aggregation),
                            "trigger_decision": [root],
                            "aggregation_decision_overlap": [root],
                            "unique_role_instance_count": len(parents),
                        }
                    )
    return {
        "is_trojan_candidate": bool(clues and partitions),
        "reason": (
            "three_roles_found_decision_overlap_allowed"
            if partitions
            else "three_roles_not_found" if clues else "no_payload_clue"
        ),
        "role_partitions": partitions,
        "basis": "combinational_control_cone_topology",
        "functional_roles_confirmed": False,
    }
