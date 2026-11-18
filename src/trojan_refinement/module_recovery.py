"""Recover missing private module fanin and bounded decoder/output-register interfaces."""

from collections import defaultdict, deque
from .graph import build_graph_index


def recover_missing_modules(
    cells,
    scope,
    groups,
    clues,
    library,
    structure_type,
    depth=2,
    max_added=100,
    max_registers=16,
    graph_store=None,
    graph_key=None,
):
    if type(depth) is not int or depth < 0 or type(max_added) is not int or (max_added < 0):
        raise ValueError("Invalid module recovery budgets")
    scope = set(scope)
    forward, reverse, drivers, consumers = build_graph_index(cells)

    def ports(n, direction):
        directions = library.get(cells[n]["cell_type"], {}).get("directions", {})
        return {
            p: s
            for p, s in cells[n]["port_connections"].items()
            if directions.get(p) == direction and p.upper() not in ("CLK", "CK", "CLOCK")
        }

    def reg(n):
        return library.get(cells[n]["cell_type"], {}).get("sequential", False)

    def unsupported(n):
        return (
            n not in cells
            or not library.get(cells[n]["cell_type"])
            or library[cells[n]["cell_type"]].get("unsupported_sequential", False)
        )

    def xor(n):
        return cells[n]["cell_type"].lower().startswith(("xor", "xnr", "xnor"))

    def qsignals(n):
        return {s for p, s in ports(n, "output").items() if p in ("Q", "QN")}

    def register_inputs(n):
        return {
            p: s for p, s in ports(n, "input").items() if p in ("D", "DIN", "EB", "E", "EN", "CE")
        }

    core = set()
    outputs = set()
    module_nodes = set()
    state_logic = set()
    payload_nodes = {c["instance_name"] for c in clues}
    for g in groups:
        ev = g["module_evidence"]
        module_nodes.update(g["candidate_instances"])
        if g["kind"] != "combinational":
            core.update(ev.get("state_registers", []))
            state_logic.update(ev.get("modules", {}).get("state_core", {}).get("instances", []))
            outputs.update(
                (
                    c["output_register"]
                    for c in ev.get("payload_connections", [])
                    if "output_register" in c
                )
            )
            payload_nodes.update((c["instance_name"] for c in ev.get("payload_connections", [])))
    if structure_type != "Comb." and (not core):
        core = {n for n in scope if reg(n) and (not any((c in payload_nodes for c in forward[n])))}
        if len(core) > max_registers:
            core = set()
    module_nodes |= core | outputs
    targets = []
    if structure_type == "Comb." or not groups:
        targets.extend(
            (
                (n, p, s, "payload_control")
                for n in sorted(payload_nodes)
                for p, s in ports(n, "input").items()
            )
        )
    for n in sorted(module_nodes - payload_nodes):
        ins = register_inputs(n) if reg(n) else ports(n, "input")
        targets.extend(
            (
                (n, p, s, "state_enable_or_data" if reg(n) else "module_fanin")
                for p, s in ins.items()
            )
        )
    interfaces = [
        {"target": n, "port": p, "signal": s, "role": role}
        for n, p, s, role in targets
        if drivers[s] - scope
    ]
    added = set()
    proposals = []
    boundaries = []
    truncated = False
    graph_queries = []
    path_members = None
    if graph_store is not None:
        if not graph_key:
            raise ValueError("Neo4j recovery requires an imported graph key")
        if depth > 30:
            raise ValueError("Neo4j module recovery depth must be at most 30")
        path_members = set(scope)
        if depth and max_added:
            for interface in interfaces:
                result = graph_store.bounded_paths(
                    graph_key,
                    targets=[interface["target"]],
                    target_port=interface["port"],
                    max_hops=depth,
                )
                graph_queries.append({"interface": interface, **result})
            if core:
                result = graph_store.bounded_paths(
                    graph_key, sources=sorted(core), max_hops=depth + 2
                )
                graph_queries.append({"interface": {"role": "state_decoder_output"}, **result})
            for query in graph_queries:
                truncated |= query["truncated"]
                path_members.update(n for path in query["paths"] for n in path["nodes"])

    def add(nodes, interface, reason):
        nonlocal truncated
        new = set(nodes) - scope - added
        if path_members is not None and not new <= path_members:
            boundaries.append({"reason": "outside_retrieved_paths", "interface": interface})
            return False
        if len(added | new) > max_added:
            truncated = True
            boundaries.append({"reason": "new_instance_budget", "interface": interface})
            return False
        if any((unsupported(n) for n in new)):
            return False
        for n in sorted(new):
            proposals.append({"instance_name": n, "interface": interface, "reason": reason})
        added.update(new)
        return True

    def return_paths(target, starts):
        """Find known-core Q/QN -> <=depth combinational gates -> target data/enable."""
        paths = []
        todo = deque(((s, (), 0) for r in starts for s in qsignals(r)))
        seen = set()
        while todo:
            signal, path, d = todo.popleft()
            if (signal, d) in seen:
                continue
            seen.add((signal, d))
            for n in sorted(consumers[signal]):
                if n == target and signal in register_inputs(n).values():
                    paths.append(set(path))
                    continue
                if reg(n) or unsupported(n) or n in payload_nodes or (d >= depth):
                    continue
                for s in ports(n, "output").values():
                    todo.append((s, path + (n,), d + 1))
        return paths

    if depth and max_added:
        for interface in interfaces:
            queue = deque([(interface["signal"], 0)])
            seen = set()
            while queue:
                signal, d = queue.popleft()
                if signal in seen:
                    continue
                seen.add(signal)
                sources = drivers[signal]
                if len(sources) != 1:
                    boundaries.append(
                        {
                            "signal": signal,
                            "reason": "primary_input_or_multiple_drivers",
                            "interface": interface,
                        }
                    )
                    continue
                n = next(iter(sources))
                if n in scope:
                    continue
                if d >= depth:
                    boundaries.append(
                        {"signal": signal, "reason": "recovery_depth", "interface": interface}
                    )
                    continue
                if unsupported(n):
                    boundaries.append(
                        {"instance_name": n, "reason": "unsupported_cell", "interface": interface}
                    )
                    continue
                if reg(n):
                    paths = return_paths(n, core) if structure_type == "FSM" else []
                    if paths and len(core | {r for r in added if reg(r)} | {n}) <= max_registers:
                        path = min(paths, key=lambda ns: (len(ns), sorted(ns)))
                        add(path | {n}, interface, "bounded_feedback_state_register")
                    else:
                        boundaries.append(
                            {
                                "instance_name": n,
                                "reason": "external_register",
                                "interface": interface,
                            }
                        )
                    continue
                external = {m for s in ports(n, "output").values() for m in consumers[s]} - (
                    scope | added
                )
                if external:
                    boundaries.append(
                        {"instance_name": n, "reason": "shared_producer", "interface": interface}
                    )
                    continue
                if add({n}, interface, "missing_module_private_fanin"):
                    queue.extend(((s, d + 1) for s in ports(n, "input").values()))
        if structure_type != "Comb.":
            for start in sorted(core):
                queue = deque(((s, (), 0) for s in qsignals(start)))
                seen = set()
                while queue:
                    signal, path, d = queue.popleft()
                    if (signal, d) in seen:
                        continue
                    seen.add((signal, d))
                    for n in sorted(consumers[signal]):
                        if unsupported(n) or n in payload_nodes:
                            continue
                        if reg(n):
                            if n in core or signal not in {
                                s for p, s in register_inputs(n).items() if p in ("D", "DIN")
                            }:
                                continue
                            endpoints = []
                            for q in qsignals(n):
                                for x in sorted(consumers[q]):
                                    if unsupported(x) or not xor(x) or x in state_logic:
                                        continue
                                    destinations = {
                                        m for s in ports(x, "output").values() for m in consumers[s]
                                    }
                                    if destinations & (core | {n}):
                                        continue
                                    if return_paths(n, {n}) and x in set().union(
                                        *return_paths(n, {n})
                                    ):
                                        continue
                                    endpoints.append(x)
                            if endpoints:
                                interface = {
                                    "source_register": start,
                                    "target_register": n,
                                    "role": "missing_decoder_output",
                                }
                                add(
                                    set(path) | {n} | set(endpoints),
                                    interface,
                                    "bounded_decoder_to_output_register_xor",
                                )
                            continue
                        if d >= depth:
                            continue
                        for s in ports(n, "output").values():
                            queue.append((s, path + (n,), d + 1))
    return {
        "scope_instances": sorted(scope | added),
        "new_scope_instances": sorted(added),
        "interfaces": interfaces,
        "proposals": proposals,
        "boundaries": boundaries,
        "budget_truncated": truncated,
        "graph_queries": graph_queries,
        "settings": {
            "external_combinational_depth": depth,
            "max_new_instances": max_added,
            "max_state_registers": max_registers,
            "decoder_policy": "at most depth combinational gates plus output DFF and directly connected XOR/XNOR",
            "upstream_policy": "private module fanin; stop at shared producers and external registers; FSM feedback register needs bounded return path",
        },
    }
