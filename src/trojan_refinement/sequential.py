"""Library-backed FSM data/enable SCCs and Counter state-chain module analysis."""

import re
from collections import defaultdict, deque


def load_cell_library(path, types):
    """Exact module retrieval, with source evidence; do not infer behavior from instance names."""
    text = path.read_text()
    found = {}
    for match in re.finditer("\\bmodule\\s+(\\w+)\\s*\\(.*?\\bendmodule", text, re.S):
        if match[1] not in types:
            continue
        body = match[0]
        directions = {}
        for direction, names in re.findall("\\b(input|output)\\s+([^;]+);", body):
            for name in names.split(","):
                directions[name.strip()] = direction
        sequential = bool(re.search("\\bp_ff\\w*\\s+", body))
        semantics = {"clock_edge": "unknown", "enable": None, "reset": "unknown", "verified": False}
        normalized = re.sub("\\s+", "", body)
        if "p_ff_i0(Q,DIN,CLK,notifier);" in normalized and "not_i1(QN,Q);" in normalized:
            semantics = {
                "clock_edge": "rising",
                "enable": None,
                "reset": None,
                "verified": True,
                "equation": "Q_next = DIN at rising CLK; QN = !Q",
            }
        elif all(
            (
                x in normalized
                for x in (
                    "not_i0(_n2,EB);",
                    "p_mux21_i1(_n1,DIN,buf_Q,_n2);",
                    "p_ff_i2(buf_Q,_n1,CLK,notifier);",
                    "not_i3(QN,buf_Q);",
                    "buf_iQ(Q,buf_Q);",
                )
            )
        ):
            semantics = {
                "clock_edge": "rising",
                "enable": {"port": "EB", "active_level": 0},
                "reset": None,
                "verified": True,
                "equation": "Q_next = (EB == 0 ? DIN : Q) at rising CLK; QN = !Q",
            }
        primitives = {}
        for primitive in set(re.findall("\\b(p_\\w+)\\s+\\w+\\s*\\(", body)):
            definition = re.search(
                "\\bprimitive\\s+" + re.escape(primitive) + "\\b.*?\\bendprimitive", text, re.S
            )
            if definition:
                primitives[primitive] = definition[0]
        found[match[1]] = {
            "directions": directions,
            "sequential": sequential,
            "primitive_definitions": primitives,
            "unsupported_sequential": bool(re.search("\\bp_(?:lat|latch)\\w*\\s+", body)),
            "source": str(path.resolve()),
            "source_line": text.count("\n", 0, match.start()) + 1,
            "definition": body,
            "control_semantics": semantics,
            "output_polarity": {
                p: "complemented state" if p == "QN" else "state"
                for p in directions
                if p in ("Q", "QN")
            },
        }
    return found


def analyze_sequential_modules(
    cells, seeds, library, max_depth=12, max_instances=200, max_registers=16, structure_type="fsm"
):
    """Recover bounded longest simple DFF chains and port-defined logic cones."""
    if structure_type not in ("fsm", "counter"):
        raise ValueError("structure_type must be fsm or counter")
    for value in (max_depth, max_instances, max_registers):
        if type(value) is not int or value < 1:
            raise ValueError("budgets must be positive integers")
    if not seeds or set(seeds) - cells.keys():
        raise ValueError("seeds must be nonempty and present in netlist")
    drivers, consumers = (defaultdict(set), defaultdict(set))
    inputs, outputs, enables = ({}, {}, {})
    registers, unsupported = (set(), set())
    for n, cell in cells.items():
        spec = library.get(cell["cell_type"])
        ports = cell["port_connections"]
        if (
            not spec
            or spec.get("unsupported_sequential")
            or any((p not in spec["directions"] for p in ports))
        ):
            unsupported.add(n)
            continue
        outputs[n] = {p: s for p, s in ports.items() if spec["directions"][p] == "output"}
        ins = {p: s for p, s in ports.items() if spec["directions"][p] == "input"}
        if spec["sequential"]:
            registers.add(n)
            inputs[n] = {p: s for p, s in ins.items() if p in ("D", "DIN")}
            enable = spec.get("control_semantics", {}).get("enable")
            ep = enable.get("port") if enable else None
            enables[n] = {p: s for p, s in ins.items() if p == ep or p in ("EB", "EN", "E", "CE")}
        else:
            inputs[n] = ins
        for signal in outputs[n].values():
            drivers[signal].add(n)
        for signal in ins.values():
            consumers[signal].add(n)
    for n in unsupported:
        for signal in cells[n]["port_connections"].values():
            consumers[signal].add(n)

    def cone(signals, stop_signals=()):
        nodes, regs, boundaries, issues = (set(), set(), set(), set())
        queue = deque(((s, 0) for s in signals))
        visited = set()
        while queue:
            signal, depth = queue.popleft()
            if signal in visited:
                continue
            visited.add(signal)
            if signal in stop_signals:
                boundaries.add(signal)
                continue
            ds = drivers[signal]
            if len(ds) != 1:
                boundaries.add(signal)
                if ds:
                    issues.add("multiple_drivers")
                continue
            n = next(iter(ds))
            if n in registers:
                if signal in [v for p, v in outputs[n].items() if p in ("Q", "QN")]:
                    regs.add(n)
                continue
            if n in unsupported:
                issues.add("unsupported_cell")
                continue
            if depth >= max_depth or len(nodes) >= max_instances:
                issues.add("cone_budget_reached")
                boundaries.add(signal)
                continue
            if n in nodes:
                continue
            nodes.add(n)
            queue.extend(((s, depth + 1) for s in inputs[n].values()))
        return (nodes, regs, boundaries, issues)

    cones = {r: cone(inputs[r].values()) for r in registers}
    forward = defaultdict(set)
    for target, (_, sources, _, _) in cones.items():
        for source in sources:
            forward[source].add(target)

    def is_xor(n):
        return (
            n not in unsupported
            and n not in registers
            and bool(
                re.search(
                    "\\bx(?:or|nor)\\s+", library[cells[n]["cell_type"]].get("definition", "")
                )
            )
        )

    internal_state_xors = defaultdict(set)
    if structure_type == "counter":
        for target, (gates, sources, _, _) in cones.items():
            if target not in sources:
                continue
            for x in gates:
                if not is_xor(x):
                    continue
                xor_sources = cone(inputs[x].values())[1]
                if target in xor_sources:
                    internal_state_xors[x].add(target)
    terminal_outputs = defaultdict(list)
    for r in sorted(registers):
        for qp, signal in outputs[r].items():
            if qp not in ("Q", "QN"):
                continue
            for x in sorted(consumers[signal]):
                if not is_xor(x) or not outputs.get(x):
                    continue
                if x in cones[r][0] or x in internal_state_xors:
                    continue
                for port, s in inputs[x].items():
                    if s == signal:
                        terminal_outputs[r].append(
                            {
                                "instance_name": x,
                                "output_register": r,
                                "register_output_port": qp,
                                "xor_input_port": port,
                                "signal": signal,
                                "boundary_policy": "stop_at_xor_allow_downstream_consumers",
                                "output_signals": sorted(outputs[x].values()),
                                "downstream_connections": [
                                    {"signal": os, "target": target, "target_port": tp}
                                    for os in sorted(outputs[x].values())
                                    for target in sorted(consumers[os])
                                    for tp, ts in cells[target]["port_connections"].items()
                                    if ts == os
                                    and library.get(cells[target]["cell_type"], {})
                                    .get("directions", {})
                                    .get(tp)
                                    == "input"
                                ],
                                "other_inputs": [
                                    {"port": ip, "signal": iv, "drivers": sorted(drivers[iv])}
                                    for ip, iv in inputs[x].items()
                                    if ip != port
                                ],
                            }
                        )
    anchors = set(seeds) & registers
    for n in set(seeds) - registers if not anchors else set():
        anchors.update(cone(inputs.get(n, {}).values())[1])
        todo = deque([(n, 0)])
        seen = set()
        while todo:
            node, depth = todo.popleft()
            if node in seen:
                continue
            seen.add(node)
            if node in registers:
                anchors.add(node)
                continue
            if depth >= max_depth:
                continue
            for signal in outputs.get(node, {}).values():
                for target in consumers[signal]:
                    if target in registers:
                        if signal in set(inputs[target].values()) | set(enables[target].values()):
                            anchors.add(target)
                    elif target not in unsupported:
                        todo.append((target, depth + 1))

    def path_gates(source_regs, target):
        allowed = cones[target][0]
        found = set()
        todo = deque()
        for r in source_regs:
            todo.extend((s for p, s in outputs[r].items() if p in ("Q", "QN")))
        seen = set()
        while todo:
            signal = todo.popleft()
            if signal in seen:
                continue
            seen.add(signal)
            for n in consumers[signal] & allowed:
                found.add(n)
                todo.extend(outputs[n].values())
        return found

    enable_cones = (
        {r: cone(enables[r].values()) for r in registers} if structure_type == "fsm" else {}
    )
    reverse = (
        {r: set(cones[r][1]) | set(enable_cones[r][1]) for r in registers}
        if structure_type == "fsm"
        else {r: set(cones[r][1]) for r in registers}
    )
    scc_forward = defaultdict(set)
    for target, sources in reverse.items():
        for source in sources:
            scc_forward[source].add(target)

    def reachable(root, adjacency):
        seen = {root}
        todo = [root]
        while todo:
            for n in adjacency.get(todo.pop(), set()) - seen:
                seen.add(n)
                todo.append(n)
        return seen

    results = []
    used_chains = set()
    rejected_components = []
    for anchor in sorted(anchors):
        if structure_type == "fsm":
            component = reachable(anchor, scc_forward) & reachable(anchor, reverse)
            if len(component) == 1 and anchor not in scc_forward[anchor]:
                continue
            key = tuple(sorted(component))
            if key in used_chains:
                continue
            used_chains.add(key)
            if len(component) > max_registers:
                rejected_components.append(
                    {
                        "state_registers": list(key),
                        "reason": "register_budget_reached",
                        "state_register_count": len(component),
                    }
                )
                continue
            best = list(key)
            chain_issues = set()
        else:
            if anchor in terminal_outputs:
                continue
            best = [anchor]
            stack = [[anchor]]
            searches = 0
            chain_issues = set()
            while stack:
                chain = stack.pop()
                searches += 1
                if len(chain) > len(best) or (len(chain) == len(best) and chain < best):
                    best = chain
                next_regs = forward[chain[-1]] - set(chain) - set(terminal_outputs)
                if len(chain) >= max_registers:
                    if next_regs:
                        chain_issues.add("register_budget_reached")
                    continue
                if searches >= 100000:
                    chain_issues.add("chain_search_budget_reached")
                    break
                for n in sorted(next_regs, reverse=True):
                    stack.append(chain + [n])
            key = tuple(best)
            if key in used_chains:
                continue
            used_chains.add(key)
        group = set(best)
        state_logic = set().union(*(path_gates(group, r) for r in group))
        chain_outputs = {s for r in group for p, s in outputs[r].items() if p in ("Q", "QN")}
        decoder = set()
        output_regs = set()
        payload = []
        for r in sorted(set(terminal_outputs) - group):
            if cones[r][1] & group:
                decoder.update(path_gates(group, r))
                output_regs.add(r)
                payload.extend(terminal_outputs[r])
        if structure_type == "fsm":
            for r in sorted(group):
                for clue in terminal_outputs[r]:
                    if clue["instance_name"] not in state_logic:
                        payload.append(dict(clue, output_role="state_and_output_register"))
        enable_port_signals = {s for r in group | output_regs for s in enables[r].values()}
        condition_input_signals = {
            signal
            for n in state_logic
            for signal in inputs[n].values()
            if not drivers[signal] & (state_logic | group)
        }
        control_signals = enable_port_signals | condition_input_signals
        enable, en_regs, en_boundary, en_issues = cone(control_signals, stop_signals=chain_outputs)
        auxiliary = output_regs | {x["instance_name"] for x in payload}
        core = group | state_logic
        all_nodes = core | enable | decoder | auxiliary
        issues = chain_issues | en_issues
        for r in group | output_regs:
            issues.update(cones[r][3])
        if len(all_nodes) > max_instances:
            issues.add("instance_budget_reached")
        priority = (
            best + sorted(state_logic) + sorted(all_nodes - core)
            if structure_type == "fsm"
            else best + sorted(all_nodes - group)
        )
        selected = priority[:max_instances]
        selected_set = set(selected)

        def module(nodes):
            return {
                "instances": sorted(nodes & selected_set),
                "total_discovered": len(nodes),
                "complete": not bool(issues),
                "functional_role_confirmed": False,
            }

        edges = []
        for target in selected:
            for port, signal in cells[target]["port_connections"].items():
                if library[cells[target]["cell_type"]]["directions"][port] != "input":
                    continue
                for source in sorted(drivers[signal] & selected_set):
                    edges.append(
                        {
                            "source": source,
                            "target": target,
                            "target_port": port,
                            "signal": signal,
                            "edge_kind": (
                                "clock"
                                if port in ("CLK", "CK")
                                else "control" if port in enables.get(target, {}) else "data"
                            ),
                        }
                    )
        results.append(
            {
                "group_id": f"seq_{len(results) + 1}",
                "type": "unknown",
                "type_hypothesis": structure_type,
                "type_confirmed": False,
                "structure_type": structure_type,
                "type_selection": "caller_requested",
                "type_evidence": {"behavior_validation": "not_performed"},
                "state_dependency_edges": [
                    {"source": source, "target": target, "edge_kind": kind}
                    for target in sorted(group)
                    for kind, cs in [("data", cones), ("enable", enable_cones)]
                    if target in cs
                    for source in sorted(cs[target][1] & group)
                ],
                "starting_dff": anchor,
                "dff_chain": best if structure_type == "counter" else [],
                "state_component": best if structure_type == "fsm" else [],
                "grouping_method": (
                    "data_and_enable_dependency_scc"
                    if structure_type == "fsm"
                    else "longest_simple_chain"
                ),
                "chain_search_complete": (
                    not bool(chain_issues) if structure_type == "counter" else None
                ),
                "state_registers": best[:max_instances],
                "state_register_count": len(group),
                "modules": dict(
                    {"enable_logic": module(enable)},
                    state_core=dict(
                        module(core),
                        role=structure_type,
                        includes=[
                            "state_component" if structure_type == "fsm" else "dff_chain",
                            "state_update_logic",
                        ],
                    ),
                    state_decoder=module(decoder),
                ),
                "shared_instances": sorted(
                    (enable & core | enable & decoder | core & decoder) & selected_set
                ),
                "auxiliary_output_instances": sorted(auxiliary & selected_set),
                "payload_connections": payload,
                "candidate_instances": selected,
                "total_discovered": len(all_nodes),
                "boundaries": {
                    "enable_signals": sorted(control_signals),
                    "enable_port_signals": sorted(enable_port_signals),
                    "state_update_condition_signals": sorted(condition_input_signals),
                    "external_inputs": sorted(en_boundary),
                    "external_registers": sorted(en_regs - group),
                    "enable_boundary_policy": "state_and_output_dff_enable_cone_stop_at_chain_q",
                    "enable_registers": sorted(group | output_regs),
                    "chain_output_stop_signals": sorted(en_boundary & chain_outputs),
                },
                "register_semantics": {
                    r: library[cells[r]["cell_type"]].get("control_semantics")
                    for r in sorted((group | output_regs) & selected_set)
                },
                "register_ports": {
                    r: cells[r]["port_connections"]
                    for r in sorted((group | output_regs) & selected_set)
                },
                "connections": edges,
                "completeness": "partial" if issues else "bounded_topology_recovered",
                "stop_reasons": sorted(issues) or ["bounded_register_interfaces_reached"],
                "warnings": [
                    "FSM roles are structural hypotheses; no behavior or maliciousness validation."
                ],
            }
        )
    evidence = {n for g in results for n in g["candidate_instances"]}
    return {
        "status": "success",
        "mode": "sequential_structure_review",
        "flow_version": "counter_internal_xor_v10",
        "structure_type": structure_type,
        "internal_state_xor_connections": [
            {"instance_name": x, "feedback_destination_dffs": sorted(rs)}
            for x, rs in sorted(internal_state_xors.items())
        ],
        "groups": results,
        "rejected_state_components": rejected_components,
        "seed_instances": sorted(seeds),
        "unassigned_seeds": sorted(set(seeds) - evidence),
        "unsupported_instances": sorted(unsupported),
        "instances": {n: cells[n] for n in sorted(evidence)},
        "warnings": [
            "FSM uses data and enable dependency SCCs; oversized SCCs are reported and not split. Counter uses bounded longest simple chains.",
            "Output XOR/XNOR may drive host consumers; downstream interfaces are recorded without expanding through the XOR.",
            "No test labels or instance-name heuristics are used.",
        ],
    }
