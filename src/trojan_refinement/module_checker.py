"""Inspect module completeness, recover cut interfaces, and retain combinational residuals in mixed cases."""

from copy import deepcopy
from .graph import build_graph_index, weakly_connected_components, check_combinational_modules
from .payload_checker import find_payload_clues
from .sequential import analyze_sequential_modules
from .module_recovery import recover_missing_modules


def _collect_module_groups(cells, seeds, payload, typ, lib):
    _, _, drivers, _ = build_graph_index(cells)
    groups = []
    for p in payload["groups"]:
        if not p["accepted"]:
            continue
        names = set(p["expanded_instances"])
        local = {n: cells[n] for n in names}
        if typ == "Comb.":
            screen = check_combinational_modules(names, p["payload_clues"], local, drivers)
            if screen["is_trojan_candidate"]:
                groups.append(
                    {
                        "kind": "combinational",
                        "candidate_instances": sorted(names),
                        "payload_clues": p["payload_clues"],
                        "module_evidence": screen,
                        "source_payload_group": p["group_id"],
                    }
                )
        else:
            r = analyze_sequential_modules(
                local, set(p["seed_instances"]) & names, lib, 32, 500, 16, typ.lower()
            )
            used = set()
            for g in r["groups"]:
                members = set(g["candidate_instances"])
                used |= members
                groups.append(
                    {
                        "kind": typ.lower(),
                        "candidate_instances": sorted(members),
                        "module_evidence": g,
                        "payload_clues": g["payload_connections"],
                        "source_payload_group": p["group_id"],
                    }
                )
            residual = names - used
            residual = {
                n for n in residual if not lib.get(cells[n]["cell_type"], {}).get("sequential")
            }
            f, b, dr, _ = build_graph_index({n: cells[n] for n in residual})
            for component in weakly_connected_components(residual, f, b):
                if not component & seeds:
                    continue
                clues = find_payload_clues(component, cells, f, b, 1)
                for c in clues:
                    c["internal_inputs"] = {
                        port: sig
                        for port, sig in cells[c["instance_name"]]["port_connections"].items()
                        if port.upper() not in ("Q", "QN", "Y", "Z", "ZN", "O", "OUT")
                        and dr[sig] & component - {c["instance_name"]}
                    }
                screen = check_combinational_modules(component, clues, cells, dr)
                if screen["is_trojan_candidate"]:
                    groups.append(
                        {
                            "kind": "combinational",
                            "candidate_instances": sorted(component),
                            "payload_clues": clues,
                            "module_evidence": screen,
                            "source_payload_group": p["group_id"],
                        }
                    )
    unique = {}
    for g in groups:
        unique.setdefault(tuple(g["candidate_instances"]), g)
    groups = list(unique.values())
    for i, g in enumerate(groups, 1):
        g["group_id"] = f"g{i:04d}"
    return groups


def check_modules(
    cells,
    seeds,
    payload,
    typ,
    lib,
    recovery_depth=2,
    max_new_instances=100,
    graph_store=None,
    graph_key=None,
    type_hypotheses=None,
):
    """Recover from cut module interfaces, then recheck the enlarged bounded scope."""
    from copy import deepcopy

    _, _, drivers, _ = build_graph_index(cells)
    combined = []
    audits = []
    for p in payload["groups"]:
        if not p["accepted"]:
            continue
        local_type = (type_hypotheses or {}).get(p["group_id"], typ)
        single = {"groups": [p]}
        before = _collect_module_groups(cells, seeds, single, local_type, lib)
        audit = recover_missing_modules(
            cells,
            set(p["expanded_instances"]),
            before,
            p["payload_clues"],
            lib,
            local_type,
            depth=recovery_depth,
            max_added=max_new_instances,
            max_registers=16,
            graph_store=graph_store,
            graph_key=graph_key,
        )
        enlarged = deepcopy(p)
        enlarged["expanded_instances"] = audit["scope_instances"]
        members = set(audit["scope_instances"])
        for clue in enlarged["payload_clues"]:
            clue["internal_inputs"] = {
                port: sig
                for port, sig in cells[clue["instance_name"]]["port_connections"].items()
                if port.upper() not in ("Q", "QN", "Y", "Z", "ZN", "O", "OUT")
                and drivers[sig] & members - {clue["instance_name"]}
            }
        after = _collect_module_groups(cells, seeds, {"groups": [enlarged]}, local_type, lib)
        for old in before:
            if not any(
                (
                    old["kind"] == g["kind"]
                    and set(old["candidate_instances"]) <= set(g["candidate_instances"])
                    for g in after
                )
            ):
                after.append(old)
        accepted = {n for g in after for n in g["candidate_instances"]}
        audit.update(
            source_payload_group=p["group_id"],
            accepted_new_instances=sorted(accepted - set(p["expanded_instances"])),
            rejected_new_scope_instances=sorted(set(audit["new_scope_instances"]) - accepted),
        )
        audits.append(audit)
        combined.extend(after)
    unique = {}
    for g in combined:
        unique.setdefault((g["kind"], tuple(g["candidate_instances"])), g)
    result = list(unique.values())
    for i, g in enumerate(result, 1):
        g["group_id"] = f"g{i:04d}"
    payload["module_recovery"] = audits
    return result
