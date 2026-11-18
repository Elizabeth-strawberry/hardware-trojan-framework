"""Anonymous read-only snapshots of detector, PayloadChecker, and ModuleChecker outputs."""

from copy import deepcopy
import json
import math

TOOL_NAME = "read_pipeline_results"
STAGES = ("all", "nhtd_gl", "payloadchecker", "modulechecker")
TOOL = {
    "type": "function",
    "name": TOOL_NAME,
    "strict": True,
    "description": "Read previously recorded NHTD-GL initial predictions, PayloadChecker expansion, and ModuleChecker recovery results for a candidate group. Does not run any tool or change nodes. Returns anonymous detection ranks/scores, stage membership, expansion steps and module-recovery evidence. Use stage=all for provenance comparison; instance_ids=[] returns the full group. Context-only IDs cannot be added to the final inspection set.",
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "required": ["group_id", "stage", "instance_ids"],
        "properties": {
            "group_id": {"type": "string"},
            "stage": {"type": "string", "enum": list(STAGES)},
            "instance_ids": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 128,
                "description": "Existing in-group IDs to focus on; [] means all",
            },
        },
    },
}
INSTRUCTIONS = "You can call read_pipeline_results to inspect the existing outputs of NHTD-GL, PayloadChecker and ModuleChecker. This is a read-only result lookup, not a request to rerun those tools. Consult stage=all for groups whose XOR/XNOR input roles or expanded module boundaries are uncertain. Compare which instances were initial detections with those introduced by payload expansion or module recovery. Detector scores are ranking evidence, not calibrated probabilities or labels. A structural tool's role assignment is a hypothesis to review, not an authoritative Trojan/benign verdict.\n\nUse provenance together with pin-level structure to distinguish an original-data branch from trigger, state, enable and payload logic. In particular, a recovered data transformation or arithmetic/parity subnetwork feeding one payload input may be ordinary host context even if the structural tools assigned it aggregation/trigger roles. Keep its boundary connection when excluding its upstream instances from inspection. Initial nonselection, low scores, expansion origin, or a tree/XOR pattern alone never justify deletion. Preserve supported trigger/state logic even when it was recovered rather than initially detected. Context-only IDs in stored results are not additional decision candidates. No test labels or target inspection size are available. Record concrete structural removal evidence; never force a smaller result."


def _ids(values, mapping):
    return [mapping[value] for value in values]


def snapshot(
    payload,
    module,
    identities,
    payload_result,
    recovery,
    initial_nodes,
    ranked_nodes,
    detector_samples,
):
    """Build an anonymous snapshot from recorded outputs, without ground truth.

    initial_nodes/ranked_nodes preserve the prepared inspection selection and
    ranking order, including score ties and any preprocessing exclusions.
    No labels, nt, metrics, benchmark logs, or netlist files are accepted here.
    """
    groups = {g["group_id"]: g for g in payload["candidate_groups"]}
    raw_groups = {g["group_id"]: g for g in module["groups"]}
    if set(groups) != set(raw_groups):
        raise ValueError("Stage-result group mismatch")
    samples = {s["node_id"]: s for s in detector_samples}
    if len(samples) != len(detector_samples) or len(set(ranked_nodes)) != len(ranked_nodes):
        raise ValueError("Duplicate detector node IDs")
    if not set(ranked_nodes) <= set(samples) or ranked_nodes[: len(initial_nodes)] != initial_nodes:
        raise ValueError("Initial detector result does not match frozen ranking")
    selected = set(initial_nodes)
    detected_names = {
        d["instance_name"] for node in initial_nodes for d in samples[node].get("drivers", [])
    }
    seed_names = {n for g in payload_result["groups"] for n in g["seed_instances"]}
    if not seed_names <= detected_names:
        raise ValueError("Payload seeds disagree with detector predictions")
    mapping = {name: anon for anon, name in identities.items()}
    names = detected_names | set(mapping)
    for g in payload_result["groups"]:
        names.update(g["seed_instances"])
        names.update(g["expanded_instances"])
    for audit in recovery:
        for field in (
            "scope_instances",
            "new_scope_instances",
            "accepted_new_instances",
            "rejected_new_scope_instances",
        ):
            names.update(audit.get(field, []))
        for field in ("interfaces", "proposals", "boundaries"):
            for row in audit.get(field, []):
                if "instance_name" in row:
                    names.add(row["instance_name"])
                endpoint = row.get("interface", row)
                for field in ("target", "source_register", "target_register"):
                    if endpoint.get(field):
                        names.add(endpoint[field])
    for i, name in enumerate(sorted(names - set(mapping)), 1):
        mapping[name] = f"context_{i:05d}"
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("Nonunique anonymous stage IDs")
    initial_predictions, predictions = ([], {name: [] for name in identities.values()})
    for rank, node in enumerate(ranked_nodes, 1):
        s = samples[node]
        score = s.get("trojan_score")
        if not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError("Invalid detector score")
        drivers = []
        for driver in s.get("drivers", []):
            name = driver["instance_name"]
            if name in predictions:
                predictions[name].append(
                    {
                        "rank": rank,
                        "score": score,
                        "output_port": driver["output_port"],
                        "selected_initially": node in selected,
                    }
                )
            if node in selected:
                drivers.append(
                    {
                        "instance_id": mapping[name],
                        "cell_type": driver["cell_type"],
                        "output_port": driver["output_port"],
                    }
                )
        if node in selected:
            initial_predictions.append({"rank": rank, "score": score, "drivers": drivers})
    payload_names = set(payload_result["candidate_instances"])
    payload_groups = {g["group_id"]: g for g in payload_result["groups"]}

    def interface(row):
        clean = {"role": row["role"]}
        for field in ("target", "source_register", "target_register"):
            if field in row:
                clean[field] = mapping[row[field]]
        if "port" in row:
            clean["port"] = row["port"]
        return clean

    def audit_row(row):
        clean = {"reason": row["reason"], "interface": interface(row["interface"])}
        if "instance_name" in row:
            clean["instance_id"] = mapping[row["instance_name"]]
        return clean

    clean_groups = {}
    for gid, group in groups.items():
        raw = raw_groups[gid]
        source = raw["source_payload_group"]
        p = payload_groups[source]
        members = {n["id"] for n in group["graph"]["nodes"]}
        if {mapping[n] for n in raw["candidate_instances"]} != members:
            raise ValueError("Stage-result identity map disagrees with candidate graph")
        rows = []
        for anon in sorted(members):
            name = identities[anon]
            rows.append(
                {
                    "instance_id": anon,
                    "selected_initially": name in detected_names,
                    "in_payload_result": name in payload_names,
                    "origin_stage": (
                        "nhtd_gl"
                        if name in detected_names
                        else "payloadchecker" if name in payload_names else "modulechecker"
                    ),
                    "predictions": predictions[name],
                }
            )
        audits = []
        for a in recovery:
            if a["source_payload_group"] != source:
                continue
            audits.append(
                {
                    "new_scope_instances": _ids(a["new_scope_instances"], mapping),
                    "accepted_new_instances": _ids(a["accepted_new_instances"], mapping),
                    "rejected_new_scope_instances": _ids(
                        a["rejected_new_scope_instances"], mapping
                    ),
                    "interfaces": [interface(row) for row in a["interfaces"]],
                    "proposals": [audit_row(row) for row in a["proposals"]],
                    "boundaries": [audit_row(row) for row in a["boundaries"]],
                    "budget_truncated": a["budget_truncated"],
                    "settings": a["settings"],
                }
            )
        clean_groups[gid] = {
            "instances": rows,
            "nhtd_gl": {
                "initial_screened_nodes": len(initial_nodes),
                "initial_predictions": initial_predictions,
            },
            "payloadchecker": {
                "source_group_id": source,
                "accepted": p["accepted"],
                "seed_instances": _ids(p["seed_instances"], mapping),
                "expanded_instances": _ids(p["expanded_instances"], mapping),
                "expansion_steps": [
                    {
                        "hop": s["hop"],
                        "active_directions": s["active_directions"],
                        "added_instances": _ids(s["added_instances"], mapping),
                    }
                    for s in p["expansion_steps"]
                ],
                "payload_clues": deepcopy(group["payload_clues"]),
                "budget_truncated": p["budget_truncated"],
                "settings": payload_result["settings"],
            },
            "modulechecker": {
                "kind": raw["kind"],
                "candidate_instances": sorted(members),
                "module_evidence": deepcopy(group["module_evidence"]),
                "recovery": audits,
            },
        }
    return {"schema": "generator_pipeline_results_v1", "groups": clean_groups}


class PipelineResults:
    """Query only this job's prepared snapshot. There is no file/path argument."""

    def __init__(self, frozen, payload, max_result_bytes=131072):
        if frozen.get("schema") != "generator_pipeline_results_v1":
            raise ValueError("Unknown stage-result snapshot")
        self.frozen = frozen
        self.groups = {
            g["group_id"]: {n["id"] for n in g["graph"]["nodes"]}
            for g in payload["candidate_groups"]
        }
        if set(self.groups) != set(frozen["groups"]):
            raise ValueError("Stage-result snapshot group mismatch")
        self.max_result_bytes = max_result_bytes

    def lookup(self, args):
        if not isinstance(args, dict) or set(args) != {"group_id", "stage", "instance_ids"}:
            raise ValueError("Use group_id, stage and instance_ids only")
        gid, stage, ids = (args["group_id"], args["stage"], args["instance_ids"])
        if not isinstance(gid, str) or gid not in self.groups or stage not in STAGES:
            raise ValueError("Unknown candidate group or stage")
        if (
            not isinstance(ids, list)
            or len(ids) > 128
            or any((not isinstance(i, str) for i in ids))
            or (len(set(ids)) != len(ids))
            or (not set(ids) <= self.groups[gid])
        ):
            raise ValueError("instance_ids must be unique in-group candidate IDs")
        data = deepcopy(self.frozen["groups"][gid])
        wanted = set(ids) or self.groups[gid]
        result = {
            "status": "ok",
            "group_id": gid,
            "stage": stage,
            "read_only": True,
            "instances": [r for r in data["instances"] if r["instance_id"] in wanted],
            "limitations": "Stored predictions and structural hypotheses, not ground truth. Context-only IDs cannot be selected in final decisions.",
        }
        for name in STAGES[1:] if stage == "all" else [stage]:
            result[name] = data[name]
        if ids:
            if "nhtd_gl" in result:
                result["nhtd_gl"]["initial_predictions"] = [
                    r
                    for r in data["nhtd_gl"]["initial_predictions"]
                    if any((d["instance_id"] in wanted for d in r["drivers"]))
                ]
                result["nhtd_gl"]["initial_predictions_filtered"] = True
            if "modulechecker" in result:
                result["modulechecker"].pop("module_evidence")
                result["modulechecker"][
                    "module_evidence_location"
                ] = "Already supplied in initial candidate graph"
        encoded = json.dumps(result, ensure_ascii=False).encode()
        if len(encoded) > self.max_result_bytes:
            raise ValueError(
                "Stored result exceeds tool-result byte budget; request one stage and focused instance_ids"
            )
        return result


def execute_call(reader, arguments):
    args = None
    try:
        args = json.loads(arguments)
        result = reader.lookup(args)
    except (ValueError, TypeError) as exc:
        result = {"status": "error", "read_only": True, "error": str(exc)}
    return {"name": TOOL_NAME, "arguments": args, "result": result}
