"""Instance-level decision schema, complete partitions, removal evidence, and offline identity validation."""

_IDS = {"type": "array", "items": {"type": "string"}}
REMOVAL_CATEGORIES = [
    "host_data_branch",
    "ordinary_datapath",
    "unrelated_logic",
    "invalid_payload_interpretation",
    "other_structural_reason",
]
SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["decisions", "summary"],
    "properties": {
        "summary": {"type": "string"},
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "group_id",
                    "decision",
                    "retained_instances",
                    "removed_instances",
                    "reason",
                    "evidence_instances",
                    "reference_ids",
                    "removal_evidence",
                ],
                "properties": {
                    "group_id": {"type": "string"},
                    "decision": {"type": "string", "enum": ["keep", "reject", "refine"]},
                    "retained_instances": _IDS,
                    "removed_instances": _IDS,
                    "reason": {"type": "string"},
                    "evidence_instances": _IDS,
                    "reference_ids": _IDS,
                    "removal_evidence": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["instances", "category", "reason", "evidence_instances"],
                            "properties": {
                                "instances": _IDS,
                                "category": {"type": "string", "enum": REMOVAL_CATEGORIES},
                                "reason": {"type": "string"},
                                "evidence_instances": _IDS,
                            },
                        },
                    },
                },
            },
        },
    },
}


def _ids(value, allowed, field):
    if not isinstance(value, list) or any((not isinstance(x, str) for x in value)):
        raise ValueError("Expected string array: " + field)
    if len(set(value)) != len(value) or not set(value) <= allowed:
        raise ValueError("Duplicate/unknown IDs: " + field)
    return set(value)


def _reason(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Missing structural reason")


def validate(decision, payload, extra_reference_ids=()):
    """Check complete partitions and removal evidence, without reading labels."""
    groups = {
        g["group_id"]: {n["id"] for n in g["graph"]["nodes"]} for g in payload["candidate_groups"]
    }
    refs = {r["reference_id"] for r in payload["references"]} | set(extra_reference_ids)
    if not isinstance(decision, dict) or not isinstance(decision.get("summary"), str):
        raise ValueError("Missing decision object/summary")
    rows = decision.get("decisions")
    if not isinstance(rows, list) or len(rows) != len(groups):
        raise ValueError("Wrong group coverage")
    selected, seen = ({}, set())
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Decision must be an object")
        gid = row.get("group_id")
        if not isinstance(gid, str) or gid not in groups or gid in seen:
            raise ValueError("Unknown/duplicate group")
        seen.add(gid)
        _reason(row.get("reason"))
        kept = _ids(row.get("retained_instances"), groups[gid], "retained_instances")
        removed = _ids(row.get("removed_instances"), groups[gid], "removed_instances")
        if kept & removed or kept | removed != groups[gid]:
            raise ValueError("Retained/removed instances must form a complete disjoint partition")
        evidence = _ids(row.get("evidence_instances"), groups[gid], "evidence_instances")
        _ids(row.get("reference_ids"), refs, "reference_ids")
        action = row.get("decision")
        if action == "keep":
            if removed or kept != groups[gid]:
                raise ValueError("keep must retain the entire group")
        elif action == "reject":
            if kept or removed != groups[gid] or (not evidence):
                raise ValueError("reject must remove the entire group with evidence")
        elif action == "refine":
            if not kept or not removed or (not evidence):
                raise ValueError("refine needs retained and removed instances with evidence")
        else:
            raise ValueError("Unknown decision")
        blocks = row.get("removal_evidence")
        if not isinstance(blocks, list):
            raise ValueError("Missing removal_evidence array")
        covered = set()
        for block in blocks:
            if not isinstance(block, dict):
                raise ValueError("Removal evidence must be an object")
            instances = _ids(block.get("instances"), removed, "removal_evidence.instances")
            supporting = _ids(
                block.get("evidence_instances"), groups[gid], "removal_evidence.evidence_instances"
            )
            if not instances or not supporting or covered & instances:
                raise ValueError(
                    "Removal blocks need nonoverlapping instances and structural evidence"
                )
            if block.get("category") not in REMOVAL_CATEGORIES:
                raise ValueError("Unknown removal category")
            _reason(block.get("reason"))
            covered |= instances
        if covered != removed:
            raise ValueError("Every removed instance needs removal evidence")
        selected[gid] = kept
    return selected


def validate_identity_map(payload, module, identities):
    """Check the offline anonymous-to-original map separately from model input."""
    original = {g["group_id"]: set(g["candidate_instances"]) for g in module["groups"]}
    all_ids = {n["id"] for g in payload["candidate_groups"] for n in g["graph"]["nodes"]}
    if set(identities) != all_ids or len(set(identities.values())) != len(identities):
        raise ValueError("Incomplete/nonunique anonymous identity map")
    for group in payload["candidate_groups"]:
        mapped = {identities[n["id"]] for n in group["graph"]["nodes"]}
        if mapped != original.get(group["group_id"]):
            raise ValueError("Anonymous identity map does not match group membership")


def validate_json_schema(value, schema, path="$"):
    """Validate the explicit JSON contract locally without repairing model output."""
    typ = schema.get("type")
    valid_type = {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "integer": type(value) is int,
        "number": type(value) in (int, float),
        "boolean": type(value) is bool,
        "null": value is None,
    }
    if typ not in valid_type or not valid_type[typ]:
        raise ValueError("Output schema type mismatch at " + path)
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError("Output schema enum mismatch at " + path)
    if typ == "object":
        props = schema.get("properties", {})
        if not set(schema.get("required", [])) <= set(value):
            raise ValueError("Missing required output fields at " + path)
        if schema.get("additionalProperties") is False and (not set(value) <= set(props)):
            raise ValueError("Unexpected output fields at " + path)
        for k, v in value.items():
            if k in props:
                validate_json_schema(v, props[k], path + "." + k)
    elif typ == "array":
        for i, v in enumerate(value):
            validate_json_schema(v, schema["items"], f"{path}[{i}]")
