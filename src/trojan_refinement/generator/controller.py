"""Finite, read-only tool loop with strict per-instance decision validation."""

from copy import deepcopy
import json
import re

from . import contracts, stage_results
from .prompts import SYSTEM, STAGE_INSTRUCTIONS
from ..rag import retriever as retrieval


def parse_decision(text):
    match = re.fullmatch(r"\s*```(?:json)?\s*\n(.*?)\n```\s*", text, re.S)
    value = json.loads(match[1] if match else text)
    contracts.validate_json_schema(value, contracts.SCHEMA)
    return value


def refine_candidates(payload, snapshot, retriever, client, *, on_event=None):
    """No structural tools or baseline inference can be executed from this loop."""
    config = client.config
    on_event = on_event or (lambda event: None)
    reader = stage_results.PipelineResults(snapshot, payload, config.max_result_bytes)
    available = [(stage_results.TOOL, config.max_result_calls)]
    if retriever is not None:
        available.append((retrieval.TOOL, config.max_rag_calls))
    remaining = {tool["name"]: budget for tool, budget in available}
    system = (
        SYSTEM
        + "\n\n"
        + STAGE_INSTRUCTIONS
        + f"\nStored-result lookup budget: {config.max_result_calls}."
        + f"\nRAG search budget: {config.max_rag_calls if retriever is not None else 0}."
    )
    history = [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, allow_nan=False)},
    ]
    log = {
        "status": "running",
        "model_config": config.public_dict(),
        "rag_calls": [],
        "stage_result_calls": [],
        "api_responses": [],
    }
    seen = set()
    if not payload["candidate_groups"]:
        decision = {"decisions": [], "summary": "No candidate groups; no API request."}
    else:
        for _ in range(sum(remaining.values()) + 1):
            tools = [tool for tool, _ in available if remaining[tool["name"]] > 0]
            reply = client.create(history, tools)
            log["api_responses"].append(reply["raw"])
            on_event(deepcopy(log))
            calls = reply["calls"]
            if not calls:
                decision = parse_decision(reply["text"])
                break
            # Check the entire requested batch before executing even read-only calls.
            requested = {}
            for call in calls:
                name, call_id = call["name"], call["id"]
                if not isinstance(call_id, str) or not call_id or call_id in seen:
                    raise ValueError("Missing or duplicate tool-call ID")
                seen.add(call_id)
                requested[name] = requested.get(name, 0) + 1
                if requested[name] > remaining.get(name, 0):
                    raise ValueError("Unknown tool or exceeded call budget")
                if not isinstance(call["arguments"], str):
                    raise ValueError("Expected serialized JSON tool arguments")
            results = []
            for call in calls:
                name = call["name"]
                remaining[name] -= 1
                if name == stage_results.TOOL_NAME:
                    entry = stage_results.execute_call(reader, call["arguments"])
                    log["stage_result_calls"].append(entry)
                else:
                    entry = retrieval.execute_call(retriever, name, call["arguments"])
                    log["rag_calls"].append(entry)
                results.append(
                    {"id": call["id"], "content": json.dumps(entry["result"], ensure_ascii=False)}
                )
            history.extend(deepcopy(reply["continuation"]))
            history.extend(client.tool_messages(results))
            on_event(deepcopy(log))
        else:
            raise ValueError("Generator did not finish within its tool budget")
    refs = retrieval.returned_reference_ids(log["rag_calls"])
    contracts.validate_json_schema(decision, contracts.SCHEMA)
    selected = contracts.validate(decision, payload, refs)
    # Overlapping groups are combined by union: a removal in one group cannot
    # erase the same instance retained by another supported group.
    retained = sorted(set().union(*selected.values())) if selected else []
    log.update(status="valid", decision=decision, retained_anonymous_instances=retained)
    on_event(deepcopy(log))
    return log
