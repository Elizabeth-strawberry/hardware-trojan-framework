"""Small JSON helpers and a content manifest for prepared read-only inputs."""

import hashlib
import json
import os
from pathlib import Path

INPUT_FILES = (
    "generator_input.json",
    "pipeline_results.json",
    "payload_result.json",
    "module_result.json",
    "identity_map.json",
    "node_mapping.json",
    "metadata.json",
)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_prepared(directory, result):
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Use an empty output directory; prepared inputs are immutable")
    for filename in INPUT_FILES:
        write_json(directory / filename, result[filename.removesuffix(".json")])
    write_json(
        directory / "manifest.json",
        {
            "schema": "trojan_refinement_prepared_v1",
            "sha256": {filename: file_hash(directory / filename) for filename in INPUT_FILES},
        },
    )


def load_prepared(directory):
    directory = Path(directory)
    manifest = read_json(directory / "manifest.json")
    if manifest.get("schema") != "trojan_refinement_prepared_v1" or set(manifest["sha256"]) != set(
        INPUT_FILES
    ):
        raise ValueError("Unknown prepared-input manifest")
    for filename in INPUT_FILES:
        if file_hash(directory / filename) != manifest["sha256"][filename]:
            raise ValueError("Prepared input changed: " + filename)
    return {
        filename.removesuffix(".json"): read_json(directory / filename) for filename in INPUT_FILES
    }
