"""Fail-closed content-addressed cache helpers for the clean pipeline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return value.resolve().as_posix()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def canonical_json(value: Any) -> str:
    """Return a stable JSON representation suitable for hashing."""

    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_record(path: Path) -> dict[str, Any]:
    """Describe exact file bytes; missing files are an error."""

    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Required artifact is missing: {resolved}")
    return {
        "path": resolved.as_posix(),
        "size_bytes": int(resolved.stat().st_size),
        "sha256": sha256_file(resolved),
    }


def artifact_records(paths: Iterable[Path]) -> list[dict[str, Any]]:
    return [artifact_record(path) for path in sorted(paths, key=lambda item: item.as_posix())]


def signature(payload: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json(payload).encode("utf-8"))


def write_cache_metadata(
    path: Path,
    *,
    stage: str,
    signature_value: str,
    config_hash: str,
    implementation_hash: str,
    inputs: list[dict[str, Any]],
    outputs: Iterable[Path],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write metadata after hashing every output byte."""

    output_records = artifact_records(outputs)
    payload = {
        "pipeline_version": "leakage_free_tabular_v2",
        "stage": stage,
        "signature": signature_value,
        "config_hash": config_hash,
        "implementation_hash": implementation_hash,
        "inputs": inputs,
        "outputs": output_records,
        **(extra or {}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def cache_hit(
    metadata_path: Path,
    *,
    expected_signature: str,
    expected_outputs: Iterable[Path] | None = None,
) -> tuple[bool, str]:
    """Validate signature, byte size, and SHA-256 for every cached output."""

    if not metadata_path.is_file():
        return False, "metadata missing"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"metadata unreadable: {exc!r}"
    if metadata.get("signature") != expected_signature:
        return False, "signature changed"
    outputs = metadata.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        return False, "output inventory missing"
    if expected_outputs is not None:
        declared_paths = {
            Path(str(record.get("path", ""))).resolve().as_posix()
            for record in outputs
            if isinstance(record, dict)
        }
        required_paths = {path.resolve().as_posix() for path in expected_outputs}
        if declared_paths != required_paths:
            return False, "output inventory differs from the required stage outputs"
    for expected in outputs:
        try:
            path = Path(str(expected["path"]))
            current = artifact_record(path)
        except (KeyError, OSError, FileNotFoundError) as exc:
            return False, f"output unavailable: {exc!r}"
        if current["size_bytes"] != expected.get("size_bytes"):
            return False, f"output size changed: {path}"
        if current["sha256"] != expected.get("sha256"):
            return False, f"output SHA-256 changed: {path}"
    return True, "signature and output bytes match"
