"""Schema-aware storage helpers for evolution artifacts."""

from __future__ import annotations

import json
import platform
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from liq.core.security import serialize_sensitive_payload
from liq.store.exceptions import ArtifactMigrationError, PathTraversalError, SchemaVersionError

EVOLUTION_ARTIFACT_SCHEMA_VERSION = "2.0"
SUPPORTED_EVOLUTION_ARTIFACT_SCHEMA_VERSIONS = ("1.0", "2.0")


def _normalize_seed_lineage(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [int(value)]
    if not isinstance(value, (list, tuple)):
        return []

    lineage: list[int] = []
    for entry in value:
        if isinstance(entry, (int, float)) and not isinstance(entry, bool):
            lineage.append(int(entry))
    return lineage


def _migrate_v1_to_v2(payload: Mapping[str, Any]) -> dict[str, Any]:
    run_id = str(payload.get("run_id", "unknown_run"))
    metadata = payload.get("metadata")
    metadata_map = dict(metadata) if isinstance(metadata, Mapping) else {}

    selected = payload.get("selected_candidate_ids")
    if not isinstance(selected, list):
        selected = payload.get("selected_programs")
    if not isinstance(selected, list):
        selected = []
    selected_candidate_ids = [str(item) for item in selected]

    rejection_events: list[dict[str, Any]] = []
    raw_events = payload.get("rejection_events")
    if isinstance(raw_events, list):
        for event in raw_events:
            if not isinstance(event, Mapping):
                continue
            rejection_events.append(
                {
                    "code": str(event.get("code", "unknown")),
                    "phase": str(event.get("phase", "unknown")),
                    "detail": event.get("detail"),
                    "penalty": float(event.get("penalty", 0.0) or 0.0),
                }
            )
    else:
        raw_reasons = payload.get("rejection_reasons", payload.get("rejections", []))
        if isinstance(raw_reasons, list):
            for reason in raw_reasons:
                rejection_events.append(
                    {
                        "code": str(reason),
                        "phase": "unknown",
                        "detail": None,
                        "penalty": 0.0,
                    }
                )

    raw_dependencies = payload.get("dependency_fingerprint", payload.get("dependencies", {}))
    dependency_versions = {}
    if isinstance(raw_dependencies, Mapping):
        for name, version in raw_dependencies.items():
            dependency_versions[str(name)] = str(version)

    seed_lineage = _normalize_seed_lineage(payload.get("seed_lineage"))
    if not seed_lineage:
        seed_lineage = _normalize_seed_lineage(payload.get("seed"))

    per_split_metrics = payload.get("per_split_metrics")
    if not isinstance(per_split_metrics, Mapping):
        per_split_metrics = {}

    return {
        "schema_version": EVOLUTION_ARTIFACT_SCHEMA_VERSION,
        "run_id": run_id,
        "protocol_version": str(payload.get("protocol_version", "1.0")),
        "created_at_utc": str(payload.get("created_at_utc", payload.get("created_at", ""))),
        "dependency_fingerprint": {
            "python_version": str(payload.get("python_version", sys.version.split()[0])),
            "platform": str(payload.get("platform", platform.platform())),
            "package_versions": dict(sorted(dependency_versions.items())),
            "seed_lineage": seed_lineage,
            "captured_at_utc": str(payload.get("captured_at_utc", "")),
            "evaluator_fingerprint": payload.get("evaluator_fingerprint"),
        },
        "selected_candidate_ids": selected_candidate_ids,
        "per_split_metrics": dict(per_split_metrics),
        "rejection_events": rejection_events,
        "metadata": metadata_map,
    }


def normalize_evolution_artifact_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize any supported schema into canonical v2 payload."""
    version = str(payload.get("schema_version", "1.0"))
    if version not in SUPPORTED_EVOLUTION_ARTIFACT_SCHEMA_VERSIONS:
        supported = ", ".join(SUPPORTED_EVOLUTION_ARTIFACT_SCHEMA_VERSIONS)
        raise SchemaVersionError(
            f"Unsupported evolution artifact schema_version={version!r}; supported: {supported}."
        )

    if version == EVOLUTION_ARTIFACT_SCHEMA_VERSION:
        return dict(payload)

    try:
        return _migrate_v1_to_v2(payload)
    except Exception as exc:  # pragma: no cover - defensive guard
        raise ArtifactMigrationError(f"Failed to migrate schema_version={version!r}") from exc


def serialize_evolution_artifact_payload(payload: Mapping[str, Any]) -> bytes:
    """Serialize a payload with deterministic key ordering."""
    normalized = normalize_evolution_artifact_payload(payload)
    return serialize_sensitive_payload(normalized)


def deserialize_evolution_artifact_payload(
    raw_payload: bytes | str | Mapping[str, Any],
) -> dict[str, Any]:
    """Deserialize and normalize an evolution artifact payload."""
    if isinstance(raw_payload, bytes):
        parsed = json.loads(raw_payload.decode("utf-8"))
    elif isinstance(raw_payload, str):
        parsed = json.loads(raw_payload)
    elif isinstance(raw_payload, Mapping):
        parsed = dict(raw_payload)
    else:
        raise TypeError("raw_payload must be bytes, str, or mapping")

    if not isinstance(parsed, Mapping):
        raise ArtifactMigrationError("artifact payload must decode to a mapping")
    return normalize_evolution_artifact_payload(parsed)


class LocalArtifactStore:
    """Filesystem-backed key-value store for artifact payload bytes."""

    def __init__(self, data_root: str | Path) -> None:
        self.data_root = Path(data_root).resolve()
        self.data_root.mkdir(parents=True, exist_ok=True)

    def _key_to_path(self, key: str) -> Path:
        rel = Path(key)
        if rel.is_absolute() or ".." in rel.parts:
            raise PathTraversalError(f"invalid artifact key: {key!r}")
        candidate = (self.data_root / rel).resolve()
        try:
            candidate.relative_to(self.data_root)
        except ValueError as exc:
            raise PathTraversalError(f"artifact key escapes data root: {key!r}") from exc
        return candidate

    def put(self, key: str, data: bytes) -> None:
        path = self._key_to_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def get(self, key: str) -> bytes | None:
        path = self._key_to_path(key)
        if not path.exists():
            return None
        return path.read_bytes()
