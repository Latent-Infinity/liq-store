"""Tests for schema-aware artifact helpers."""

from __future__ import annotations

import json

import pytest

from liq.store.artifacts import (
    EVOLUTION_ARTIFACT_SCHEMA_VERSION,
    LocalArtifactStore,
    deserialize_evolution_artifact_payload,
    serialize_evolution_artifact_payload,
)
from liq.store.exceptions import PathTraversalError, SchemaVersionError


def test_local_artifact_store_round_trip(tmp_path) -> None:  # type: ignore[annotation-unchecked]
    store = LocalArtifactStore(tmp_path)
    key = "evolution/runs/run-1/run.json"
    payload = b'{"ok":true}'
    store.put(key, payload)
    assert store.get(key) == payload


def test_local_artifact_store_rejects_path_traversal(tmp_path) -> None:  # type: ignore[annotation-unchecked]
    store = LocalArtifactStore(tmp_path)
    with pytest.raises(PathTraversalError):
        store.put("../escape.json", b"{}")


def test_deserialize_unknown_version_raises() -> None:
    raw = json.dumps({"schema_version": "99.0", "run_id": "x"}).encode("utf-8")
    with pytest.raises(SchemaVersionError, match="Unsupported evolution artifact schema_version"):
        deserialize_evolution_artifact_payload(raw)


def test_migrate_v1_payload_to_v2() -> None:
    migrated = deserialize_evolution_artifact_payload(
        {
            "schema_version": "1.0",
            "run_id": "run-legacy",
            "seed": 42,
            "selected_programs": ["a", "b"],
            "rejection_reasons": ["degenerate_scores"],
        }
    )
    assert migrated["schema_version"] == EVOLUTION_ARTIFACT_SCHEMA_VERSION
    assert migrated["dependency_fingerprint"]["seed_lineage"] == [42]
    assert migrated["selected_candidate_ids"] == ["a", "b"]
    assert migrated["rejection_events"][0]["code"] == "degenerate_scores"


def test_serialize_deserialize_round_trip_is_deterministic() -> None:
    payload = {
        "schema_version": "2.0",
        "run_id": "run-2",
        "protocol_version": "1.0",
        "created_at_utc": "2026-03-03T00:00:00Z",
        "dependency_fingerprint": {
            "python_version": "3.12.0",
            "platform": "x",
            "package_versions": {"liq-evolution": "0.2.0"},
            "seed_lineage": [7, 11],
            "captured_at_utc": "2026-03-03T00:00:00Z",
            "evaluator_fingerprint": "abc",
        },
        "selected_candidate_ids": ["cand-1"],
        "per_split_metrics": {"time_window:0:train": {"cagr": 0.2}},
        "rejection_events": [],
        "metadata": {"z": 1, "a": 2},
    }
    first = serialize_evolution_artifact_payload(payload)
    second = serialize_evolution_artifact_payload(payload)
    assert first == second
    assert deserialize_evolution_artifact_payload(first)["run_id"] == "run-2"


def test_serialize_evolution_artifact_payload_redacts_sensitive_values() -> None:
    payload = {
        "schema_version": "2.0",
        "run_id": "run-secure",
        "protocol_version": "1.0",
        "created_at_utc": "2026-03-03T00:00:00Z",
        "dependency_fingerprint": {
            "python_version": "3.12.0",
            "platform": "linux",
            "package_versions": {"liq-evolution": "0.2.0"},
            "seed_lineage": [7],
            "captured_at_utc": "2026-03-03T00:00:00Z",
            "evaluator_fingerprint": "abc",
        },
        "selected_candidate_ids": ["cand-1"],
        "per_split_metrics": {"time_window:0:train": {"cagr": 0.2}},
        "rejection_events": [],
        "metadata": {
            "api_key": "top-secret",
            "token": "session",
            "model": "xgboost",
        },
    }

    encoded = serialize_evolution_artifact_payload(payload)
    decoded = encoded.decode("utf-8")
    assert "top-secret" not in decoded
    assert "session" not in decoded

    round_trip = deserialize_evolution_artifact_payload(encoded)
    assert round_trip["metadata"]["api_key"] == "***REDACTED***"
    assert round_trip["metadata"]["token"] == "***REDACTED***"
    assert round_trip["metadata"]["model"] == "xgboost"
