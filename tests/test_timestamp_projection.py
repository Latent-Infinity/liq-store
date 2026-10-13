import hashlib
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from liq.store import timestamp_projection
from liq.store.timestamp_metadata import (
    PinnedParquetFile,
    TimestampFileMetadata,
    TimestampMetadataRefused,
    TimestampRowGroupMetadata,
    inspect_timestamp_metadata,
)
from liq.store.timestamp_projection import TimestampProjectionLimits, iter_pinned_timestamps


@pytest.fixture
def empty_projection_source(tmp_path: Path) -> tuple[Path, TimestampFileMetadata]:
    path = tmp_path / "empty.parquet"
    pq.write_table(pa.table({"timestamp": pa.array([], type=pa.timestamp("us", tz="UTC"))}), path)
    identity = PinnedParquetFile(path.name, hashlib.sha256(path.read_bytes()).hexdigest())
    expected = TimestampFileMetadata(identity, 1, (TimestampRowGroupMetadata(0, 1, 0, 1),))
    return tmp_path, expected


def test_empty_schema_refuses_before_projection(empty_projection_source, monkeypatch) -> None:
    root, expected = empty_projection_source
    monkeypatch.setattr(
        pq.ParquetFile, "iter_batches", lambda *_a, **_k: pytest.fail("pages reached")
    )

    with pytest.raises(TimestampMetadataRefused, match="empty"):
        tuple(iter_pinned_timestamps(root, expected, TimestampProjectionLimits(10000, 1, 1, 1)))


def test_changed_bytes_refuse_before_parser(empty_projection_source, monkeypatch) -> None:
    root, expected = empty_projection_source
    (root / expected.identity.path).write_bytes(b"Nonfinancial corrupt-format sentinel")
    monkeypatch.setattr(pq, "read_metadata", lambda *_a, **_k: pytest.fail("parser reached"))

    with pytest.raises(TimestampMetadataRefused, match="digest"):
        tuple(iter_pinned_timestamps(root, expected, TimestampProjectionLimits(10000, 1, 1, 1)))


@pytest.mark.parametrize("position", range(4))
def test_projection_limits_refuse_nonpositive_values(position: int) -> None:
    values = [10000, 1, 1, 1]
    values[position] = 0

    with pytest.raises(TimestampMetadataRefused, match="budget"):
        TimestampProjectionLimits(*values)


@pytest.fixture
def real_projection_source(tmp_path: Path) -> tuple[Path, TimestampFileMetadata, tuple[int, ...]]:
    fixture = Path(__file__).parent / "fixtures/approved/khc-timestamps-20230421-20230424.parquet"
    digest = "7920d9eabd8b2a3ce9392ad82815b8a9eb8c52409c10846ffbcfcecb5fbc49bc"
    assert hashlib.sha256(fixture.read_bytes()).hexdigest() == digest
    target = tmp_path / fixture.name
    shutil.copyfile(fixture, target)
    identity = PinnedParquetFile(target.name, digest)
    expected = inspect_timestamp_metadata(
        tmp_path,
        (identity,),
        max_files=1,
        max_file_bytes=10000,
        max_row_groups=1,
    )[0]
    reference = pq.read_table(target, columns=["timestamp"])["timestamp"]
    values = tuple(value * 1000 for value in reference.cast(pa.int64()).to_pylist())
    assert expected.row_count == len(values) == 449
    return tmp_path, expected, values


def test_real_timestamp_projection_preserves_all_values_and_order(
    real_projection_source, monkeypatch
) -> None:
    root, expected, reference = real_projection_source
    actual_batches = pq.ParquetFile.iter_batches
    requests = []

    def observe(source, **options):
        requests.append(options)
        return actual_batches(source, **options)

    monkeypatch.setattr(pq.ParquetFile, "iter_batches", observe)
    batches = tuple(
        iter_pinned_timestamps(root, expected, TimestampProjectionLimits(10000, 449, 1, 128))
    )

    assert tuple(value for batch in batches for value in batch.timestamps_ns) == reference
    assert [batch.row_offset for batch in batches] == [0, 128, 256, 384]
    assert requests == [
        {
            "batch_size": 128,
            "row_groups": [0],
            "columns": ["timestamp"],
            "use_threads": False,
            "use_pandas_metadata": False,
        }
    ]


def test_projection_uses_verified_buffer_after_path_changes(
    real_projection_source, monkeypatch
) -> None:
    root, expected, reference = real_projection_source
    read = timestamp_projection._pinned_bytes

    def replace_after_read(root, identity, maximum):
        payload = read(root, identity, maximum)
        (root / identity.path).write_bytes(b"Nonfinancial replacement sentinel")
        return payload

    monkeypatch.setattr(timestamp_projection, "_pinned_bytes", replace_after_read)
    batches = tuple(
        iter_pinned_timestamps(root, expected, TimestampProjectionLimits(10000, 449, 1, 128))
    )

    assert tuple(value for batch in batches for value in batch.timestamps_ns) == reference


def test_changed_admitted_footer_refuses_before_pages(real_projection_source, monkeypatch) -> None:
    root, expected, _ = real_projection_source
    expected = replace(expected, row_count=448)
    monkeypatch.setattr(
        pq.ParquetFile, "iter_batches", lambda *_a, **_k: pytest.fail("pages reached")
    )

    with pytest.raises(TimestampMetadataRefused, match="footer differs"):
        tuple(iter_pinned_timestamps(root, expected, TimestampProjectionLimits(10000, 449, 1, 128)))


def test_declared_rows_exceeding_budget_refuse_before_bytes(
    real_projection_source, monkeypatch
) -> None:
    root, expected, _ = real_projection_source
    monkeypatch.setattr(
        timestamp_projection, "_pinned_bytes", lambda *_: pytest.fail("bytes reached")
    )

    with pytest.raises(TimestampMetadataRefused, match="declared row budget"):
        tuple(iter_pinned_timestamps(root, expected, TimestampProjectionLimits(10000, 448, 1, 128)))


@pytest.mark.parametrize(
    "mode,reason",
    [
        ("schema", "schema mismatch"),
        ("batch_rows", "batch row budget"),
        ("total_rows", "physical row budget"),
        ("null_metadata", "null physical timestamp"),
        ("empty", "row-group declaration mismatch"),
        ("arrow_error", "unreadable timestamp pages"),
    ],
)
def test_projection_refuses_control_metadata_faults(
    real_projection_source, monkeypatch, mode: str, reason: str
) -> None:
    root, expected, _ = real_projection_source
    original = pq.ParquetFile.iter_batches
    if mode == "total_rows":
        expected = replace(expected, row_groups=(replace(expected.row_groups[0], row_count=127),))
        monkeypatch.setattr(timestamp_projection, "_inspect_file", lambda *_: expected)

    def faulty(source, **options):
        batch = next(original(source, **options))
        match mode:
            case "schema":
                yield SimpleNamespace(schema=pa.schema([]))
            case "batch_rows":
                yield SimpleNamespace(schema=batch.schema, column=batch.column, num_rows=1000)
            case "total_rows":
                yield batch
            case "null_metadata":
                column = SimpleNamespace(type=batch.column(0).type, null_count=1)
                yield SimpleNamespace(schema=batch.schema, column=lambda _: column, num_rows=128)
            case "empty":
                return
            case "arrow_error":
                raise pa.ArrowInvalid("Nonfinancial decoder-failure sentinel")
            case _:
                pytest.fail("unknown control-metadata mode")

    monkeypatch.setattr(pq.ParquetFile, "iter_batches", faulty)
    with pytest.raises(TimestampMetadataRefused, match=reason):
        tuple(iter_pinned_timestamps(root, expected, TimestampProjectionLimits(10000, 449, 1, 128)))


@pytest.mark.parametrize(
    "widen,reason", [(False, "outside declared bounds"), (True, "row-group declaration mismatch")]
)
def test_physical_extrema_must_equal_declared_metadata(
    real_projection_source, monkeypatch, widen: bool, reason: str
) -> None:
    root, expected, _ = real_projection_source
    group = expected.row_groups[0]
    group = replace(group, minimum_ns=group.minimum_ns + (-1 if widen else 1))
    expected = replace(expected, row_groups=(group,))
    monkeypatch.setattr(timestamp_projection, "_inspect_file", lambda *_: expected)

    with pytest.raises(timestamp_projection.TimestampProjectionRefused, match=reason) as caught:
        tuple(iter_pinned_timestamps(root, expected, TimestampProjectionLimits(10000, 449, 1, 128)))
    if not widen:
        assert caught.value.row_offset == 0
        assert caught.value.timestamp_ns == group.minimum_ns - 1


@pytest.mark.parametrize(
    "fault,reason",
    [("root", "root path"), ("digest", "invalid digest"), ("groups", "row-group budget")],
)
def test_invalid_projection_request_refuses_before_bytes(
    real_projection_source, monkeypatch, fault: str, reason: str
) -> None:
    root, expected, _ = real_projection_source
    match fault:
        case "root":
            root = Path("relative")
        case "digest":
            expected = replace(expected, identity=replace(expected.identity, sha256="invalid"))
        case "groups":
            expected = replace(expected, row_groups=())
        case _:
            pytest.fail("unknown request fault")
    monkeypatch.setattr(
        timestamp_projection, "_pinned_bytes", lambda *_: pytest.fail("bytes reached")
    )

    with pytest.raises(TimestampMetadataRefused, match=reason):
        tuple(iter_pinned_timestamps(root, expected, TimestampProjectionLimits(10000, 449, 1, 128)))
