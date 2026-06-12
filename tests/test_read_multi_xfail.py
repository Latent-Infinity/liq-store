"""Phase 0 contract stub: ``TimeSeriesStore.read_multi`` (Phase 3 deliverable).

Strict xfail flips green when Phase 3 lands the cross-sectional read
path per liq-scan-plan §3.6:

    def read_multi(
        self,
        keys: Sequence[str],
        start: datetime,
        end: datetime,
        *,
        columns: Sequence[str] | None = None,
    ) -> pl.DataFrame: ...

The signature here is intentionally minimal — Phase 3 owns the full
contract test, including missing-key behavior and the monthly-partition
backwards-compat union (plan §3.7).
"""

from __future__ import annotations

import pytest


@pytest.mark.xfail(
    strict=True,
    reason="Phase 3 deliverable — TimeSeriesStore.read_multi not yet implemented",
)
def test_read_multi_protocol_member_exists() -> None:
    from liq.store.protocols import TimeSeriesStore  # noqa: PLC0415 — protocol target

    assert callable(getattr(TimeSeriesStore, "read_multi", None))
