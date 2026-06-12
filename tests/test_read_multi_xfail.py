"""Contract stub for ``TimeSeriesStore.read_multi`` (planned, not built).

Strict xfail. Flips green when the cross-sectional read path lands
per the liq-scan plan §3.6:

    def read_multi(
        self,
        keys: Sequence[str],
        start: datetime,
        end: datetime,
        *,
        columns: Sequence[str] | None = None,
    ) -> pl.DataFrame: ...

The signature here is intentionally minimal — the full contract test
(missing-key behavior, monthly-partition backwards-compat union, etc.)
lands with the real implementation. See ``liq-scan-plan.md`` §3.6 / §3.7.
"""

from __future__ import annotations

import pytest


@pytest.mark.xfail(
    strict=True,
    reason="TimeSeriesStore.read_multi not yet implemented (planned)",
)
def test_read_multi_protocol_member_exists() -> None:
    from liq.store.protocols import TimeSeriesStore  # noqa: PLC0415 — protocol target

    assert callable(getattr(TimeSeriesStore, "read_multi", None))
