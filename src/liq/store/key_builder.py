"""KeyBuilder helpers for standardized storage keys.

Provides consistent names for bars, features, indicators, and fundamentals
so that producers/consumers across the LIQ stack share the same key schema.
"""

from __future__ import annotations


def bars(symbol: str, timeframe: str) -> str:
    """Build key for bar data."""
    return f"{symbol}/bars/{timeframe}"


def features(symbol: str, feature_set: str) -> str:
    """Build key for feature sets."""
    return f"{symbol}/features/{feature_set}"


def indicators(symbol: str, indicator: str, params_id: str) -> str:
    """Build key for indicator outputs."""
    return f"{symbol}/indicators/{indicator}/{params_id}"


def fundamentals(symbol: str) -> str:
    """Build key for fundamentals."""
    return f"{symbol}/fundamentals"
