"""Tests for key builder helpers."""

from liq.store import key_builder


def test_bars_key() -> None:
    assert key_builder.bars("EUR_USD", "1m") == "EUR_USD/bars/1m"


def test_features_key() -> None:
    assert key_builder.features("AAPL", "midrange_default") == "AAPL/features/midrange_default"


def test_indicators_key() -> None:
    assert key_builder.indicators("AAPL", "rsi", "period14") == "AAPL/indicators/rsi/period14"


def test_fundamentals_key() -> None:
    assert key_builder.fundamentals("AAPL") == "AAPL/fundamentals"


def test_quotes_key() -> None:
    assert key_builder.quotes("EUR_USD") == "EUR_USD/quotes"


def test_corp_actions_key() -> None:
    assert key_builder.corp_actions("SPY") == "SPY/corp_actions"
