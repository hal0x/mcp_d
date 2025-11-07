import pathlib
import sys
from typing import List

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from types import SimpleNamespace as _SimpleNamespace

from finance.finrl_agent import AnalysisResult, FinRLAgent


def test_basic_analysis(monkeypatch: pytest.MonkeyPatch) -> None:
    dates = pd.date_range("2023-01-01", periods=25, freq="D")
    close = 100 * (1.01 ** np.arange(25))
    df = pd.DataFrame({"Close": close}, index=dates)

    dummy_yf = _SimpleNamespace(download=lambda *args, **kwargs: df.copy())
    monkeypatch.setattr("finance.finrl_agent.yf", dummy_yf)

    agent = FinRLAgent()
    result = agent._basic_analysis(["MOCK"], "3mo", "1d")

    metrics = result.metrics["MOCK"]
    assert metrics["mean_return"] == pytest.approx(0.01, rel=1e-6)
    assert metrics["volatility"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["momentum_5"] == pytest.approx((1.01**5) - 1, rel=1e-6)
    assert metrics["momentum_20"] == pytest.approx((1.01**20) - 1, rel=1e-6)
    assert result.bullets == ["MOCK: восходящий импульс (5/20)."]


def test_analyze_uses_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy_yf = _SimpleNamespace(
        download=lambda *args, **kwargs: pd.DataFrame(
            {"Close": [1.0]}, index=pd.date_range("2023-01-01", periods=1)
        )
    )
    monkeypatch.setattr("finance.finrl_agent.yf", dummy_yf)

    agent = FinRLAgent(default_period="1mo", default_interval="1h")

    captured: dict[str, str] = {}

    def fake_basic(tickers: List[str], period: str, interval: str) -> AnalysisResult:
        captured["period"] = period
        captured["interval"] = interval
        return AnalysisResult("", [], {}, [])

    agent._basic_analysis = fake_basic  # type: ignore[method-assign]

    agent.analyze(["MOCK"])

    assert captured["period"] == "1mo"
    assert captured["interval"] == "1h"
