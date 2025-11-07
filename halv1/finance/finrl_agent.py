"""FinRL-based (optional) financial analysis agent.

If FinRL is not installed, falls back to a lightweight analysis using yfinance
and simple indicators to keep the system functional.
"""

from __future__ import annotations

import base64
import datetime as dt
import logging
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


try:  # Optional dependencies
    import yfinance as yf  # type: ignore
except Exception:  # pragma: no cover
    yf = None

try:  # Check for FinRL availability
    import finrl  # type: ignore  # noqa: F401

    _HAS_FINRL = True
except Exception:  # pragma: no cover
    _HAS_FINRL = False


@dataclass
class AnalysisResult:
    report_markdown: str
    bullets: List[str]
    metrics: Dict[str, Any] = field(default_factory=dict)
    charts: List[str] = field(default_factory=list)
    advanced: bool = False


class FinRLAgent:
    """Facade to run financial analysis.

    For now, implements a minimal, dependency-light analysis that computes
    basic momentum/volatility metrics from yfinance. This can be upgraded to a
    real FinRL pipeline later.
    """

    def __init__(
        self,
        data_source: Any | None = None,
        default_period: str = "3mo",
        default_interval: str = "1d",
        use_finrl: bool | None = None,
    ) -> None:
        """Create a new analysis facade.

        Parameters
        ----------
        data_source:
            Object providing a ``download`` method compatible with
            :mod:`yfinance`. If ``None`` (default) the globally imported
            :mod:`yfinance` module is used. This allows dependency injection in
            tests or alternative data providers.
        default_period:
            Default history period used when ``analyze`` is invoked without an
            explicit ``period`` argument.
        default_interval:
            Sampling interval for the downloaded price data when no explicit
            ``interval`` is supplied.
        use_finrl:
            Whether to attempt the advanced FinRL-based analysis. If ``None``
            (default) this flag mirrors whether the :mod:`finrl` package is
            installed. Set to ``False`` to force the lightweight analysis even
            when FinRL is available.
        """

        self.data_source = data_source or yf
        self.default_period = default_period
        self.default_interval = default_interval
        self.use_finrl = _HAS_FINRL if use_finrl is None else use_finrl

    def analyze(
        self,
        tickers: List[str],
        period: str | None = None,
        interval: str | None = None,
    ) -> AnalysisResult:
        period = period or self.default_period
        interval = interval or self.default_interval
        if self.use_finrl:
            try:
                return self._advanced_analysis(tickers, period, interval)
            except Exception:
                # Fall back to basic analysis on any error
                pass
        return self._basic_analysis(tickers, period, interval)

    # ---------------------------- Basic analysis ----------------------------
    def _basic_analysis(
        self, tickers: List[str], period: str, interval: str
    ) -> AnalysisResult:
        bullets: List[str] = []
        lines: List[str] = [
            "**Финансовый обзор (базовый)**",
            "",
            "Тикеры: " + ", ".join(tickers),
            "",
        ]
        metrics: Dict[str, Dict[str, float]] = {}
        charts: List[str] = []

        if not self.data_source:
            logger.warning(
                "yfinance is not installed; install it with 'pip install yfinance'"
            )
            lines.append("yfinance не установлен — выполнен только шаблонный отчёт.")
            lines.append(
                "Установите пакет yfinance, например командой: `pip install yfinance`."
            )
            bullets.append(
                "Установите yfinance для базового анализа или finrl для продвинутого (пример: `pip install yfinance`)."
            )
            return AnalysisResult(
                "\n".join(lines), bullets, metrics, charts, advanced=False
            )

        for ticker in tickers:
            try:
                df = self.data_source.download(
                    ticker, period=period, interval=interval, progress=False
                )
                if df.empty:
                    bullets.append(f"{ticker}: нет данных за период {period}")
                    continue
                df["ret"] = df["Close"].pct_change()
                last_close = float(df["Close"].iloc[-1])
                mean_ret = float(df["ret"].mean())
                vol = float(df["ret"].std())
                mom_5 = (
                    float((df["Close"].pct_change(5)).iloc[-1]) if len(df) >= 6 else 0.0
                )
                mom_20 = (
                    float((df["Close"].pct_change(20)).iloc[-1])
                    if len(df) >= 21
                    else 0.0
                )
                metrics[ticker] = {
                    "last_close": last_close,
                    "mean_return": mean_ret,
                    "volatility": vol,
                    "momentum_5": mom_5,
                    "momentum_20": mom_20,
                }
                line = (
                    f"- {ticker}: цена={last_close:.2f}, μ={mean_ret*100:.2f}%/день, σ={vol*100:.2f}%/день, "
                    f"моментум5={mom_5*100:.2f}%, моментум20={mom_20*100:.2f}%"
                )
                lines.append(line)
                if mom_5 > 0 and mom_20 > 0:
                    bullets.append(f"{ticker}: восходящий импульс (5/20).")
                elif mom_5 < 0 and mom_20 < 0:
                    bullets.append(f"{ticker}: нисходящий импульс (5/20).")

                # Optional chart
                try:
                    import matplotlib.pyplot as plt  # type: ignore

                    fig, ax = plt.subplots()
                    df["Close"].plot(ax=ax, title=f"{ticker} Close")
                    buf = BytesIO()
                    fig.savefig(buf, format="png")
                    plt.close(fig)
                    charts.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
                except Exception:
                    pass
            except Exception as exc:  # pragma: no cover
                bullets.append(f"{ticker}: ошибка загрузки данных: {exc}")

        if not bullets:
            bullets.append("Сильных сигналов не обнаружено по текущим метрикам.")

        lines.append("")
        lines.append("Резюме:")
        for b in bullets[:10]:
            lines.append(f"- {b}")

        return AnalysisResult(
            "\n".join(lines), bullets, metrics, charts, advanced=False
        )

    # --------------------------- FinRL analysis ----------------------------
    def _advanced_analysis(
        self, tickers: List[str], period: str, interval: str
    ) -> AnalysisResult:
        """Run FinRL analysis if the library is available."""
        from finrl.agents.stablebaselines3.models import DRLAgent  # type: ignore
        from finrl.config import INDICATORS  # type: ignore
        from finrl.meta.env_stock_trading.env_stocktrading import (  # type: ignore[import-not-found]
            StockTradingEnv,
        )
        from finrl.meta.preprocessor.data_processor import DataProcessor  # type: ignore

        bullets: List[str] = []
        metrics: Dict[str, Any] = {}
        charts: List[str] = []

        # Roughly convert ``period`` to days
        days_map = {
            "1mo": 30,
            "3mo": 90,
            "6mo": 180,
            "1y": 365,
            "2y": 730,
            "5y": 1825,
        }
        days = days_map.get(period, 365)
        end_date = dt.date.today()
        start_date = end_date - dt.timedelta(days=days)
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        dp = DataProcessor(
            data_source="yahoo",
            start_date=start_str,
            end_date=end_str,
            time_interval=interval,
            ticker_list=tickers,
            technical_indicator_list=INDICATORS,
            if_vix=True,
        )
        data = dp.download_data()
        data = dp.clean_data(data)
        data = dp.add_technical_indicator(data)
        data = dp.add_turbulence(data)

        stock_dim = len(tickers)
        env_kwargs = {
            "hmax": 100,
            "initial_amount": 1000000,
            "buy_cost_pct": 0.001,
            "sell_cost_pct": 0.001,
            "state_space": 1 + len(INDICATORS) * stock_dim + 2 * stock_dim,
            "stock_dim": stock_dim,
            "tech_indicator_list": INDICATORS,
            "action_space": stock_dim,
            "reward_scaling": 1e-4,
        }
        e_train_gym = StockTradingEnv(df=data, **env_kwargs)
        env_train, _ = e_train_gym.get_sb_env()

        agent = DRLAgent(env=env_train)
        model = agent.get_model("ppo")
        trained_model = agent.train_model(model=model, total_timesteps=100)
        account_value, _ = agent.DRL_prediction(
            model=trained_model, environment=env_train
        )
        final_value = float(account_value["account_value"].iloc[-1])
        metrics["final_value"] = final_value
        bullets.append(f"Финальная стоимость портфеля: {final_value:.2f}")

        # Account value chart
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
            account_value["account_value"].plot(ax=ax, title="Account Value")
            buf = BytesIO()
            fig.savefig(buf, format="png")
            plt.close(fig)
            charts.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
        except Exception:
            pass

        lines = [
            "**Финансовый обзор (FinRL)**",
            "",
            "Тикеры: " + ", ".join(tickers),
            "",
            f"Финальная стоимость портфеля: {final_value:.2f}",
        ]

        return AnalysisResult("\n".join(lines), bullets, metrics, charts, advanced=True)
