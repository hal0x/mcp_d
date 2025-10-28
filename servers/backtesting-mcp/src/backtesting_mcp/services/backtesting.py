"""Backtesting service logic without MCP dependencies."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import optuna
import pandas as pd
from optuna.pruners import MedianPruner, NopPruner, PercentilePruner
from optuna.samplers import CmaEsSampler, GridSampler, RandomSampler, TPESampler

from ..datasource import MarketDataSource


@dataclass
class BacktestResult:
    """Result of a backtest run."""
    
    strategy: str
    symbol: str
    timeframe: str
    start: datetime
    end: datetime
    params: Dict[str, float]
    total_return: float
    annualized_return: float
    sharpe: float
    max_drawdown: float
    trades: int
    equity_curve: List[Dict[str, float]]
    extra_metrics: Dict[str, float]
    trades_log: List[Dict[str, Any]]


class BacktestingService:
    """Service for running backtests and optimizations."""
    
    def __init__(
        self,
        data_source: MarketDataSource,
        optuna_storage_url: str | None = None,
        *,
        sampler_name: str = "TPE",
        pruner_name: str = "median",
        n_jobs: int = 1,
    ):
        self.data_source = data_source
        self.optuna_storage_url = optuna_storage_url
        self._sampler_name = sampler_name.upper()
        self._pruner_name = pruner_name.lower()
        self._n_jobs = max(1, n_jobs)
    
    
    def _moving_average_signal(self, series: pd.Series, fast: int, slow: int) -> pd.Series:
        """Generate moving average crossover signals."""
        fast_ma = series.rolling(window=fast).mean()
        slow_ma = series.rolling(window=slow).mean()
        signal = pd.Series(0, index=series.index)
        signal[fast_ma > slow_ma] = 1
        signal[fast_ma < slow_ma] = -1
        return signal.shift(1).fillna(0)
    
    def _simulate_trades(self, prices: pd.Series, signal: pd.Series) -> pd.DataFrame:
        """Simulate trading based on signals."""
        returns = prices.pct_change().fillna(0)
        strategy_returns = returns * signal
        equity = (strategy_returns + 1).cumprod()
        cummax = equity.cummax()
        drawdown = equity / cummax - 1
        return pd.DataFrame(
            {
                "price": prices,
                "position": signal,
                "returns": returns,
                "strategy_returns": strategy_returns,
                "equity": equity,
                "drawdown": drawdown,
            }
        )
    
    def _extract_trades(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Extract individual trades from simulation frame."""
        if frame.empty:
            return pd.DataFrame()
        
        position_changes = frame["position"].diff().abs() > 0
        trade_starts = frame[position_changes].index
        
        trades = []
        for start_idx in trade_starts:
            start_pos = frame.index.get_loc(start_idx)
            if start_pos >= len(frame) - 1:
                continue
                
            position = frame.loc[start_idx, "position"]
            if position == 0:
                continue
                
            # Find end of trade
            end_idx = None
            start_pos = frame.index.get_loc(start_idx)
            for i in range(start_pos + 1, len(frame)):
                if frame.iloc[i]["position"] != position:
                    end_idx = frame.index[i]
                    break
            
            if end_idx is None:
                end_idx = frame.index[-1]
            
            trade_data = frame.loc[start_idx:end_idx]
            if len(trade_data) < 2:
                continue
                
            entry_price = trade_data.iloc[0]["price"]
            exit_price = trade_data.iloc[-1]["price"]
            trade_return = (exit_price / entry_price - 1) * position
            
            trades.append({
                "entry_time": start_idx.isoformat(),
                "exit_time": end_idx.isoformat(),
                "entry_price": float(entry_price),
                "exit_price": float(exit_price),
                "position": int(position),
                "return": float(trade_return),
                "duration_bars": len(trade_data),
            })
        
        return pd.DataFrame(trades)
    
    def _summarize_metrics(self, trades_df: pd.DataFrame, frame: pd.DataFrame) -> Dict[str, float]:
        """Calculate additional trading metrics."""
        if trades_df.empty:
            return {
                "winrate": 0.0,
                "profit_factor": 0.0,
                "avg_trade_return": 0.0,
                "volatility": float(frame["strategy_returns"].std()),
            }
        
        winning_trades = trades_df[trades_df["return"] > 0]
        losing_trades = trades_df[trades_df["return"] < 0]
        
        winrate = len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0.0
        
        gross_profit = winning_trades["return"].sum() if not winning_trades.empty else 0.0
        gross_loss = abs(losing_trades["return"].sum()) if not losing_trades.empty else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
        
        avg_trade_return = trades_df["return"].mean() if not trades_df.empty else 0.0
        
        return {
            "winrate": float(winrate),
            "profit_factor": float(profit_factor),
            "avg_trade_return": float(avg_trade_return),
            "volatility": float(frame["strategy_returns"].std()),
        }
    
    def run_backtest(
        self,
        *,
        strategy: str,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        parameters: Dict[str, float],
    ) -> BacktestResult:
        """Run a single backtest."""
        # Get real market data
        candles = self.data_source.fetch_candles(
            symbol=symbol, timeframe=timeframe, start=start, end=end
        )
        
        if candles is None or candles.empty:
            raise ValueError(f"No market data available for {symbol} {timeframe} from {start} to {end}")
        
        candles = candles.clip(lower=1e-9)
        series = candles["close"]
        
        # Extract strategy parameters
        fast = int(parameters.get("fast", 10))
        slow = int(parameters.get("slow", 30))
        fast = max(2, fast)
        slow = max(fast + 1, slow)
        
        # Generate signals and simulate trades
        signal = self._moving_average_signal(series, fast=fast, slow=slow)
        frame = self._simulate_trades(series, signal)
        
        # Calculate metrics
        total_return = frame["equity"].iloc[-1] - 1
        years = max((end - start).days / 365, 1 / 365)
        annualized_return = (frame["equity"].iloc[-1]) ** (1 / years) - 1
        sharpe = frame["strategy_returns"].mean() / (frame["strategy_returns"].std() + 1e-9) * math.sqrt(252)
        max_drawdown = frame["drawdown"].min()
        trades = int((frame["position"].diff().abs() > 0).sum())
        
        # Extract trades and calculate additional metrics
        trades_df = self._extract_trades(frame)
        extra_metrics = self._summarize_metrics(trades_df, frame)
        
        # Prepare equity curve
        equity_curve = [
            {
                "timestamp": ts.isoformat(),
                "equity": float(value),
            }
            for ts, value in frame["equity"].items()
        ]
        
        trades_log = trades_df.to_dict(orient="records") if not trades_df.empty else []
        
        return BacktestResult(
            strategy=strategy,
            symbol=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
            params={"fast": fast, "slow": slow},
            total_return=float(total_return),
            annualized_return=float(annualized_return),
            sharpe=float(sharpe),
            max_drawdown=float(max_drawdown),
            trades=trades,
            equity_curve=equity_curve,
            extra_metrics=extra_metrics,
            trades_log=trades_log,
        )
    
    def optimize_parameters(
        self,
        *,
        strategy: str,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        parameter_space: Dict[str, Iterable[int]],
        objective: str,
        trials: int,
    ) -> Dict[str, Any]:
        """Optimize strategy parameters using Optuna."""
        if trials < 1:
            raise ValueError("Number of trials must be at least 1.")
        objective = objective.lower()
        if objective not in {"return", "sharpe"}:
            raise ValueError("Objective must be either 'return' or 'sharpe'.")
        
        normalized_space = self._normalize_parameter_space(parameter_space)
        sampler = self._create_sampler(normalized_space)
        pruner = self._create_pruner()
        
        study_name = f"{strategy}_{symbol}_{timeframe}_{start.date()}_{end.date()}"
        create_study_kwargs: Dict[str, Any] = {
            "study_name": study_name,
            "direction": "maximize",
            "sampler": sampler,
            "pruner": pruner,
        }
        if self.optuna_storage_url:
            create_study_kwargs["storage"] = self.optuna_storage_url
            create_study_kwargs["load_if_exists"] = True
        
        study = optuna.create_study(**create_study_kwargs)
        
        def objective_fn(trial: optuna.Trial) -> float:
            fast = self._suggest_parameter(trial, "fast", normalized_space["fast"])
            slow = self._suggest_parameter(trial, "slow", normalized_space["slow"])
            
            try:
                result = self.run_backtest(
                    strategy=strategy,
                    symbol=symbol,
                    timeframe=timeframe,
                    start=start,
                    end=end,
                    parameters={"fast": fast, "slow": slow},
                )
            except Exception as exc:
                trial.set_user_attr("error", str(exc))
                raise
            
            metrics = {
                "total_return": result.total_return,
                "sharpe": result.sharpe,
                "max_drawdown": result.max_drawdown,
                "trades": result.trades,
            }
            for key, value in metrics.items():
                trial.set_user_attr(key, float(value))
            
            actual_params = result.params
            trial.set_user_attr("resolved_fast", actual_params.get("fast"))
            trial.set_user_attr("resolved_slow", actual_params.get("slow"))
            
            if objective == "sharpe":
                return result.extra_metrics.get("sharpe", result.sharpe)
            return result.total_return
        
        study.optimize(
            objective_fn,
            n_trials=trials,
            n_jobs=self._n_jobs,
            catch=(Exception,),
        )
        
        evaluations = self._build_evaluations(study)
        try:
            best_trial = study.best_trial
        except ValueError:
            best_trial = None
        best_params = best_trial.params if best_trial is not None else {}
        best_score = best_trial.value if best_trial is not None else None
        
        return {
            "strategy": strategy,
            "symbol": symbol,
            "timeframe": timeframe,
            "objective": objective,
            "best_params": best_params,
            "best_score": best_score,
            "trials_completed": len(study.trials),
            "evaluations": evaluations,
            "optuna": self._build_optuna_metadata(study, best_trial),
        }

    def _normalize_parameter_space(
        self, parameter_space: Dict[str, Iterable[int]]
    ) -> Dict[str, Dict[str, Any]]:
        """Normalize parameter space to metadata required for Optuna samplers."""
        # Use default parameter space if None or empty
        defaults = {"fast": [5, 10, 15], "slow": [20, 30, 40]}
        if not parameter_space:
            parameter_space = defaults
        
        normalized: Dict[str, Dict[str, Any]] = {}
        for key in ("fast", "slow"):
            values = parameter_space.get(key)
            # Use default values if key is missing or has empty values
            if not values:
                values = defaults[key]
            numeric = sorted({int(value) for value in values})
            if not numeric:
                raise ValueError(f"Invalid '{key}' parameter space for optimization.")
            step = self._infer_step(numeric)
            normalized[key] = {
                "values": numeric,
                "low": numeric[0],
                "high": numeric[-1],
                "step": step,
            }
        return normalized

    def _create_sampler(self, parameter_space: Dict[str, Dict[str, Any]]):
        """Create Optuna sampler based on configured sampler name."""
        name = self._sampler_name
        if name == "RANDOM":
            return RandomSampler()
        if name == "CMAES":
            return CmaEsSampler()
        if name == "GRID":
            search_space = {
                key: spec["values"]
                for key, spec in parameter_space.items()
            }
            return GridSampler(search_space)
        return TPESampler()

    def _create_pruner(self):
        """Create Optuna pruner based on configured pruner name."""
        name = self._pruner_name
        if name == "percentile":
            return PercentilePruner()
        if name == "none":
            return NopPruner()
        return MedianPruner()

    def _build_evaluations(self, study: optuna.study.Study) -> List[Dict[str, Any]]:
        """Build backward-compatible evaluations payload."""
        evaluations: List[Dict[str, Any]] = []
        for trial in study.get_trials(deepcopy=False):
            evaluations.append(
                {
                    "trial_number": trial.number,
                    "fast": trial.params.get("fast"),
                    "slow": trial.params.get("slow"),
                    "score": float(trial.value) if trial.value is not None else None,
                    "total_return": trial.user_attrs.get("total_return"),
                    "sharpe": trial.user_attrs.get("sharpe"),
                    "max_drawdown": trial.user_attrs.get("max_drawdown"),
                    "trades": trial.user_attrs.get("trades"),
                    "state": trial.state.name,
                    "error": trial.user_attrs.get("error"),
                }
            )
        return evaluations

    def _build_optuna_metadata(
        self,
        study: optuna.study.Study,
        best_trial: Optional[optuna.trial.FrozenTrial],
    ) -> Dict[str, Any]:
        """Build Optuna metadata block for response payload."""
        trials = study.get_trials(deepcopy=False)
        datetime_start = (
            trials[0].datetime_start.isoformat()
            if trials and trials[0].datetime_start
            else None
        )
        datetime_complete = (
            trials[-1].datetime_complete.isoformat()
            if trials and trials[-1].datetime_complete
            else None
        )
        return {
            "study_name": study.study_name,
            "sampler": self._sampler_name,
            "pruner": self._pruner_name,
            "storage": self.optuna_storage_url,
            "n_trials": len(trials),
            "best_trial_number": best_trial.number if best_trial is not None else None,
            "datetime_start": datetime_start,
            "datetime_complete": datetime_complete,
        }

    @staticmethod
    def _infer_step(values: List[int]) -> Optional[int]:
        """Infer arithmetic step for a sorted list of integers."""
        if len(values) <= 1:
            return 1
        step = values[1] - values[0]
        if step <= 0:
            return None
        expected = list(range(values[0], values[-1] + step, step))
        if expected != values:
            return None
        return step

    def _suggest_parameter(
        self,
        trial: optuna.Trial,
        name: str,
        spec: Dict[str, Any],
    ) -> int:
        """Suggest parameter value honoring configuration and sampler constraints."""
        step = spec["step"]
        if step is not None:
            return int(trial.suggest_int(name, spec["low"], spec["high"], step=step))
        if self._sampler_name == "CMAES":
            raise ValueError(
                f"Sampler '{self._sampler_name}' requires evenly spaced values for '{name}'."
            )
        return int(trial.suggest_categorical(name, spec["values"]))
    
    def compare_strategies(self, results: Sequence[BacktestResult]) -> Dict[str, Any]:
        """Compare multiple backtest results."""
        sorted_results = sorted(results, key=lambda res: res.total_return, reverse=True)
        leaderboard = [
            {
                "rank": idx + 1,
                "strategy": res.strategy,
                "symbol": res.symbol,
                "timeframe": res.timeframe,
                "total_return": res.total_return,
                "sharpe": res.sharpe,
                "max_drawdown": res.max_drawdown,
                "trades": res.trades,
            }
            for idx, res in enumerate(sorted_results)
        ]
        return {
            "leaderboard": leaderboard,
            "winner": leaderboard[0] if leaderboard else None,
            "comparison_metrics": ["total_return", "sharpe", "max_drawdown"],
            "total_strategies": len(leaderboard),
        }
