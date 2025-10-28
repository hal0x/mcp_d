"""Benchmarking script for Learning MCP training pipeline."""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from learning_mcp.config import get_settings
from learning_mcp.models import TrainingRequest
from learning_mcp.services.supervisor_client import SupervisorClient
from learning_mcp.services.trainer import TrainerService

BENCHMARK_DIR = Path("servers/learning-mcp/data/benchmarks")


async def run_benchmark(windows: List[str], focus_metrics: List[str], min_samples: int) -> Dict[str, Any]:
    """Execute benchmark runs for provided windows and metrics."""
    supervisor_client = SupervisorClient()
    trainer_service = TrainerService(supervisor_client=supervisor_client)

    results: List[Dict[str, Any]] = []

    try:
        for window in windows:
            for focus_metric in focus_metrics:
                request = TrainingRequest(window=window, min_samples=min_samples, focus_metric=focus_metric)
                try:
                    training_result = await trainer_service.train_offline(request)
                    results.append(
                        {
                            "window": window,
                            "focus_metric": focus_metric,
                            "profile_id": training_result.profile.profile_id,
                            "samples_used": training_result.samples_used,
                            "training_duration": training_result.training_duration,
                            "validation_score": training_result.validation_score,
                            "cv_scores": training_result.cv_scores,
                            "best_params": training_result.best_params,
                            "feature_importance": training_result.feature_importance,
                            "permutation_importance": training_result.permutation_importance,
                            "timestamp": datetime.utcnow().isoformat(),
                        }
                    )
                except Exception as exc:  # pragma: no cover - benchmarking diagnostics
                    results.append(
                        {
                            "window": window,
                            "focus_metric": focus_metric,
                            "error": str(exc),
                            "timestamp": datetime.utcnow().isoformat(),
                        }
                    )
    finally:
        await trainer_service.close()
        await supervisor_client.close()

    return {
        "executed_at": datetime.utcnow().isoformat(),
        "windows": windows,
        "focus_metrics": focus_metrics,
        "run_count": len(results),
        "results": results,
    }


def save_results(payload: Dict[str, Any]) -> Path:
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    filename = BENCHMARK_DIR / f"benchmark_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    filename.write_text(json.dumps(payload, indent=2))
    return filename


async def async_main(args: argparse.Namespace) -> None:
    settings = get_settings()
    windows = args.windows or sorted(set(settings.available_windows + [settings.online_learning_window]))
    metrics = args.metrics or settings.available_metrics
    min_samples = args.min_samples or settings.min_samples

    payload = await run_benchmark(windows, metrics, min_samples)
    path = save_results(payload)
    print(f"Benchmark results stored at {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Learning MCP benchmarking tool")
    parser.add_argument("--windows", nargs="*", help="Windows to benchmark, e.g. 7d 30d 1d")
    parser.add_argument("--metrics", nargs="*", help="Focus metrics to evaluate")
    parser.add_argument("--min-samples", type=int, help="Minimum samples threshold per run")
    args = parser.parse_args()

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
