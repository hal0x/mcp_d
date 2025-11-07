"""Конфигурация pytest для подавления предупреждений SWIG и заглушек зависимостей."""

import sys
import types
import warnings


class _MetricStub:
    def __init__(self, *args, **kwargs) -> None:  # noqa: D401 - simple stub
        pass

    def labels(self, *args, **kwargs):
        return self

    def observe(self, *args, **kwargs):
        return None

    def inc(self, *args, **kwargs):
        return None

    def set(self, *args, **kwargs):
        return None


if "prometheus_client" not in sys.modules:
    sys.modules["prometheus_client"] = types.SimpleNamespace(
        Counter=_MetricStub,
        Gauge=_MetricStub,
        Histogram=_MetricStub,
    )

# Подавляем предупреждения SWIG от faiss-cpu на уровне pytest
warnings.filterwarnings("ignore", message=".*SwigPyPacked has no __module__ attribute.*", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*SwigPyObject has no __module__ attribute.*", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*swigvarlink has no __module__ attribute.*", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="builtin type SwigPyPacked has no __module__ attribute", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="builtin type SwigPyObject has no __module__ attribute", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="builtin type swigvarlink has no __module__ attribute", category=DeprecationWarning)
