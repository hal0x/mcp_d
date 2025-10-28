"""Professional scanners package wiring."""

from __future__ import annotations

from typing import Any

from .config import load_infrastructure_config, load_profiles_config, resolve_config_path
from .profiles import ScannerProfiles

__all__ = [
    "ScannerService",
    "ScannerProfiles",
    "load_profiles_config",
    "load_infrastructure_config",
    "resolve_config_path",
]


def __getattr__(name: str) -> Any:
    if name == "ScannerService":
        from .service import ScannerService

        return ScannerService
    raise AttributeError(name)
