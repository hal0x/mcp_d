"""Policy engine loading YAML tool policies."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

import yaml

from tools import Tool

if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    from executor import ToolPolicy


class PolicyEngine:
    """Load and provide execution policies for tools."""

    def __init__(self, path: str | Path = "config/policies.yaml") -> None:
        self.path = Path(path)
        self._policies: dict[Tool, dict[str, Any]] = {}
        self._load()

    def _coerce_float(self, v: Any, *, positive: bool = False, max_value: float | None = None) -> float | None:
        try:
            val = float(v)
        except Exception:
            return None
        if positive and val <= 0:
            return None
        if max_value is not None and val > max_value:
            return None
        return val

    def _coerce_int(self, v: Any, *, positive: bool = False) -> int | None:
        try:
            val = int(v)
        except Exception:
            return None
        if positive and val <= 0:
            return None
        return val

    def _sanitize(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """Filter unknown fields and coerce supported ones.

        Unknown fields and nested structures are ignored. Invalid values are
        dropped so that defaults from ToolPolicy will apply.
        """
        allowed: set[str] = {
            "max_wall_time_s",
            "max_mem_mb",
            "cpu_quota",
            "network_mode",
            "network_proxy",
            "userns",
            "cap_drop",
            "seccomp_profile",
            "apparmor_profile",
        }
        clean: dict[str, Any] = {}
        for key, val in params.items():
            if key not in allowed:
                continue
            # Ignore nested structures
            if isinstance(val, (dict, list)):
                continue
            if key == "max_wall_time_s":
                v = self._coerce_float(val, positive=True)
                if v is not None:
                    clean[key] = v
                continue
            if key == "max_mem_mb":
                v = self._coerce_int(val, positive=True)
                if v is not None:
                    clean[key] = v
                continue
            if key == "cpu_quota":
                v = self._coerce_float(val, positive=True, max_value=1.0)
                if v is not None:
                    clean[key] = v
                continue
            if key == "network_mode":
                mode = str(val).strip().lower()
                if mode in {"none", "host", "bridge"}:
                    clean[key] = mode
                else:
                    clean[key] = "none"
                continue
            if key in {"cap_drop", "network_proxy", "userns", "seccomp_profile", "apparmor_profile"}:
                sval = str(val).strip()
                if sval:
                    clean[key] = sval
                continue
        return clean

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            with self.path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception:
            # Malformed YAML: treat as empty policies
            data = {}
        if not isinstance(data, Mapping):
            return
        for name, params in data.items():
            try:
                tool = Tool(name)
            except ValueError:
                continue
            if isinstance(params, Mapping):
                self._policies[tool] = self._sanitize(params)

    def get_policy(self, tool: Tool) -> "ToolPolicy":
        """Return policy for ``tool`` or default policy."""

        from executor import ToolPolicy

        params: dict[str, Any] = self._policies.get(tool, {})
        return ToolPolicy(**params)


__all__ = ["PolicyEngine"]
