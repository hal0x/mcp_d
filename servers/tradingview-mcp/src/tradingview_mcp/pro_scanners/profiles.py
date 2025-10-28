"""Profile registry for scanner presets."""

from __future__ import annotations

from typing import Iterable

from .config import ProfileConfig, ProfilesConfig, load_profiles_config


class ScannerProfiles:
    """Wrapper that exposes profile presets with validation."""

    def __init__(self, config: ProfilesConfig | None = None):
        self._config = config or load_profiles_config()

    @classmethod
    def from_file(cls, path):
        config = load_profiles_config(path)
        return cls(config)

    def list_profiles(self) -> list[str]:
        return sorted(self._config.profiles.keys())

    def get(self, name: str) -> ProfileConfig:
        return self._config.get(name)

    def ensure_exists(self, names: Iterable[str]) -> None:
        for name in names:
            self.get(name)

    def get_all_profiles(self) -> dict[str, dict]:
        """Get all profiles with their configurations."""
        result = {}
        for name in self.list_profiles():
            profile = self.get(name)
            result[name] = {
                "params": profile.params.model_dump() if hasattr(profile.params, 'model_dump') else profile.params.__dict__,
                "risk": profile.risk.model_dump() if hasattr(profile.risk, 'model_dump') else profile.risk.__dict__,
                "cache_ttl": profile.cache_ttl,
                "description": profile.description
            }
        return result


__all__ = ["ScannerProfiles"]

