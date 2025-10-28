"""Configuration loaders for professional scanners."""

from __future__ import annotations

import os
import tempfile
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Iterable, Optional


import yaml
from pydantic import BaseModel, Field, PositiveInt, field_validator


_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CONFIG_DIR = _PACKAGE_ROOT / "configs"
_RESOURCE_CACHE_DIR = Path(tempfile.gettempdir()) / "tradingview_mcp_configs"


def resolve_config_path(filename: str, override: Optional[Path] = None) -> Path:
    if override:
        return override

    env_dir = os.getenv("TRADINGVIEW_MCP_CONFIG_DIR")
    search_dirs = []
    if env_dir:
        search_dirs.append(Path(env_dir))

    search_dirs.append(_DEFAULT_CONFIG_DIR)

    project_root_dir = Path(__file__).resolve().parents[3] / "configs"
    if project_root_dir not in search_dirs:
        search_dirs.append(project_root_dir)

    for base_dir in search_dirs:
        candidate = base_dir / filename
        if candidate.exists():
            return candidate

    try:
        resource = resources.files("tradingview_mcp").joinpath("configs", filename)
        data: bytes | None = None
        resolved_path: Path | None = None
        with resources.as_file(resource) as resource_path:
            resolved_path = Path(resource_path)
            if resolved_path.exists():
                try:
                    data = resolved_path.read_bytes()
                except OSError:
                    data = None
        if resolved_path and resolved_path.exists():
            return resolved_path
        if data is None:
            data = resource.read_bytes()
        _RESOURCE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = _RESOURCE_CACHE_DIR / filename
        cache_path.write_bytes(data)
        return cache_path
    except (FileNotFoundError, ModuleNotFoundError):
        pass

    searched = ", ".join(str(path) for path in search_dirs)
    raise FileNotFoundError(f"Unable to locate '{filename}'. Checked: {searched}")


class ScannerParams(BaseModel):
    ema_fast: PositiveInt = Field(..., description="Fast EMA period")
    ema_slow: PositiveInt = Field(..., description="Slow EMA period")
    adx_min: float = Field(..., ge=0, description="ADX minimum threshold")
    vol_z_min: float = Field(..., ge=0, description="Volume z-score threshold")

    @field_validator("ema_slow")
    @classmethod
    def validate_periods(cls, ema_slow: int, info):
        ema_fast = info.data.get("ema_fast")
        if ema_fast and ema_slow <= ema_fast:
            raise ValueError("ema_slow must be greater than ema_fast")
        return ema_slow


class RiskSettings(BaseModel):
    atr_stop_mult: float = Field(..., gt=0)
    leverage_bounds: tuple[int, int] = Field(..., description="Min/Max leverage")
    time_stop_bars: PositiveInt = Field(..., description="Bars before time stop")

    @field_validator("leverage_bounds")
    @classmethod
    def validate_leverage(cls, bounds: Iterable[int]):
        values = list(bounds)
        if len(values) != 2:
            raise ValueError("leverage_bounds must contain exactly two integers")
        low, high = values
        if low <= 0 or high <= 0 or high < low:
            raise ValueError("Invalid leverage bounds")
        return low, high


class ProfileConfig(BaseModel):
    params: ScannerParams
    risk: RiskSettings
    cache_ttl: PositiveInt = Field(default=300, description="Cache TTL in seconds")
    description: str = Field(default="No description available", description="Profile description")


class ProfilesConfig(BaseModel):
    profiles: dict[str, ProfileConfig]

    def get(self, name: str) -> ProfileConfig:
        try:
            return self.profiles[name]
        except KeyError as exc:
            raise KeyError(f"Unknown profile '{name}'") from exc


class RedisTTLConfig(BaseModel):
    indicators: PositiveInt = Field(default=300)
    alerts: PositiveInt = Field(default=1800)
    sessions: PositiveInt = Field(default=3600)


class RedisConfig(BaseModel):
    host: str = Field(default="localhost")
    port: PositiveInt = Field(default=6379)
    db: int = Field(default=0, ge=0)
    ttl: RedisTTLConfig = Field(default_factory=RedisTTLConfig)


class PostgresConfig(BaseModel):
    host: str = Field(default="localhost")
    port: PositiveInt = Field(default=5432)
    database: str = Field(default="tradingview_scanners")
    user: str | None = None
    password: str | None = None
    pool_size: PositiveInt = Field(default=10)


class HTTPServiceConfig(BaseModel):
    url: str
    timeout: PositiveInt = Field(default=30)



class DerivativesFilterConfig(BaseModel):
    min_open_interest: float | None = Field(default=None)
    max_funding_abs: float = Field(default=0.05, ge=0)
    min_cvd_abs: float | None = Field(default=None)


class ContextFilterConfig(BaseModel):
    min_confidence: int = Field(default=55, ge=0)
    max_leverage: float = Field(default=20.0, ge=0)
    require_stop_loss: bool = Field(default=True)


class FiltersConfig(BaseModel):
    derivatives: DerivativesFilterConfig = Field(default_factory=DerivativesFilterConfig)
    context: ContextFilterConfig = Field(default_factory=ContextFilterConfig)


class InfrastructureConfig(BaseModel):
    redis: RedisConfig
    postgres: PostgresConfig
    binance_mcp: HTTPServiceConfig
    halv1_bot: HTTPServiceConfig
    memory_mcp: HTTPServiceConfig | None = None
    filters: FiltersConfig | None = None


@lru_cache
def load_profiles_config(path: Optional[Path] = None) -> ProfilesConfig:
    filepath = resolve_config_path("scanner_profiles.yaml", path)
    data = yaml.safe_load(filepath.read_text("utf-8"))
    return ProfilesConfig.model_validate(data)


@lru_cache
def load_infrastructure_config(path: Optional[Path] = None) -> InfrastructureConfig:
    filepath = resolve_config_path("infrastructure.yaml", path)
    data = yaml.safe_load(filepath.read_text("utf-8"))
    redis_cfg = data.get("redis", {}) or {}
    host_override = (
        os.getenv("TRADINGVIEW_REDIS_HOST")
        or os.getenv("REDIS_HOST")
        or os.getenv("UPSTASH_REDIS_HOST")
    )
    port_override = (
        os.getenv("TRADINGVIEW_REDIS_PORT")
        or os.getenv("REDIS_PORT")
        or os.getenv("UPSTASH_REDIS_PORT")
    )
    db_override = os.getenv("TRADINGVIEW_REDIS_DB") or os.getenv("REDIS_DB")
    if host_override:
        redis_cfg["host"] = host_override
    if port_override:
        try:
            redis_cfg["port"] = int(port_override)
        except ValueError:
            pass
    if db_override:
        try:
            redis_cfg["db"] = int(db_override)
        except ValueError:
            pass
    data["redis"] = redis_cfg

    postgres_cfg = data.get("postgres", {}) or {}
    pg_host = os.getenv("TRADINGVIEW_POSTGRES_HOST") or os.getenv("POSTGRES_HOST")
    pg_port = os.getenv("TRADINGVIEW_POSTGRES_PORT") or os.getenv("POSTGRES_PORT")
    pg_db = os.getenv("TRADINGVIEW_POSTGRES_DATABASE") or os.getenv("POSTGRES_DB")
    pg_user = os.getenv("TRADINGVIEW_POSTGRES_USER") or os.getenv("POSTGRES_USER")
    pg_password = os.getenv("TRADINGVIEW_POSTGRES_PASSWORD") or os.getenv("POSTGRES_PASSWORD")
    if pg_host:
        postgres_cfg["host"] = pg_host
    if pg_port:
        try:
            postgres_cfg["port"] = int(pg_port)
        except ValueError:
            pass
    if pg_db:
        postgres_cfg["database"] = pg_db
    if pg_user:
        postgres_cfg["user"] = pg_user
    if pg_password:
        postgres_cfg["password"] = pg_password
    data["postgres"] = postgres_cfg
    
    # Поддержка переменной окружения для binance-mcp URL
    binance_mcp_cfg = data.get("binance_mcp", {}) or {}
    binance_url = os.getenv("BINANCE_MCP_URL")
    if binance_url:
        binance_mcp_cfg["url"] = binance_url
    binance_timeout = os.getenv("BINANCE_MCP_TIMEOUT")
    if binance_timeout:
        try:
            binance_mcp_cfg["timeout"] = int(binance_timeout)
        except ValueError:
            pass
    data["binance_mcp"] = binance_mcp_cfg
    
    return InfrastructureConfig.model_validate(data)


__all__ = [
    "InfrastructureConfig",
    "ProfilesConfig",
    "ProfileConfig",
    "RedisConfig",
    "PostgresConfig",
    "HTTPServiceConfig",
    "FiltersConfig",
    "DerivativesFilterConfig",
    "ContextFilterConfig",
    "RiskSettings",
    "ScannerParams",
    "load_profiles_config",
    "load_infrastructure_config",
    "resolve_config_path",
]
