"""Конфигурация приложения."""

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Config:
    """Конфигурация приложения."""

    # API ключи
    api_key: str
    api_secret: str
    demo_api_key: Optional[str]
    demo_api_secret: Optional[str]

    # Настройки режима
    demo_trading: bool

    # Сервер
    host: str
    port: int

    # Telegram уведомления
    telegram_bot_token: Optional[str]
    telegram_chat_id: Optional[str]

    # Redis кэш
    redis_enabled: bool
    redis_url: Optional[str]
    redis_cache_ttl: int

    # PostgreSQL
    postgres_enabled: bool
    postgres_host: str | None
    postgres_port: int
    postgres_database: str | None
    postgres_user: Optional[str]
    postgres_password: Optional[str]

    # Risk settings
    risk_per_trade_pct: float
    taker_fee_pct: float
    slippage_pct: float
    daily_loss_cap_pct: float
    max_consecutive_losses: int
    tp_rr_min: float
    tp_atr_multiple: float

    @property
    def is_demo_mode(self) -> bool:
        """Проверяет, используется ли демо режим."""
        return self.demo_trading


    @property
    def effective_api_key(self) -> str:
        """Возвращает активный API ключ."""
        key = self.demo_api_key if self.is_demo_mode else self.api_key
        if not key:
            raise ValueError("API ключ не установлен")
        return key

    @property
    def effective_api_secret(self) -> str:
        """Возвращает активный API секрет."""
        secret = self.demo_api_secret if self.is_demo_mode else self.api_secret
        if not secret:
            raise ValueError("API секрет не установлен")
        return secret

    @property
    def has_redis(self) -> bool:
        """Проверяет, доступен ли Redis кэш."""
        return self.redis_enabled and bool(self.redis_url)

    @property
    def has_postgres(self) -> bool:
        """Проверяет, доступно ли подключение к PostgreSQL."""
        return (
            self.postgres_enabled
            and bool(self.postgres_host)
            and bool(self.postgres_database)
        )


@lru_cache
def get_config() -> Config:
    """Получает конфигурацию из переменных окружения."""
    # Основные API ключи
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")

    # Демо API ключи
    demo_api_key = os.getenv("DEMO_BINANCE_API_KEY")
    demo_api_secret = os.getenv("DEMO_BINANCE_API_SECRET")

    # Настройки режима
    demo_trading = os.getenv("BINANCE_DEMO_TRADING", "false").lower() in {
        "true",
        "1",
        "yes",
        "on",
    }

    # Сервер
    host = os.getenv("BINANCE_MCP_HOST") or os.getenv("HOST", "0.0.0.0")
    port_str = os.getenv("BINANCE_MCP_PORT") or os.getenv("PORT", "8000")
    port = int(port_str) if port_str else 8000

    # Telegram уведомления
    telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")

    redis_enabled = os.getenv("REDIS_ENABLED", "true").lower() in {
        "true",
        "1",
        "yes",
        "on",
    }
    redis_url = os.getenv("REDIS_URL") if redis_enabled else None
    if redis_enabled and not redis_url:
        redis_url = "redis://localhost:6379/0"
    redis_cache_ttl = int(os.getenv("REDIS_CACHE_TTL", "30"))

    postgres_enabled = os.getenv("POSTGRES_ENABLED", "false").lower() in {
        "true",
        "1",
        "yes",
        "on",
    }
    postgres_host = os.getenv("POSTGRES_HOST")
    postgres_port = int(os.getenv("POSTGRES_PORT", "5432"))
    postgres_database = os.getenv("POSTGRES_DATABASE")
    postgres_user = os.getenv("POSTGRES_USER")
    postgres_password = os.getenv("POSTGRES_PASSWORD")

    # Валидация - API ключи нужны только для приватных операций
    # Для публичных данных (цены, книги ордеров) ключи не обязательны
    # В демо режиме разрешаем использование фиктивных ключей для тестирования

    # Если используются только демо ключи, устанавливаем их как основные
    if (
        demo_trading
        and demo_api_key
        and demo_api_secret
        and (not api_key or not api_secret)
    ):
        api_key = demo_api_key
        api_secret = demo_api_secret

    return Config(
        api_key=api_key or "",
        api_secret=api_secret or "",
        demo_api_key=demo_api_key,
        demo_api_secret=demo_api_secret,
        demo_trading=demo_trading,
        host=str(host),
        port=port,
        telegram_bot_token=telegram_bot_token,
        telegram_chat_id=telegram_chat_id,
        redis_enabled=redis_enabled,
        redis_url=redis_url,
        redis_cache_ttl=redis_cache_ttl,
        postgres_enabled=postgres_enabled,
        postgres_host=postgres_host,
        postgres_port=postgres_port,
        postgres_database=postgres_database,
        postgres_user=postgres_user,
        postgres_password=postgres_password,
        risk_per_trade_pct=float(os.getenv("RISK_PER_TRADE_PCT", "1.0")),
        taker_fee_pct=float(os.getenv("TAKER_FEE_PCT", "0.04")) / 100.0,
        slippage_pct=float(os.getenv("SLIPPAGE_PCT", "0.02")) / 100.0,
        daily_loss_cap_pct=float(os.getenv("DAILY_LOSS_CAP_PCT", "6.0")),
        max_consecutive_losses=int(os.getenv("MAX_CONSECUTIVE_LOSSES", "3")),
        tp_rr_min=float(os.getenv("TP_RR_MIN", "1.0")),
        tp_atr_multiple=float(os.getenv("TP_ATR_MULTIPLE", "1.5")),
    )
