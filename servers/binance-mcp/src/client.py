"""Binance клиент."""

import logging
from functools import lru_cache

from binance.client import Client

from .config import get_config

logger = logging.getLogger(__name__)


@lru_cache
def get_binance_client() -> Client:
    """Создает и возвращает настроенный Binance клиент."""
    config = get_config()

    if config.is_demo_mode:
        logger.info("Инициализация демо клиента Binance (futures)")
        logger.info("Используется демо API ключ: %s...", config.effective_api_key[:8])

        # ВАЖНО: python-binance использует testnet только для спотовой торговли (testnet.binance.vision)
        # Для фьючерсной демо торговли нужно переопределить URL на demo-fapi.binance.com
        demo_futures_host = "https://demo-fapi.binance.com"
        Client.FUTURES_TESTNET_URL = f"{demo_futures_host}/fapi"
        Client.FUTURES_DATA_TESTNET_URL = f"{demo_futures_host}/futures/data"
        Client.FUTURES_COIN_TESTNET_URL = "https://demo-dapi.binance.com/dapi"
        Client.FUTURES_COIN_DATA_TESTNET_URL = (
            "https://demo-dapi.binance.com/futures/data"
        )

        logger.info("REST URL: %s", Client.FUTURES_TESTNET_URL)

        return Client(
            config.effective_api_key,
            config.effective_api_secret,
            testnet=True,
        )
    else:
        logger.info("Инициализация live клиента Binance (spot)")
        logger.info("Используется live API ключ: %s...", config.effective_api_key[:8])

        return Client(
            config.effective_api_key, config.effective_api_secret, testnet=False
        )


def get_client_info() -> dict:
    """Возвращает информацию о клиенте."""
    config = get_config()
    
    if config.is_demo_mode:
        return {
            "mode": "DEMO",
            "api_key_prefix": config.effective_api_key[:8] + "...",
        }
    else:
        return {
            "mode": "LIVE",
            "api_key_prefix": config.effective_api_key[:8] + "...",
        }
