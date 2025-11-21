#!/usr/bin/env python3
"""
Константы и доменные эвристики для session_summarizer
"""

import re

SESSION_SUMMARY_VERSION = "1.0.0"

# --- Доменные эвристики ---

CRYPTO_TICKERS = {
    "BTC",
    "ETH",
    "TON",
    "SOL",
    "BNB",
    "XRP",
    "ADA",
    "DOGE",
    "TRX",
    "MATIC",
    "DOT",
    "AVAX",
    "LTC",
    "USDT",
    "USDC",
}

CRYPTO_TERMS = {
    "sec",
    "lawsuit",
    "exploit",
    "airdrop",
    "staking",
    "chain halt",
    "mainnet",
    "testnet",
    "governance",
    "token",
    "ledger",
    "defi",
    "dex",
}

CRYPTO_EXCHANGES = {
    "binance",
    "okx",
    "okex",
    "bybit",
    "coinbase",
    "kraken",
    "kucoin",
    "bitfinex",
    "huobi",
    "gate.io",
    "mexc",
    "bitget",
}

SCI_TECH_TERMS = {
    "arxiv",
    "preprint",
    "paper",
    "dataset",
    "benchmark",
    "sota",
    "research",
    "doi",
    "publication",
    "peer review",
    "open source",
}

SCI_TECH_PATTERNS = [
    re.compile(r"arxiv:\s*\d{4}\.\d{4,5}", re.IGNORECASE),
    re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE),
]

GEOPOLITICS_PATTERNS = [
    (re.compile(r"росси", re.IGNORECASE), "Russia"),
    (re.compile(r"украин", re.IGNORECASE), "Ukraine"),
    (re.compile(r"кита", re.IGNORECASE), "China"),
    (re.compile(r"сша|usa|u\.s\.", re.IGNORECASE), "USA"),
    (
        re.compile(r"евросоюз|европейский союз|european union|\bEU\b", re.IGNORECASE),
        "EU",
    ),
    (re.compile(r"\bUN\b|организац.. объединенных нац", re.IGNORECASE), "UN"),
    (re.compile(r"\bNATO\b", re.IGNORECASE), "NATO"),
    (re.compile(r"санкц", re.IGNORECASE), "Sanctions"),
    (re.compile(r"министерств|ministry", re.IGNORECASE), "Ministry"),
    (
        re.compile(r"пресс(-| )?служб|press-service", re.IGNORECASE),
        "Official statement",
    ),
    (re.compile(r"правительств", re.IGNORECASE), "Government"),
    (re.compile(r"парламент", re.IGNORECASE), "Parliament"),
]

