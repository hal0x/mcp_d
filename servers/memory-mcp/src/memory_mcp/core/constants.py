"""Константы для HTTP-клиентов и других модулей."""

# Лимиты для графовых запросов
DEFAULT_GRAPH_QUERY_LIMIT = 10000

# Лимиты для поиска
DEFAULT_SEARCH_LIMIT = 50

# Параметры HTTP-коннектора
HTTP_CONNECTOR_LIMIT = 5
HTTP_CONNECTOR_LIMIT_PER_HOST = 2

# Таймауты по умолчанию (в секундах)
DEFAULT_TIMEOUT_CONNECT = 30
DEFAULT_TIMEOUT_TOTAL = 60
DEFAULT_TIMEOUT_MAX_RETRY = 120  # Максимальный таймаут для retry-логики

