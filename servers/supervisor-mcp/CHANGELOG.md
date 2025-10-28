# Changelog

Все заметные изменения Supervisor MCP фиксируются здесь. Формат основан на [Keep a Changelog](https://keepachangelog.com/ru/1.1.0/).

## [0.2.0] - 2025-10-28
### Added
- Асинхронный слой работы с PostgreSQL (SQLAlchemy async engine, ORM модели).
- Поддержка Redis-кэша для агрегатов метрик.
- REST эндпоинты `/query/agg` и `/query/facts` подключены к persistent-слою.
- Файл `.env.example` с основными переменными окружения.
- Документ `AGENTS.md` с архитектурой и потоками данных.

### Changed
- MetricsService переписан на PostgreSQL/Redis, память более не используется как хранилище.
- README дополнен инструкциями по подключению реальных БД и Redis.

## [0.1.0] - 2025-10-23
### Added
- Начальная версия Supervisor MCP с in-memory хранением метрик/фактов и MCP инструментами.

