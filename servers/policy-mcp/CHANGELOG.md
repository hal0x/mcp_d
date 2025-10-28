# Changelog

Все заметные изменения проекта будут документироваться в этом файле.

Формат основан на [Keep a Changelog](https://keepachangelog.com/ru/1.1.0/), а проект следует семантическому версионированию.

## [0.2.0] - 2025-10-28
### Added
- Асинхронная интеграция с PostgreSQL через SQLAlchemy/asyncpg.
- Автогенерация схемы БД и хранение версий профилей.
- Новые инструменты `get_active_profile` и `activate_profile`.
- Файлы `.env.example`, `AGENTS.md` и базовый changelog.
- REST API `/profiles` для синхронизации с learning-mcp.
- MCP инструменты для версионирования (`list_profile_versions`, `rollback_profile`) и A/B экспериментов (`configure_profile_experiment`).

### Changed
- ProfileService переведён с in-memory хранения на PostgreSQL.
- README обновлён с указанием требований к БД и переменным окружения.

## [0.1.0] - 2025-10-23
### Added
- Начальный каркас Policy MCP с in-memory хранением профилей и MCP инструментами.
