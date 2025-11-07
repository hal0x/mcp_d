"""Entry point for the Telegram AI assistant project."""

from __future__ import annotations

import asyncio
import logging
import json as _json
import os
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, cast

import yaml
from dotenv import load_dotenv

# Подавляем предупреждения SWIG от faiss-cpu
import config.warnings_config  # noqa: F401

from agent.commands import handle_command
from agent.coordinator import ModuleCoordinator
from agent.core import AgentCore
from agent.adaptation import AdaptationEngine, set_adaptation_engine
from agent.modules.context_aware_base import ContextAwareEventsModule, ContextAwareThemesModule
from bot.telegram_bot import TelegramBot
from events.models import Event, MessageReceived, ReplyReady
from executor import CodeGenerator, create_executor
# finance module removed - not used
from index.cluster_manager import ClusterManager
from index.insight_store import InsightStore
from index.raw_storage import RawStorage, _sanitize_component
from index.summarizer import Summarizer
from index.telethon_indexer import TelethonIndexer
from index.theme_store import ThemeStore, sanitize_name
from index.vector_index import DEFAULT_MODEL, VectorIndex
from internet import SearchClient
from llm import create_llm_client
from llm.base_client import LLMClient
from llm.prompt_manager import PromptManager
from llm.hot_reload import HotReloadManager
from llm.context_factory import create_context_aware_client, create_context_aware_code_generator, create_context_aware_search_client
from memory import MemoryServiceAdapter
from planner import LLMTaskPlanner
from retriever.retriever import Retriever
from services.chronicle_service import ChronicleService
from services.event_bus import AsyncEventBus
from services.monitoring import MonitoringService, set_monitoring_service, get_monitoring_service
from services.summary_service import SummaryService
from services.telethon_service import TelethonService
from tasks.scheduler import TaskScheduler, schedule_cluster_jobs, schedule_memory_jobs
from tools import ToolRegistry
from utils.performance import log_performance_summary
from web.dashboard.server import app as dashboard_app
import uvicorn

# Настройка логирования в самом начале
log_level_str = os.getenv("LOGLEVEL", "INFO").upper()
log_level = getattr(logging, log_level_str, logging.INFO)
class _ExtraContextFormatter(logging.Formatter):
    """Formatter that appends known extra fields as JSON context.

    Shows important keys when present (topic, id, error, payload, cmd, exit_code, cwd, PATH, stdout, stderr).
    Falls back to plain message if nothing extra is provided.
    """

    _known = {
        "topic",
        "id",
        "error",
        "error_type",
        "event_type",
        "handler",
        "attempts",
        "payload",
        "cmd",
        "exit_code",
        "cwd",
        "PATH",
        "stdout",
        "stderr",
    }

    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        base = super().format(record)
        try:
            data = {}
            for k in self._known:
                if hasattr(record, k):
                    v = getattr(record, k)
                    # Ограничим длину длинных строк
                    if isinstance(v, str) and len(v) > 1000:
                        v = v[:1000] + "..."
                    data[k] = v
            if data:
                return f"{base} | ctx: " + _json.dumps(data, ensure_ascii=False)
        except Exception:
            pass
        return base

# Reconfigure root logger with custom formatter
root = logging.getLogger()
root.setLevel(log_level)
root.handlers.clear()
_handler = logging.StreamHandler()
_handler.setLevel(log_level)
_handler.setFormatter(
    _ExtraContextFormatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
)
root.addHandler(_handler)

# Фильтруем бинарные данные из логов Telethon
class BinaryDataFilter(logging.Filter):
    def filter(self, record):
        if hasattr(record, 'msg'):
            msg = str(record.msg)
            # Проверяем на наличие бинарных данных (непечатаемые символы)
            if any(ord(c) < 32 and c not in '\t\n\r' for c in msg):
                return False
        return True

# Применяем фильтр к Telethon логгерам
telethon_logger = logging.getLogger('telethon')
telethon_logger.addFilter(BinaryDataFilter())
telethon_logger.setLevel(logging.WARNING)  # Уменьшаем уровень логирования Telethon

logger = logging.getLogger(__name__)
# ---------------------------------------------------------------------------

async def setup_broadcast_executor(bot: TelegramBot) -> None:
    executor = await create_broadcast_executor()
    bot.set_broadcast_executor(executor)

# Load environment variables from a .env file if present so that credentials
# defined there (e.g. Telethon API keys) are available via ``os.getenv``.
load_dotenv()


def load_config(path: str = "config/settings.yaml") -> Dict[str, Any]:
    logger.debug(f"🔧 [CONFIG] Загружаем конфигурацию из {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    logger.debug(f"🔧 [CONFIG] Базовая конфигурация загружена: {list(cfg.keys())}")

    # override secrets with environment variables
    telegram_cfg = cfg.setdefault("telegram", {})
    if (bot_token := os.getenv("TELEGRAM_BOT_TOKEN")) is not None:
        telegram_cfg["bot_token"] = bot_token
        # Обрезаем токен для дебаг логов
        debug_token = bot_token[:10] + "..." if len(bot_token) > 10 else bot_token
        logger.debug(f"🔧 [CONFIG] TELEGRAM_BOT_TOKEN найден: {debug_token}")
    else:
        telegram_cfg.setdefault("bot_token", "")
        logger.debug("🔧 [CONFIG] TELEGRAM_BOT_TOKEN не найден, используем пустую строку")

    llm_cfg = cfg.setdefault("llm", {})
    if (llm_key := os.getenv("LLM_API_KEY")) is not None:
        llm_cfg["api_key"] = llm_key
        # Обрезаем API ключ для дебаг логов
        debug_key = llm_key[:10] + "..." if len(llm_key) > 10 else llm_key
        logger.debug(f"🔧 [CONFIG] LLM_API_KEY найден: {debug_key}")
    else:
        llm_cfg.setdefault("api_key", "")
        logger.debug("🔧 [CONFIG] LLM_API_KEY не найден, используем пустую строку")

    emb_cfg = cfg.setdefault("embeddings", {})
    if (emb_key := os.getenv("EMBEDDINGS_API_KEY")) is not None:
        emb_cfg["api_key"] = emb_key
        # Обрезаем API ключ для дебаг логов
        debug_key = emb_key[:10] + "..." if len(emb_key) > 10 else emb_key
        logger.debug(f"🔧 [CONFIG] EMBEDDINGS_API_KEY найден: {debug_key}")
    else:
        emb_cfg.setdefault("api_key", "")
        logger.debug("🔧 [CONFIG] EMBEDDINGS_API_KEY не найден, используем пустую строку")

    tele_cfg = cfg.setdefault("telethon", {})
    if (api_id := os.getenv("TELETHON_API_ID")) is not None:
        tele_cfg["api_id"] = int(api_id)
        logger.debug(f"🔧 [CONFIG] TELETHON_API_ID найден: {api_id}")
    else:
        tele_cfg["api_id"] = int(tele_cfg.get("api_id", 0))
        logger.debug(f"🔧 [CONFIG] TELETHON_API_ID не найден, используем значение из конфига: {tele_cfg['api_id']}")
    if (api_hash := os.getenv("TELETHON_API_HASH")) is not None:
        tele_cfg["api_hash"] = api_hash
        # Обрезаем API хеш для дебаг логов
        debug_hash = api_hash[:10] + "..." if len(api_hash) > 10 else api_hash
        logger.debug(f"🔧 [CONFIG] TELETHON_API_HASH найден: {debug_hash}")
    else:
        tele_cfg.setdefault("api_hash", "")
        logger.debug("🔧 [CONFIG] TELETHON_API_HASH не найден, используем пустую строку")
    if (session := os.getenv("TELETHON_SESSION")) is not None:
        tele_cfg["session"] = session
        logger.debug(f"🔧 [CONFIG] TELETHON_SESSION найден: {session}")
    else:
        tele_cfg.setdefault("session", "user")
        logger.debug("🔧 [CONFIG] TELETHON_SESSION не найден, используем 'user'")

    # Обрезаем длинные значения для дебаг логов
    debug_cfg = {}
    for key, value in cfg.items():
        if isinstance(value, dict):
            debug_cfg[key] = {}
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, str) and len(sub_value) > 30:
                    debug_cfg[key][sub_key] = sub_value[:30] + "..."
                else:
                    debug_cfg[key][sub_key] = sub_value
        elif isinstance(value, str) and len(value) > 30:
            debug_cfg[key] = value[:30] + "..."
        else:
            debug_cfg[key] = value
    
    # Логируем только основные параметры
    logger.debug(f"🔧 [CONFIG] Основные параметры: LLM={llm_cfg.get('provider')}, Telegram={bool(telegram_cfg.get('bot_token'))}, Telethon={bool(tele_cfg.get('api_id'))}")
    return cfg


def _ensure_agent_memory_path(cfg: Dict[str, Any]) -> str:
    """Resolve and prepare the persistent agent memory path from config."""

    paths_cfg = cfg.get("paths", {})
    default_path = Path("db") / "agent_memory.json"
    raw_path = paths_cfg.get("agent_memory", default_path)
    path = Path(raw_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def create_agent_memory(
    cfg: Dict[str, Any],
    llm_client: LLMClient | None,
    *,
    short_term_limit: int = 100,
) -> MemoryServiceAdapter:
    """Build the agent memory store using the configured persistent path."""

    path = _ensure_agent_memory_path(cfg)
    return MemoryServiceAdapter(
        path=path,
        embeddings_client=None,
        short_term_limit=short_term_limit,
        llm_client=llm_client,
    )


async def run_dashboard(cfg: Dict[str, Any]) -> None:
    """Запуск дашборда с метриками."""
    dashboard_cfg = cfg.get("dashboard", {})
    if not dashboard_cfg.get("enabled", False):
        logger.info("📊 [DASHBOARD] Дашборд отключен в конфигурации")
        return
    
    host = dashboard_cfg.get("host", "0.0.0.0")
    port = dashboard_cfg.get("port", 8080)
    
    logger.info(f"📊 [DASHBOARD] Запускаем дашборд на {host}:{port}")
    
    config = uvicorn.Config(
        dashboard_app,
        host=host,
        port=port,
        log_level="info"
    )
    server = uvicorn.Server(config)
    await server.serve()


async def main() -> None:
    logger.info("🚀 [STARTUP] Начинаем запуск приложения")
    logger.info(f"🔧 [LOGGING] Уровень логирования установлен: {log_level_str}")
    
    # Загрузка конфигурации
    logger.info("🔧 [CONFIG] Загружаем конфигурацию приложения")
    config = load_config()
    logger.info("✅ [CONFIG] Конфигурация загружена")
    logger.info("=" * 60)

    # Инициализация Memory Service (embeddings_client будет установлен позже)
    logger.info("🧠 [MEMORY] Инициализируем Memory Service")
    memory_cfg = config.get("memory", {})
    db_path = config.get("paths", {}).get("memory_db", "db/memory.db")
    
    # Создаем директорию для базы данных
    import os
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    memory = MemoryServiceAdapter(
        path=db_path,
        embeddings_client=None,  # Будет установлен позже
        short_term_limit=100,
    )
    logger.info(f"✅ [MEMORY] Memory Service создан с путем: {db_path}")
    logger.info("-" * 40)
    
    # Настройка планировщика задач для памяти
    logger.info("📅 [SCHEDULER] Настраиваем планировщик задач для памяти")
    memory_scheduler = TaskScheduler()
    schedule_memory_jobs(memory_scheduler, memory)
    logger.info("✅ [SCHEDULER] Планировщик задач для памяти настроен")
    logger.info("-" * 40)

    # Настройка планировщика целей
    logger.info("🎯 [GOALS] Настраиваем планировщик целей")
    goal_scheduler = TaskScheduler()
    logger.info("✅ [GOALS] Планировщик целей создан")

    # Настройка исполнителя кода
    logger.info("⚙️ [EXECUTOR] Настраиваем исполнитель кода")
    exec_cfg = config.get("executor", {})
    venv_path = config.get("paths", {}).get("venv", "venv")
    executor = create_executor("docker", venv_path)
    artifact_ttl = int(exec_cfg.get("artifact_ttl", 3600))
    logger.info(f"✅ [EXECUTOR] Исполнитель Docker создан, venv: {venv_path}, TTL: {artifact_ttl}s")
    logger.info("-" * 40)

    # LLM и поиск отключены для фокуса на загрузке Telegram чатов
    logger.info("🚫 [LLM] LLM клиент отключен для фокуса на загрузке Telegram чатов")
    llm_client = None
    context_aware_client = None
    search_client = None
    prompt_manager = None
    hot_reload_manager = None

    # Настройка Event Bus
    logger.info("📡 [EVENTS] Настраиваем Event Bus")
    bus: AsyncEventBus[Event] = AsyncEventBus()

    # Добавляем подписчика на errors чтобы избежать graceful shutdown
    async def error_handler(event: Event) -> None:
        logger.error("❌ [EVENTS] Error event received: %s", event)

    bus.subscribe("errors", error_handler)
    logger.info("✅ [EVENTS] Event Bus настроен с обработчиком ошибок")

    # Инструменты и планировщик отключены
    logger.info("🚫 [TOOLS] Инструменты отключены для фокуса на загрузке Telegram чатов")
    registry = None
    planner = None
    code_generator = None

    # Память агента и модули отключены
    logger.info("🚫 [AGENT_MEMORY] Память агента отключена для фокуса на загрузке Telegram чатов")
    agent_memory = None
    coordinator = None
    context_aware_events_module = None
    context_aware_themes_module = None
    adaptation_engine = None

    # Планировщики и ядро агента отключены
    logger.info("🚫 [TASKS] Планировщики отключены для фокуса на загрузке Telegram чатов")
    memory_task = None
    core = None

    # Настройка целей
    logger.info("🎯 [GOALS] Настраиваем цели")
    goal_chat_id = int(config.get("telegram", {}).get("goal_chat_id", 134432210))
    # Цели можно добавить позже через Telegram бота
    logger.info(f"✅ [GOALS] Планировщик целей готов, chat_id: {goal_chat_id}")
    # Добавляем дефолтную цель, чтобы агент периодически активировался и виделся в тестах
    try:
        goal_scheduler.add_goal(
            bus=bus,
            chat_id=goal_chat_id,
            goal="План: ежедневная сводка",
            interval=86400,
            initial_delay=3600,  # Задержка 1 час вместо немедленного запуска
        )
        logger.info("✅ [GOALS] Добавлена дефолтная цель для chat_id=%s с задержкой 1 час", goal_chat_id)
    except Exception:
        logger.exception("Не удалось добавить дефолтную цель")

    # Создание задачи планировщика целей
    try:
        asyncio.get_running_loop()
        goal_task = asyncio.create_task(goal_scheduler.run())
        logger.info("✅ [TASKS] Задача планировщика целей создана")
    except RuntimeError:
        goal_task = None
        logger.warning("⚠️ [TASKS] Нет работающего event loop, пропускаем задачу планировщика целей")

    # Настройка хранилища
    logger.info("💾 [STORAGE] Настраиваем хранилище")
    raw_storage = RawStorage(config["paths"]["raw"])
    logger.info(f"✅ [STORAGE] RawStorage создан, путь: {config['paths']['raw']}")

    # Настройка тематического хранилища
    logger.info("🎨 [THEMES] Настраиваем тематическое хранилище")
    theme_store = ThemeStore(str(Path(config["paths"]["index"]).parent / "themes.json"))
    theme_store.migrate_chat_names_to_sanitized()
    active_theme: str = "default"
    logger.info(f"✅ [THEMES] ThemeStore создан, активная тема: {active_theme}")

    # Векторный индекс и embeddings отключены
    logger.info("🚫 [VECTOR] Векторный индекс отключен для фокуса на загрузке Telegram чатов")
    vector_index = None
    embeddings_client = None

    # Кластерный менеджер, поисковик и суммаризатор отключены
    logger.info("🚫 [CLUSTERS] Кластерный менеджер отключен для фокуса на загрузке Telegram чатов")
    cluster_manager = None
    insight_store = None
    retriever = None
    summarizer = None

    # Планировщик кластеров и финансовый агент отключены
    logger.info("🚫 [CLUSTER_SCHEDULER] Планировщик кластеров отключен для фокуса на загрузке Telegram чатов")
    cluster_scheduler = None
    cluster_task = None
    fin_agent = None

    message_id = 0

    # Упрощенные обработчики сообщений (только для загрузки Telegram чатов)
    logger.info("💬 [HANDLERS] Настраиваем упрощенные обработчики сообщений")
    async def on_message(text: str, chat_id: int) -> str:
        """Простой обработчик для режима загрузки Telegram чатов."""
        nonlocal message_id
        message_id += 1
        logger.info(f"💬 [MESSAGE] ID: {message_id}, Chat: {chat_id}, Text: {text[:50]}...")
        
        # Простой ответ для тестирования
        return f"Сообщение получено (ID: {message_id}, Chat: {chat_id}). Режим загрузки Telegram чатов активен."

    # Обработчик события reply_ready для отправки ответов в Telegram
    async def handle_reply_ready(event: ReplyReady) -> None:
        """Обрабатывает событие reply_ready и отправляет ответ в Telegram."""
        try:
            logger.info(
                "reply_ready_received",
                extra={
                    "chat_id": event.chat_id,
                    "message_id": event.message_id,
                    "reply_length": len(event.reply)
                }
            )
            
            # Отправляем ответ в Telegram через бота
            if bot_enabled and bot:
                try:
                    await bot.send_message(
                        chat_id=event.chat_id,
                        text=event.reply,
                        parse_mode=None
                    )
                    logger.info(f"✅ Ответ отправлен в Telegram для chat_id={event.chat_id}")
                except Exception as e:
                    logger.error(f"❌ Ошибка отправки в Telegram: {e}")
            else:
                logger.warning("Бот не доступен для отправки ответа")
                
        except Exception as e:
            logger.error(f"Ошибка обработки reply_ready: {e}")

    # Подписка на событие reply_ready будет добавлена после создания бота

    async def on_message_stream(
        text: str, chat_id: int, send: Callable[[str], Awaitable[None]]
    ) -> None:
        """Упрощенный обработчик стриминга для режима загрузки Telegram чатов."""
        logger.info(f"💬 [STREAM] Chat: {chat_id}, Text: {text[:50]}...")
        
        # Простой ответ для тестирования
        response = f"Стрим-сообщение получено (Chat: {chat_id}). Режим загрузки Telegram чатов активен."
        await send(response)

    # Настройка Telegram бота
    logger.info("🤖 [TELEGRAM] Настраиваем Telegram бота")
    bot_token = str(config.get("telegram", {}).get("bot_token", "")).strip()
    bot_enabled = bool(bot_token) and not bot_token.startswith("YOUR_TELEGRAM")
    summary_chat_id = config.get("telegram", {}).get("summary_chat_id")
    logger.info(f"🤖 [TELEGRAM] Bot token: {'найден' if bot_enabled else 'не найден'}, summary_chat_id: {summary_chat_id}")

    # Настройка планировщика сводок
    logger.info("📊 [SUMMARY] Настраиваем планировщик сводок")
    scheduler = TaskScheduler()
    summary_interval = int(
        config.get("scheduler", {}).get("summary_interval_seconds", 3600)
    )
    summary_interval_file = Path("config/summary_interval.yaml")
    if summary_interval_file.exists():
        try:
            data = (
                yaml.safe_load(summary_interval_file.read_text(encoding="utf-8")) or {}
            )
            summary_interval = int(
                data.get("summary_interval_seconds", summary_interval)
            )
            logger.info(f"📊 [SUMMARY] Интервал сводок загружен из файла: {summary_interval}s")
        except Exception:
            logger.exception("❌ [SUMMARY] Ошибка загрузки интервала сводок")
    logger.info(f"📊 [SUMMARY] Интервал сводок: {summary_interval}s")

    bot: Optional[TelegramBot] = None

    # ----------------------- Services setup ---------------------------------
    logger.info("🔧 [SERVICES] Настраиваем сервисы")
    tele_cfg = config.get("telethon", {})
    api_id = tele_cfg.get("api_id")
    api_hash = tele_cfg.get("api_hash")
    
    # Only initialize Telethon if we have valid API credentials
    tele_indexer = None
    telethon_service = None
    
    if api_id and api_hash and api_id != 0 and api_hash != "":
        logger.info("📱 [TELETHON] Настраиваем Telethon сервис")
        try:
            tele_indexer = TelethonIndexer(
                api_id,
                api_hash,
                tele_cfg.get("session", "user"),
            )
            logger.info("✅ [TELETHON] TelethonIndexer создан")
            
            index_state_path = Path(config["paths"]["index"]).parent / "last_indexed.txt"
            try:
                telethon_service = TelethonService(
                    tele_indexer,
                    raw_storage,
                    vector_index,
                    cluster_manager,
                    theme_store,
                    index_state_path,
                    lambda: active_theme,
                )
                logger.info("✅ [TELETHON] TelethonService создан")
            except TypeError:
                telethon_service = TelethonService()  # type: ignore[call-arg]
                logger.info("✅ [TELETHON] TelethonService создан (mock режим)")
                
            # Подключаемся к Telethon асинхронно в фоне
            asyncio.create_task(telethon_service.ensure_connected())
            logger.info("✅ [TELETHON] Telethon сервис запущен в фоне")
        except Exception as e:
            logger.warning(f"⚠️ [TELETHON] Ошибка инициализации Telethon сервиса: {e}")
            tele_indexer = None
            telethon_service = None
    else:
        logger.warning("⚠️ [TELETHON] Telethon сервис отключен: нет валидных API credentials")
        index_state_path = Path(config["paths"]["index"]).parent / "last_indexed.txt"

    # Сервисы сводок и хроник отключены
    logger.info("🚫 [SERVICES] Сервисы сводок и хроник отключены для фокуса на загрузке Telegram чатов")
    summary_service = None
    chronicle_service = None

    # Вспомогательные функции отключены
    logger.info("🚫 [HELPERS] Вспомогательные функции отключены для фокуса на загрузке Telegram чатов")
    async def summarize_cluster_texts(texts: List[str]) -> str:
        return "Функция суммаризации отключена"
    
    async def agent_report_texts(texts: List[str]) -> str:
        return "Функция отчетов агента отключена"

    # ----------------------- Initial backup (delayed) -----------------------
    logger.info("⏳ [BOOTSTRAP] Настраиваем первичный бэкап и индексацию (отложенный запуск)")
    try:
        initial_delay = int(config.get("scheduler", {}).get("initial_backup_delay_seconds", 3600))
    except Exception:
        initial_delay = 3600

    bootstrap_flag = Path("config/.initial_backup_done")
    bootstrap_scheduler = TaskScheduler()

    async def _initial_backup() -> None:
        try:
            # Выполняем только один раз
            if bootstrap_flag.exists():
                return
            if not telethon_service:
                logger.warning("[BOOTSTRAP] Пропуск: Telethon сервис недоступен")
                bootstrap_flag.write_text("skip", encoding="utf-8")
                return
            logger.info("[BOOTSTRAP] Запускаем dump_since(1) → index_dumped(1)")
            try:
                await telethon_service.dump_since(1)
            except Exception:
                logger.exception("[BOOTSTRAP] Ошибка dump_since(1)")
            try:
                await telethon_service.index_dumped(1)
            except Exception:
                logger.exception("[BOOTSTRAP] Ошибка index_dumped(1)")
            try:
                bootstrap_flag.parent.mkdir(parents=True, exist_ok=True)
                bootstrap_flag.write_text("done", encoding="utf-8")
            except Exception:
                logger.exception("[BOOTSTRAP] Не удалось сохранить флаг завершения")
            # Уведомление в Telegram (если доступно)
            try:
                if bot and getattr(bot, "app", None) and summary_chat_id:
                    await bot.app.bot.send_message(
                        chat_id=summary_chat_id,
                        text="Первичный бэкап и индексация за 1 день завершены",
                    )
            except Exception:
                logger.exception("[BOOTSTRAP] Не удалось отправить уведомление")
        except Exception:
            logger.exception("[BOOTSTRAP] Неожиданная ошибка первичного бэкапа")

    # Планируем задачу с отложенным стартом
    bootstrap_scheduler.add_periodic(lambda: _initial_backup(), interval=24 * 3600, initial_delay=initial_delay)
    try:
        asyncio.get_running_loop()
        asyncio.create_task(bootstrap_scheduler.run())
        logger.info("✅ [BOOTSTRAP] Задача первичного бэкапа запланирована через %s сек", initial_delay)
    except RuntimeError:
        logger.warning("⚠️ [BOOTSTRAP] Нет рабочего event loop для запуска планировщика первичного бэкапа")

    async def list_themes() -> List[str]:
        """Return available theme names."""
        return theme_store.list_themes()

    async def get_active_theme_name() -> str:
        return active_theme

    async def set_active_theme(name: str) -> None:
        """Switch to a different theme and reload indexes."""
        nonlocal active_theme, vector_index
        active_theme = sanitize_name(name)
        vector_index = VectorIndex.themed(
            str(Path(config["paths"]["index"]).parent),
            active_theme,
            model_name=embed_model,
            host=emb_host,
            port=emb_port,
            api_key=emb_api_key,
        )
        retriever.index = vector_index
        if telethon_service:
            telethon_service.vector_index = vector_index
        summary_service.vector_index = vector_index

    async def create_new_theme(name: str, chats: List[str]) -> bool:
        """Create or update a theme with selected chats."""
        try:
            mapping = {_sanitize_component(c): c for c in chats}
            theme_store.set_theme(name, mapping)
            return True
        except Exception:
            logger.exception("Failed to create theme")
            return False

    async def delete_theme_by_name(name: str) -> bool:
        try:
            return theme_store.delete_theme(name)
        except Exception:
            logger.exception("Failed to delete theme: %s", name)
            return False

    async def get_theme_chats(name: str) -> List[str]:
        return list(theme_store.get_chats(name).values())

    async def add_chat_to_theme_by_name(theme: str, chat: str) -> bool:
        try:
            theme_store.add_chat_to_theme(theme, _sanitize_component(chat), chat)
            return True
        except Exception:
            logger.exception("Failed to add chat to theme")
            return False

    async def remove_chat_from_theme_by_name(theme: str, chat: str) -> bool:
        try:
            return theme_store.remove_chat_from_theme(theme, _sanitize_component(chat))
        except Exception:
            logger.exception("Failed to remove chat from theme")
            return False

    # Финансовый анализ отключен (модуль finance удален)
    async def run_finance_analysis(tickers: List[str]):
        return {
            "report_markdown": "Финансовый анализ отключен для фокуса на загрузке Telegram чатов",
            "recommendations": [],
            "risk_score": 0.0
        }

    logger.info("✅ [HELPERS] Вспомогательные функции настроены")

    # Optional smoke-test mode to verify startup without long-running services
    if os.getenv("SMOKE_TEST"):
        logger.info(
            "🧪 [SMOKE_TEST] Запуск в режиме SMOKE_TEST: выполнение одной сводки и выход"
        )
        await summary_service.hourly_summary()
        await summary_service.stop()
        await chronicle_service.stop()
        await memory_scheduler.stop()
        await goal_scheduler.stop()
        await cluster_scheduler.stop()

        # Собираем только существующие задачи
        tasks_to_gather = []
        if memory_task:
            tasks_to_gather.append(memory_task)
        if goal_task:
            tasks_to_gather.append(goal_task)
        if cluster_task:
            tasks_to_gather.append(cluster_task)

        if tasks_to_gather:
            await asyncio.gather(*tasks_to_gather, return_exceptions=True)
        await bus.join()
        await bus.graceful_shutdown()
        
        if search_client:
            await search_client.close()
        
        if agent_memory:
            agent_memory.save()
        if memory:
            memory.save()
        logger.info("✅ [SMOKE_TEST] SMOKE_TEST завершен")
        return

    # Optional indexing mode to index new messages from last indexed date
    if os.getenv("INDEX_NOW"):
        logger.info(
            "📚 [INDEX_NOW] Запуск в режиме INDEX_NOW: индексация новых сообщений с последней даты"
        )
        messages_count = 0
        start_time = datetime.now(UTC)
        last_log_time = start_time
        
        # Импортируем функцию извлечения данных
        from utils.message_extractor import extract_message_data
        
        async for message in tele_indexer.index_once():
            # Проверяем, является ли это маркером завершения чата
            if isinstance(message, dict) and message.get("type") == "chat_completed":
                chat_name = message["chat_name"]
                new_messages = message["new_messages"]
                
                # Удаляем старые сообщения, если добавили новые
                if new_messages > 0:
                    # Специальная логика для Marketapp чатов - не удаляем старые сообщения
                    if "marketapp" in chat_name.lower():
                        logger.info(f"📊 Marketapp чат '{chat_name}': сохранены все {new_messages} новых сообщений (без удаления старых)")
                    else:
                        removed_count = raw_storage.trim_old_messages(chat_name, max_messages=2000)
                        if removed_count > 0:
                            logger.info(f"🗑️  Удалено {removed_count} старых сообщений из чата '{chat_name}'")
                continue
            
            # Используем новую функцию извлечения расширенной структуры
            msg_data = extract_message_data(message)
            
            # Сохраняем сообщение в raw storage
            raw_storage.save(msg_data["chat"], msg_data)
            
            messages_count += 1
            
            # Логируем прогресс каждые 100 сообщений или каждые 30 секунд
            current_time = datetime.now(UTC)
            if messages_count % 100 == 0 or (current_time - last_log_time).total_seconds() > 30:
                elapsed = (current_time - start_time).total_seconds()
                speed = messages_count / elapsed if elapsed > 0 else 0
                msg_date = getattr(message, "date", current_time)
                logger.info(f"📚 [INDEX_NOW] {messages_count} сообщений | Чат: {msg_data['chat']} | Дата: {msg_date} | Скорость: {speed:.1f}/с")
                last_log_time = current_time
            
            # Векторный индекс отключен, пропускаем индексацию

        index_state_path.write_text(datetime.now(UTC).isoformat(), encoding="utf-8")
        logger.info(f"✅ [INDEX_NOW] Полная индексация завершена: обработано {messages_count} сообщений")

        if summary_service:
            await summary_service.stop()
        if chronicle_service:
            await chronicle_service.stop()
        if memory_scheduler:
            await memory_scheduler.stop()
        if goal_scheduler:
            await goal_scheduler.stop()
        if cluster_scheduler:
            await cluster_scheduler.stop()

        # Собираем только существующие задачи
        tasks_to_gather = []
        if memory_task:
            tasks_to_gather.append(memory_task)
        if goal_task:
            tasks_to_gather.append(goal_task)
        if cluster_task:
            tasks_to_gather.append(cluster_task)

        if tasks_to_gather:
            await asyncio.gather(*tasks_to_gather, return_exceptions=True)
        await bus.join()
        await bus.graceful_shutdown()
        
        if search_client:
            await search_client.close()
        
        if agent_memory:
            agent_memory.save()
        if memory:
            memory.save()
        logger.info("✅ [INDEX_NOW] INDEX_NOW завершен")
        return

    # Создание упрощенного списка задач (без дашборда)
    logger.info("📋 [TASKS] Создаем упрощенный список задач для загрузки Telegram чатов")
    tasks: List[asyncio.Task[Any]] = []
    logger.info(f"📋 [TASKS] Создано {len(tasks)} задач (без дашборда)")

    # Создание упрощенного Telegram бота для загрузки чатов
    if bot_enabled and telethon_service:
        logger.info("🤖 [BOT] Создаем упрощенный Telegram бот для загрузки чатов")
        bot = TelegramBot(
            bot_token,
            on_message,
            on_message_stream=on_message_stream,
            list_chats=telethon_service.list_chats,
            index_last=telethon_service.index_last,
            summarize_interval=None,  # Отключено
            set_summary_interval=None,  # Отключено
            list_themes=list_themes,
            set_active_theme=set_active_theme,
            get_active_theme_name=get_active_theme_name,
            run_finance_analysis=None,  # Отключено
            telethon_auth_request_code=telethon_service.telethon_auth_request_code,
            telethon_auth_sign_in=telethon_service.telethon_auth_sign_in,
            telethon_is_authorized=telethon_service.telethon_is_authorized,
            create_new_theme=create_new_theme,
            delete_theme_by_name=delete_theme_by_name,
            get_theme_chats=get_theme_chats,
            add_chat_to_theme_by_name=add_chat_to_theme_by_name,
            remove_chat_from_theme_by_name=remove_chat_from_theme_by_name,
            refresh_chat_cache=telethon_service.refresh_chat_cache,
            dump_since=telethon_service.dump_since,
            index_dumped=telethon_service.index_dumped,
            publish_chronicle=None,  # Отключено
            retriever=None,  # Отключено
            index_state_path=index_state_path,
            summarize_cluster=None,  # Отключено
            summarize_as_agent=None,  # Отключено
            summarize_url=None,  # Отключено
            bus=bus,
            tele_indexer=tele_indexer,
            telethon_service=telethon_service,
        )
        # attach interval reindex callback
        setattr(bot, "_index_since", telethon_service.index_since)
        tasks.append(asyncio.create_task(bot.start()))
        logger.info("✅ [BOT] Telegram бот создан и добавлен в задачи")
        asyncio.create_task(setup_broadcast_executor(bot))
        
        # Подписываемся на событие reply_ready для отправки ответов
        bus.subscribe("reply_ready", handle_reply_ready)
        logger.info("✅ [BOT] Подписка на reply_ready настроена")
        
        # Настройка сервиса мониторинга
        logger.info("🔍 [MONITORING] Настраиваем сервис мониторинга")
        monitoring_service = MonitoringService(
            summary_chat_id=summary_chat_id,
            telegram_bot=bot,
            check_interval=60  # Проверка каждую минуту
        )
        set_monitoring_service(monitoring_service)
        # Запускаем сервис мониторинга в фоне
        asyncio.create_task(monitoring_service.start())
        logger.info("✅ [MONITORING] Сервис мониторинга запущен в фоне")

        # Ретрансляция этапов работы агента в чат пользователя
        async def _try_send(text: str) -> None:
            try:
                if bot and getattr(bot, "app", None) and getattr(bot.app, "bot", None):
                    if core.chat_id is not None:
                        await bot.app.bot.send_message(chat_id=core.chat_id, text=text)
            except Exception:
                logger.exception("Не удалось отправить прогресс в чат")

        async def _on_plan(event: Event) -> None:
            try:
                from agent.core import PlanGenerated  # local import for typing
                if isinstance(event, PlanGenerated):
                    steps = len(getattr(event.plan, "steps", []) or [])
                    await _try_send(f"План сформирован: {steps} шаг(ов)")
            except Exception:
                pass

        async def _on_exec(event: Event) -> None:
            try:
                from events.models import ExecutionCompleted as _EC
                if isinstance(event, _EC):
                    await _try_send("Шаг выполнен")
            except Exception:
                pass

        async def _on_report(event: Event) -> None:
            try:
                from events.models import ExecutionCompleted as _EC
                if isinstance(event, _EC):
                    await _try_send("Готово")
            except Exception:
                pass

        async def _on_error(event: Event) -> None:
            try:
                from events.models import ErrorOccurred as _Err
                if isinstance(event, _Err):
                    await _try_send(f"Ошибка: {event.error}")
            except Exception:
                pass

        bus.subscribe("plan", _on_plan)
        bus.subscribe("execution", _on_exec)
        bus.subscribe("report", _on_report)
        bus.subscribe("errors", _on_error)
    else:
        if not bot_enabled:
            logger.warning("⚠️ [BOT] Telegram бот отключен (нет валидного токена)")
        if not telethon_service:
            logger.warning("⚠️ [BOT] Telegram бот отключен (нет валидных Telethon credentials)")

    # Запуск всех задач
    logger.info(f"🚀 [STARTUP] Запускаем {len(tasks)} задач")
    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        logger.info("⏹️ [SHUTDOWN] Получен сигнал отмены")
        pass
    finally:
        logger.info("🛑 [SHUTDOWN] Начинаем завершение работы")
        
        # Останавливаем только активные компоненты
        if hot_reload_manager:
            hot_reload_manager.stop()
        
        # Останавливаем мониторинг
        monitoring_service = get_monitoring_service()
        if monitoring_service:
            await monitoring_service.stop()
        
        await bus.join()
        await bus.graceful_shutdown()
        
        if search_client:
            await search_client.close()
        
        if memory_scheduler:
            await memory_scheduler.stop()
        
        if goal_scheduler:
            await goal_scheduler.stop()
        
        if cluster_scheduler:
            await cluster_scheduler.stop()
        
        if agent_memory:
            agent_memory.save()
        
        if memory:
            memory.save()

        # Логируем сводку производительности
        log_performance_summary()
        logger.info("✅ [SHUTDOWN] Завершение работы завершено")


if __name__ == "__main__":  # pragma: no cover - script entry
    asyncio.run(main())
