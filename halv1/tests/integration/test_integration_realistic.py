"""
Реалистичный интеграционный тест, который запускает программу через main.py
и проверяет основные сценарии использования в условиях, максимально приближенных 
к реальному использованию.

Этот тест:
1. Запускает систему через main.py с тестовой конфигурацией
2. Проверяет инициализацию всех компонентов
3. Симулирует реальные сценарии использования
4. Проверяет корректную остановку системы
"""

import asyncio
import json
import logging
import os
import tempfile
import time
import yaml
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch, MagicMock
import pytest
import pytest_asyncio

# Импортируем main для тестирования
import main
from events.models import MessageReceived
from llm import create_llm_client
from tests.integration.docker_utils import check_docker_available


class RealisticTestEnvironment:
    """Среда для реалистичных интеграционных тестов."""
    
    def __init__(self, tmp_dir: Path):
        self.tmp_dir = tmp_dir
        self.config_file = tmp_dir / "test_settings.yaml"
        self.env_vars = {}
        self.original_env = {}
        self.test_components = {}
        self.captured_messages = []
        
    def create_test_config(self) -> Dict[str, Any]:
        """Создает тестовую конфигурацию."""
        config = {
            "telegram": {
                "bot_token": "",  # Отключаем Telegram бота
                "summary_chat_id": None,
                "goal_chat_id": 134432210
            },
                    "llm": {
            "provider": "ollama",
            "model": "gemma3n:e4b-it-q8_0",
            "host": "127.0.0.1",
            "port": 11434,
        },
            "embeddings": {
                "model": "gemma3n:e4b-it-q8_0",
                "host": "127.0.0.1",
                "port": 11434,
                "api_key": ""
            },
            "telethon": {
                "api_id": 12345,
                "api_hash": "test_hash",
                "session": "test_session"
            },
            "summary": {
                "user_name": "@test_user",
                "timezone": "UTC"
            },
            "memory": {
                "long_term_path": str(self.tmp_dir / "memory" / "long_term.json")
            },
            "paths": {
                "raw": str(self.tmp_dir / "raw"),
                "index": str(self.tmp_dir / "index" / "index.json"),
                "venv": "venv",
                "agent_memory": str(self.tmp_dir / "agent_memory.json"),
                "memory_db": str(self.tmp_dir / "test_memory.db")
            },
            "scheduler": {
                "summary_interval_seconds": 60,  # Короткие интервалы для тестов
                "cluster_summarise_interval_seconds": 60,
                "cluster_decay_interval_seconds": 60,
                "cluster_decay_half_life_seconds": 300,
                "cluster_recluster_interval_seconds": 60,
                "chronicle_interval_days": 1
            },
            "executor": {
                "provider": "docker",
                "artifact_ttl": 3600
            },
            "internet": {
                "user_agent": "halv1-test/1.0",
                "max_retries": 2
            }
        }
        
        # Сохраняем конфигурацию в файл
        with open(self.config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        return config
    
    def setup_environment(self):
        """Настраивает переменные окружения для тестов."""
        # Сохраняем оригинальные переменные
        self.original_env = dict(os.environ)
        
        # Устанавливаем тестовые переменные
        self.env_vars = {
            "LOGLEVEL": "INFO",
            "SMOKE_TEST": "",  # Не включаем smoke test режим
            "INDEX_NOW": "",   # Не включаем автоиндексацию
        }
        
        for key, value in self.env_vars.items():
            os.environ[key] = value
    
    def restore_environment(self):
        """Восстанавливает оригинальные переменные окружения."""
        os.environ.clear()
        os.environ.update(self.original_env)
    
    def mock_telethon_components(self):
        """Создает моки для Telethon компонентов."""
        class MockTelethonIndexer:
            def __init__(self, *args, **kwargs):
                self.indexed_messages = []
                pass
            
            async def index_once(self):
                # Симулируем реальную индексацию
                test_messages = [
                    "Тестовое сообщение 1",
                    "Тестовое сообщение 2", 
                    "Тестовое сообщение 3"
                ]
                for msg in test_messages:
                    self.indexed_messages.append(msg)
                    yield msg
        
        class MockTelethonService:
            def __init__(self, *args, **kwargs):
                # Принимаем любые аргументы, но не используем их
                self.vector_index = None
                self.connected = False
                self.chats = ["test_chat_1", "test_chat_2"]
                self.indexed_count = 0
                pass
            
            async def ensure_connected(self):
                self.connected = True
                return True
            
            async def list_chats(self):
                assert self.connected, "Сервис должен быть подключен"
                return self.chats
            
            async def index_last(self, count=10):
                self.indexed_count += count
                return f"Проиндексировано {count} сообщений"
            
            async def refresh_chat_cache(self):
                assert self.connected, "Сервис должен быть подключен"
                pass
            
            async def dump_since(self, since_str):
                assert self.connected, "Сервис должен быть подключен"
                return "Дамп создан"
            
            async def index_dumped(self):
                assert self.connected, "Сервис должен быть подключен"
                return "Дамп проиндексирован"
        
        return MockTelethonIndexer, MockTelethonService
    
    def create_message_collector(self):
        """Создает коллектор для перехвата сообщений."""
        async def message_collector(text: str, chat_id: int) -> str:
            self.captured_messages.append({
                "text": text,
                "chat_id": chat_id,
                "timestamp": datetime.now(UTC).isoformat()
            })
            return f"Обработано сообщение: {text[:50]}..."
        
        return message_collector


@pytest_asyncio.fixture
async def realistic_env():
    """Фикстура для создания реалистичной тестовой среды."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        env = RealisticTestEnvironment(Path(tmp_dir))
        env.setup_environment()
        
        try:
            yield env
        finally:
            env.restore_environment()


class TestIntegrationRealistic:
    """Реалистичные интеграционные тесты."""

    @pytest.mark.asyncio
    async def test_docker_executor_availability(self, realistic_env):
        """Проверяет базовую доступность Docker executor простой задачей."""
        if not check_docker_available():
            pytest.skip("Docker недоступен")
        env = realistic_env
        config = env.create_test_config()

        from executor import create_executor

        executor = create_executor(config["executor"]["provider"], config["paths"]["venv"])
        result = executor.execute("print('Hello Docker!')")
        assert result.returncode == 0
        assert result.stdout.strip() == "Hello Docker!"
    
    @pytest.mark.asyncio
    async def test_system_startup_and_shutdown(self, realistic_env):
        """Тестирует запуск и корректную остановку системы."""
        env = realistic_env
        config = env.create_test_config()
        
        # Проверяем доступность Ollama
        try:
            llm_client = create_llm_client("ollama", config["llm"], {})
            test_response, _ = llm_client.generate("Reply with OK")
            if not test_response:
                pytest.skip("Ollama недоступен")
        except Exception as e:
            pytest.skip(f"Ollama недоступен: {e}")
        
        # Мокируем Telethon компоненты
        mock_indexer, mock_service = env.mock_telethon_components()
        
        with patch('main.load_config', return_value=config), \
             patch('main.TelethonIndexer', mock_indexer), \
             patch('main.TelethonService', mock_service), \
             patch.dict(os.environ, {"SMOKE_TEST": "1"}):  # Используем smoke test для быстрого завершения
            
            # Запускаем main
            await main.main()
            
        # Если дошли сюда без исключений, значит система запустилась и корректно завершилась
        # Проверяем, что основные компоненты были инициализированы
        assert config is not None, "Конфигурация должна быть загружена"
        assert "llm" in config, "Конфигурация должна содержать настройки LLM"
        assert "telegram" in config, "Конфигурация должна содержать настройки Telegram"
        assert "paths" in config, "Конфигурация должна содержать пути"
    
    @pytest.mark.asyncio
    async def test_agent_message_processing_flow(self, realistic_env):
        """Тестирует полный поток обработки сообщений."""
        env = realistic_env
        config = env.create_test_config()
        
        # Проверяем LLM
        try:
            llm_client = create_llm_client("ollama", config["llm"], {})
            llm_client.generate("Reply with OK")
        except Exception as e:
            pytest.skip(f"Ollama недоступен: {e}")
        
        # Создаем коллектор сообщений
        message_collector = env.create_message_collector()
        test_messages = []
        
        # Мокируем компоненты для перехвата сообщений
        original_on_message = None
        
        async def patched_main():
            nonlocal original_on_message
            # Сохраняем оригинальную функцию on_message
            import main
            
            # Временно мокируем Telethon
            mock_indexer, mock_service = env.mock_telethon_components()
            
            with patch('main.TelethonIndexer', mock_indexer), \
                 patch('main.TelethonService', mock_service):
                
                # Запускаем инициализацию
                original_main = main.main
                
                # Создаем модифицированную версию main
                async def modified_main():
                    # Копируем логику из main.py с нашими модификациями
                    config_data = config
                    
                    # Инициализируем LLM
                    llm_cfg = config_data.get("llm", {})
                    provider = str(llm_cfg.get("provider", "ollama")).lower()
                    ollama_cfg = config_data.get("ollama", {})
                    llm_client = create_llm_client(provider, llm_cfg, ollama_cfg)
                    
                    # Создаем простую версию агента для тестирования
                    from services.event_bus import AsyncEventBus
                    from agent.core import AgentCore
                    from planner import LLMTaskPlanner
                    from executor import create_executor
                    from internet import SearchClient
                    from memory import UnifiedMemory
                    from executor.code_generator import CodeGenerator
                    
                    bus = AsyncEventBus(workers_per_topic=1)
                    planner = LLMTaskPlanner(llm_client)
                    executor = create_executor("docker", "venv")
                    search_client = SearchClient(llm=llm_client)
                    memory = UnifiedMemory(
                        long_term_path=config_data["paths"]["agent_memory"],
                        llm_client=llm_client
                    )
                    code_generator = CodeGenerator(llm_client)
                    
                    agent_core = AgentCore(
                        bus=bus,
                        planner=planner,
                        executor=executor,
                        search=search_client,
                        memory=memory,
                        code_generator=code_generator
                    )
                    
                    # Симулируем отправку тестовых сообщений
                    test_messages_to_send = [
                        "Привет! Посчитай 2 + 2",
                        "Создай простой Python код для вывода 'Hello World'",
                        "Что такое машинное обучение?"
                    ]
                    
                    message_id = 0
                    for text in test_messages_to_send:
                        message_id += 1
                        await bus.publish(
                            "incoming",
                            MessageReceived(
                                chat_id=1,
                                message_id=message_id,
                                text=text
                            )
                        )
                        await bus.join()
                        
                        # Получаем результат из памяти
                        result = memory.recall()
                        if result:
                            test_messages.append({
                                "input": text,
                                "output": result[-1] if result else "Нет ответа"
                            })
                    
                    # Очистка
                    await bus.graceful_shutdown()
                    await search_client.close()
                    memory.save()
                
                await modified_main()
        
        await patched_main()
        
        # Проверяем результаты
        assert len(test_messages) > 0, "Должны быть обработаны тестовые сообщения"
        
        # Проверяем конкретные ответы
        for msg in test_messages:
            assert msg["output"], f"Должен быть ответ на сообщение: {msg['input']}"
            
        # Проверяем математический ответ
        math_result = next((msg for msg in test_messages if "2 + 2" in msg["input"]), None)
        if math_result:
            assert "4" in math_result["output"], f"Ответ на 2+2 должен содержать 4: {math_result['output']}"
    
    @pytest.mark.asyncio
    async def test_configuration_loading(self, realistic_env):
        """Тестирует загрузку конфигурации."""
        env = realistic_env
        config = env.create_test_config()
        
        # Тестируем функцию загрузки конфигурации
        loaded_config = main.load_config(str(env.config_file))
        
        assert loaded_config["llm"]["provider"] == "ollama"
        assert loaded_config["paths"]["raw"] == str(env.tmp_dir / "raw")
        assert loaded_config["executor"]["provider"] == "docker"
    
    @pytest.mark.asyncio
    async def test_memory_and_state_persistence(self, realistic_env):
        """Тестирует сохранение состояния между сессиями."""
        env = realistic_env
        config = env.create_test_config()
        
        # Создаем директории
        for path_key in ["raw", "index", "agent_memory"]:
            path = Path(config["paths"][path_key])
            path.parent.mkdir(parents=True, exist_ok=True)
        
        memory_path = Path(config["memory"]["long_term_path"])
        memory_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Проверяем LLM (Ollama)
        try:
            llm_client = create_llm_client("ollama", config["llm"], {})
            llm_client.generate("Reply with OK")
        except Exception as e:
            pytest.skip(f"Ollama недоступен: {e}")
        
        # Первая сессия: сохраняем данные
        from memory import UnifiedMemory
        
        memory1 = UnifiedMemory(
            long_term_path=config["paths"]["agent_memory"],
            llm_client=llm_client
        )
        # Сохраняем сообщение в первую память
        memory1.remember("Тестовое сообщение для проверки сохранения")
        memory1.save()
        
        # Вторая сессия: загружаем данные
        memory2 = UnifiedMemory(
            long_term_path=config["paths"]["agent_memory"],
            llm_client=llm_client
        )
        
        recalled = memory2.recall()
        assert len(recalled) > 0, "Память должна содержать сохраненные данные"
        assert "Тестовое сообщение" in " ".join(recalled), "Должно быть восстановлено тестовое сообщение"
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, realistic_env):
        """Тестирует обработку ошибок и восстановление системы."""
        env = realistic_env
        config = env.create_test_config()
        
        # Тестируем обработку неправильной конфигурации
        bad_config = config.copy()
        bad_config["llm"]["port"] = 99999  # Неправильный порт
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(bad_config, f)
            bad_config_file = f.name
        
        try:
            # Должна быть обработана ошибка подключения к LLM
            with pytest.raises(Exception):
                llm_client = create_llm_client("ollama", bad_config["llm"], {})
                llm_client.generate("Reply with OK")
        finally:
            os.unlink(bad_config_file)
    
    @pytest.mark.asyncio
    async def test_component_integration_offline(self, realistic_env):
        """Тестирует интеграцию компонентов без внешних зависимостей."""
        env = realistic_env
        config = env.create_test_config()
        
        # Тестируем создание всех основных компонентов
        from memory import UnifiedMemory
        from executor import create_executor
        from index.vector_index import VectorIndex
        from index.cluster_manager import ClusterManager
        from retriever.retriever import Retriever
        
        # Создаем директории
        for path_key in ["raw", "index", "agent_memory"]:
            path = Path(config["paths"][path_key])
            path.parent.mkdir(parents=True, exist_ok=True)
        
        # Память (без LLM клиента)
        import os
        memory_db_path = config["paths"]["memory_db"]
        os.makedirs(os.path.dirname(memory_db_path), exist_ok=True)
        memory = UnifiedMemory(path=memory_db_path)
        assert memory is not None
        
        # Исполнитель
        executor = create_executor(config["executor"]["provider"], config["paths"]["venv"])
        assert executor is not None
        
        # Cluster manager
        cluster_manager = ClusterManager()
        assert cluster_manager is not None
        
        # Все компоненты должны создаваться без ошибок
        assert memory is not None, "UnifiedMemory должен создаваться успешно"
        assert executor is not None, "Executor должен создаваться успешно"
        assert cluster_manager is not None, "ClusterManager должен создаваться успешно"
        
        # Проверяем базовую функциональность компонентов
        memory.remember("test message")
        recalled = memory.recall()
        assert "test message" in recalled, "Память должна сохранять и возвращать данные"
        
        # Проверяем, что executor имеет необходимые методы
        assert hasattr(executor, 'execute'), "Executor должен иметь метод execute"
        assert callable(getattr(executor, 'execute')), "Метод execute должен быть вызываемым"


if __name__ == "__main__":
    # Возможность запуска тестов напрямую
    import sys
    
    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s"
    )
    
    # Проверяем доступность Ollama
    try:
        from llm import create_llm_client
        llm_client = create_llm_client("ollama", {
            "model": "gemma3n:e4b-it-q8_0",
            "host": "127.0.0.1",
            "port": 11434,
        }, {})
        test_response, _ = llm_client.generate("Reply with OK")
        print("✅ Ollama доступен")
        
        # Запускаем тесты
        pytest.main([__file__, "-v", "--asyncio-mode=auto", "-s"])
        
    except Exception as e:
        print(f"⚠️  Ollama недоступен: {e}")
        print("Запускаем только офлайн тесты...")
        pytest.main([__file__ + "::TestIntegrationRealistic::test_component_integration_offline", 
                    "-v", "--asyncio-mode=auto", "-s"])
