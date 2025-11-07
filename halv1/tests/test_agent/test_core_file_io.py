"""
Тесты для файловых операций в AgentCore.

Проверяет логику _execute_file_io, включая:
- Чтение существующих файлов
- Запись в файлы
- Обработку ошибок доступа
- Обработку несуществующих файлов
"""

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
from agent.core import AgentCore
from planner import PlanStep, Tool
from services.event_bus import AsyncEventBus
from executor import SimpleCodeExecutor
from memory import UnifiedMemory


class TestFileIOOperations:
    """Тесты для файловых операций."""
    
    @pytest.fixture
    def real_core(self):
        """Создает реальный AgentCore для тестирования."""
        bus = AsyncEventBus(workers_per_topic=1)
        planner = Mock()  # Мок планировщика (LLM зависимость)
        executor = SimpleCodeExecutor()  # Реальный исполнитель
        search = Mock()  # Мок поиска (внешний API)
        memory = UnifiedMemory()  # Реальная память
        code_generator = Mock()  # Мок генератора кода (LLM зависимость)
        
        core = AgentCore(
            bus=bus,
            planner=planner,
            executor=executor,
            search=search,
            memory=memory,
            code_generator=code_generator
        )
        return core
    
    @pytest.fixture
    def temp_dir(self):
        """Создает временную директорию для тестов."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    def test_read_existing_file(self, real_core, temp_dir):
        """Тест: чтение существующего файла."""
        file_path = os.path.join(temp_dir, "test.txt")
        file_content = "test file content"
        
        # Создаем реальный файл
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(file_content)
        
        step = PlanStep(
            tool=Tool.FILE_IO,
            content=f"read {file_path}"
        )
        
        result = real_core._execute_file_io(step)
        
        assert result["stdout"] == file_content
        assert result["stderr"] == ""
        assert result["files"] == {}
    
    def test_read_nonexistent_file(self, real_core, temp_dir):
        """Тест: чтение несуществующего файла."""
        nonexistent_file = os.path.join(temp_dir, "nonexistent.txt")
        
        step = PlanStep(
            tool=Tool.FILE_IO,
            content=f"read {nonexistent_file}"
        )
        
        with pytest.raises(FileNotFoundError):
            real_core._execute_file_io(step)
    
    def test_write_file_success(self, real_core, temp_dir):
        """Тест: успешная запись в файл."""
        file_path = os.path.join(temp_dir, "test.txt")
        file_content = "test content"
        
        step = PlanStep(
            tool=Tool.FILE_IO,
            content=f"write {file_path}\n{file_content}"
        )
        
        result = real_core._execute_file_io(step)
        
        assert result["stdout"] == f"wrote {file_path}"
        assert result["stderr"] == ""
        assert result["files"] == {}
        
        # Проверяем, что файл действительно создан с правильным содержимым
        assert os.path.exists(file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            assert f.read() == file_content
    
    def test_write_file_empty_content(self, real_core, temp_dir):
        """Тест: запись в файл с пустым содержимым."""
        file_path = os.path.join(temp_dir, "empty.txt")
        
        step = PlanStep(
            tool=Tool.FILE_IO,
            content=f"write {file_path}\n"
        )
        
        result = real_core._execute_file_io(step)
        
        assert result["stdout"] == f"wrote {file_path}"
        assert result["stderr"] == ""
        assert result["files"] == {}
        
        # Проверяем, что файл создан с пустым содержимым
        assert os.path.exists(file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            assert f.read() == ""
    
    def test_unknown_file_io_operation(self, real_core):
        """Тест: неизвестная файловая операция."""
        step = PlanStep(
            tool=Tool.FILE_IO,
            content="delete test.txt"
        )
        
        with pytest.raises(ValueError, match="Unknown FILE_IO operation"):
            real_core._execute_file_io(step)
    
    def test_file_io_with_empty_command(self, real_core):
        """Тест: файловая операция с пустой командой."""
        step = PlanStep(
            tool=Tool.FILE_IO,
            content=""
        )
        
        with pytest.raises(ValueError, match="Unknown FILE_IO operation"):
            real_core._execute_file_io(step)
    
    def test_file_io_with_whitespace_only_command(self, real_core):
        """Тест: файловая операция только с пробелами."""
        step = PlanStep(
            tool=Tool.FILE_IO,
            content="   "
        )
        
        with pytest.raises(ValueError, match="Unknown FILE_IO operation"):
            real_core._execute_file_io(step)
    
    def test_file_io_with_newline_only_command(self, real_core):
        """Тест: файловая операция только с переносом строки."""
        step = PlanStep(
            tool=Tool.FILE_IO,
            content="\n"
        )
        
        with pytest.raises(ValueError, match="Unknown FILE_IO operation"):
            real_core._execute_file_io(step)
    
    def test_read_file_with_absolute_path(self, real_core, temp_dir):
        """Тест: чтение файла с абсолютным путем."""
        file_path = os.path.abspath(os.path.join(temp_dir, "absolute.txt"))
        file_content = "absolute path content"
        
        # Создаем реальный файл
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(file_content)
        
        step = PlanStep(
            tool=Tool.FILE_IO,
            content=f"read {file_path}"
        )
        
        result = real_core._execute_file_io(step)
        
        assert result["stdout"] == file_content
        assert result["stderr"] == ""
        assert result["files"] == {}
    
    def test_write_file_with_absolute_path(self, real_core, temp_dir):
        """Тест: запись в файл с абсолютным путем."""
        file_path = os.path.abspath(os.path.join(temp_dir, "absolute.txt"))
        file_content = "absolute path content"
        
        step = PlanStep(
            tool=Tool.FILE_IO,
            content=f"write {file_path}\n{file_content}"
        )
        
        result = real_core._execute_file_io(step)
        
        assert result["stdout"] == f"wrote {file_path}"
        assert result["stderr"] == ""
        assert result["files"] == {}
        
        # Проверяем, что файл создан с правильным содержимым
        assert os.path.exists(file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            assert f.read() == file_content
