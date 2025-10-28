"""
Тесты для ScriptStore.

Проверяет:
- Сохранение скриптов
- Загрузка скриптов
- Управление версиями
- Обработка ошибок
- Потокобезопасность
"""

import tempfile
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from shell_mcp.services.script_store import ScriptStore


class TestScriptStore:
    """Тесты для класса ScriptStore."""

    def setup_method(self):
        """Настройка перед каждым тестом."""
        self.temp_dir = tempfile.mkdtemp()
        self.store = ScriptStore()
        # Переопределяем корневую директорию для тестов
        self.store.root = Path(self.temp_dir)
        self.store.index_path = self.store.root / "index.json"
        self.store._write_index({})

    def teardown_method(self):
        """Очистка после каждого теста."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_init_creates_directory(self):
        """Тест создания директории при инициализации."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("shell_mcp.services.script_store.get_settings") as mock_settings:
                mock_settings.return_value.SAVED_SCRIPTS_DIR = temp_dir

                store = ScriptStore()
                assert store.root.exists()
                assert store.index_path.exists()

    def test_save_script_basic(self):
        """Тест базового сохранения скрипта."""
        code = "print('Hello World')"
        meta = self.store.save("test-script", "python", code)

        assert meta["name"] == "test-script"
        assert meta["slug"] == "test-script"
        assert meta["language"] == "python"
        assert meta["path"] == str(self.store.root / "test-script.py")
        assert "updated_at" in meta

        # Проверяем, что файл создан
        script_path = Path(meta["path"])
        assert script_path.exists()
        assert script_path.read_text() == code

        # Проверяем индекс
        index = self.store._read_index()
        assert "test-script" in index
        assert index["test-script"]["name"] == "test-script"

    def test_save_script_with_special_characters(self):
        """Тест сохранения скрипта со специальными символами."""
        code = "print('Hello World!')"
        meta = self.store.save("test script with spaces!", "python", code)

        assert meta["slug"] == "test-script-with-spaces"

        # Проверяем, что файл создан с правильным именем
        script_path = Path(meta["path"])
        assert script_path.exists()
        assert script_path.name == "test-script-with-spaces.py"

    def test_save_script_different_languages(self):
        """Тест сохранения скриптов на разных языках."""
        python_code = "print('Hello')"
        bash_code = "echo 'Hello'"

        python_meta = self.store.save("python-script", "python", python_code)
        bash_meta = self.store.save("bash-script", "bash", bash_code)

        assert python_meta["path"].endswith(".py")
        assert bash_meta["path"].endswith(".sh")

        # Проверяем содержимое файлов
        python_path = Path(python_meta["path"])
        bash_path = Path(bash_meta["path"])

        assert python_path.read_text() == python_code
        assert bash_path.read_text() == bash_code

    def test_save_script_collision_resolution(self):
        """Тест разрешения коллизий имен."""
        code = "print('Hello')"

        # Сохраняем первый скрипт
        meta1 = self.store.save("test-script", "python", code)
        assert meta1["slug"] == "test-script"

        # Сохраняем второй с тем же именем
        meta2 = self.store.save("test-script", "python", code)
        assert meta2["slug"] == "test-script"  # Перезаписывает

        # Сохраняем третий с похожим именем
        meta3 = self.store.save("test script", "python", code)
        assert meta3["slug"] == "test-script-2"

    def test_list_scripts(self):
        """Тест получения списка скриптов."""
        # Сохраняем несколько скриптов
        self.store.save("script1", "python", "print('1')")
        time.sleep(0.01)  # Небольшая задержка для разных времен
        self.store.save("script2", "bash", "echo '2'")

        scripts = self.store.list()

        assert len(scripts) == 2
        # Должны быть отсортированы по времени обновления (новые первыми)
        assert scripts[0]["name"] == "script2"
        assert scripts[1]["name"] == "script1"

    def test_get_script_by_slug(self):
        """Тест получения скрипта по слагу."""
        code = "print('Hello')"
        self.store.save("test-script", "python", code)

        meta = self.store.get("test-script")
        assert meta["name"] == "test-script"
        assert meta["language"] == "python"

    def test_get_script_by_name(self):
        """Тест получения скрипта по имени."""
        code = "print('Hello')"
        self.store.save("Test Script", "python", code)

        meta = self.store.get("Test Script")
        assert meta["name"] == "Test Script"
        assert meta["language"] == "python"

    def test_get_script_case_insensitive(self):
        """Тест получения скрипта без учета регистра."""
        code = "print('Hello')"
        self.store.save("Test Script", "python", code)

        meta = self.store.get("test script")
        assert meta["name"] == "Test Script"

    def test_get_script_not_found(self):
        """Тест получения несуществующего скрипта."""
        with pytest.raises(KeyError, match="Saved script not found"):
            self.store.get("nonexistent-script")

    def test_delete_script_by_slug(self):
        """Тест удаления скрипта по слагу."""
        code = "print('Hello')"
        meta = self.store.save("test-script", "python", code)
        script_path = Path(meta["path"])

        # Проверяем, что файл существует
        assert script_path.exists()

        # Удаляем скрипт
        deleted_meta = self.store.delete("test-script")

        assert deleted_meta["name"] == "test-script"
        assert not script_path.exists()

        # Проверяем, что скрипт удален из индекса
        with pytest.raises(KeyError):
            self.store.get("test-script")

    def test_delete_script_by_name(self):
        """Тест удаления скрипта по имени."""
        code = "print('Hello')"
        meta = self.store.save("Test Script", "python", code)
        script_path = Path(meta["path"])

        # Удаляем скрипт по имени
        deleted_meta = self.store.delete("Test Script")

        assert deleted_meta["name"] == "Test Script"
        assert not script_path.exists()

    def test_delete_script_not_found(self):
        """Тест удаления несуществующего скрипта."""
        with pytest.raises(KeyError, match="Saved script not found"):
            self.store.delete("nonexistent-script")

    def test_thread_safety(self):
        """Тест потокобезопасности операций."""
        results = []
        errors = []

        def save_script(script_id):
            try:
                meta = self.store.save(
                    f"script-{script_id}", "python", f"print({script_id})"
                )
                results.append(meta["slug"])
            except Exception as e:
                errors.append(str(e))

        # Запускаем несколько потоков одновременно
        threads = []
        for i in range(10):
            thread = threading.Thread(target=save_script, args=(i,))
            threads.append(thread)
            thread.start()

        # Ждем завершения всех потоков
        for thread in threads:
            thread.join()

        # Проверяем, что нет ошибок
        assert len(errors) == 0

        # Проверяем, что все скрипты сохранены
        assert len(results) == 10

        # Проверяем, что все скрипты доступны
        scripts = self.store.list()
        assert len(scripts) == 10

    def test_slugify_function(self):
        """Тест функции slugify."""
        assert ScriptStore._slugify("Hello World!") == "hello-world"
        assert ScriptStore._slugify("test@#$%script") == "test-script"
        assert ScriptStore._slugify("") == "script"
        assert ScriptStore._slugify("   ") == "script"
        assert ScriptStore._slugify("123") == "123"

    def test_resolve_slug_collision(self):
        """Тест разрешения коллизий слагов."""
        index = {}

        # Первый скрипт
        slug1 = ScriptStore._resolve_slug_collision("test", "Test", index)
        assert slug1 == "test"
        index[slug1] = {"name": "Test"}

        # Второй скрипт с тем же именем (перезаписывает)
        slug2 = ScriptStore._resolve_slug_collision("test", "Test", index)
        assert slug2 == "test"

        # Третий скрипт с похожим именем
        slug3 = ScriptStore._resolve_slug_collision("test", "Test Script", index)
        assert slug3 == "test-2"

    def test_index_corruption_recovery(self):
        """Тест восстановления после повреждения индекса."""
        # Повреждаем индекс
        self.store.index_path.write_text("invalid json")

        # Создаем новый store - должен восстановиться
        new_store = ScriptStore()
        new_store.root = self.store.root
        new_store.index_path = self.store.index_path

        # Индекс должен быть пустым
        index = new_store._read_index()
        assert index == {}

    def test_index_file_not_exists(self):
        """Тест работы когда файл индекса не существует."""
        # Удаляем файл индекса
        self.store.index_path.unlink()

        # Создаем новый store
        new_store = ScriptStore()
        new_store.root = self.store.root
        new_store.index_path = self.store.index_path

        # Индекс должен быть пустым
        index = new_store._read_index()
        assert index == {}


class TestScriptStoreIntegration:
    """Интеграционные тесты ScriptStore."""

    def test_full_workflow(self):
        """Тест полного рабочего процесса."""
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ScriptStore()
            store.root = Path(temp_dir)
            store.index_path = store.root / "index.json"
            store._write_index({})

            # 1. Сохраняем скрипт
            code = "print('Hello World')"
            store.save("hello-world", "python", code)

            # 2. Проверяем, что скрипт в списке
            scripts = store.list()
            assert len(scripts) == 1
            assert scripts[0]["name"] == "hello-world"

            # 3. Получаем скрипт
            retrieved_meta = store.get("hello-world")
            assert retrieved_meta["name"] == "hello-world"

            # 4. Проверяем содержимое файла
            script_path = Path(retrieved_meta["path"])
            assert script_path.read_text() == code

            # 5. Удаляем скрипт
            deleted_meta = store.delete("hello-world")
            assert deleted_meta["name"] == "hello-world"

            # 6. Проверяем, что скрипт удален
            assert not script_path.exists()
            with pytest.raises(KeyError):
                store.get("hello-world")

    def test_index_existing_files(self):
        """Тест индексации существующих файлов."""
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ScriptStore()
            store.root = Path(temp_dir)
            store.index_path = store.root / "index.json"
            store._write_index({})

            # Создаем тестовые файлы
            test_files = [
                ("test_script.py", "python"),
                ("bash_script.sh", "bash"),
                ("node_script.mjs", "node"),
                ("unknown_file.txt", None),  # Неподдерживаемый тип
            ]

            for filename, expected_language in test_files:
                file_path = store.root / filename
                file_path.write_text(f"# {filename} content")

            # Индексируем файлы
            indexed_files = store.index_existing_files()

            # Проверяем результаты
            assert len(indexed_files) == 3  # Только поддерживаемые файлы

            # Проверяем каждый проиндексированный файл
            indexed_names = {f["name"] for f in indexed_files}
            assert "test_script" in indexed_names
            assert "bash_script" in indexed_names
            assert "node_script" in indexed_names

            # Проверяем, что файлы добавлены в индекс
            scripts = store.list()
            assert len(scripts) == 3

            # Проверяем повторный вызов (не должно дублировать)
            indexed_files_2 = store.index_existing_files()
            assert len(indexed_files_2) == 0  # Ничего нового не найдено

    def test_detect_language_by_extension(self):
        """Тест определения языка по расширению файла."""
        store = ScriptStore()

        test_cases = [
            (".py", "python"),
            (".sh", "bash"),
            (".mjs", "node"),
            (".txt", None),
            (".js", None),
            (".PY", "python"),  # Проверяем регистронезависимость
        ]

        for extension, expected_language in test_cases:
            result = store._detect_language_by_extension(extension)
            assert result == expected_language, f"Failed for extension {extension}"

    def test_save_temp_script(self):
        """Тест сохранения временного скрипта."""
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ScriptStore()
            store.root = Path(temp_dir)
            store.temp_root = Path(temp_dir) / "tmp"
            store.temp_root.mkdir(exist_ok=True)
            store.temp_index_path = store.temp_root / "index.json"
            store._write_temp_index({})

            # Сохраняем временный скрипт
            meta = store.save_temp("test_temp", "python", "print('Hello')")
            
            assert meta["name"] == "test_temp"
            assert meta["slug"] == "test_temp"
            assert meta["language"] == "python"
            assert meta["is_temp"] is True
            assert "created_at" in meta
            assert "updated_at" in meta
            
            # Проверяем, что файл создан
            script_path = Path(meta["path"])
            assert script_path.exists()
            assert script_path.read_text() == "print('Hello')"

    def test_promote_temp_to_permanent(self):
        """Тест перевода временного скрипта в постоянный."""
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ScriptStore()
            store.root = Path(temp_dir)
            store.temp_root = Path(temp_dir) / "tmp"
            store.temp_root.mkdir(exist_ok=True)
            store.temp_index_path = store.temp_root / "index.json"
            store.index_path = store.root / "index.json"
            store._write_temp_index({})
            store._write_index({})

            # Сохраняем временный скрипт
            temp_meta = store.save_temp("test_promote", "python", "print('Hello')")
            
            # Переводим в постоянный
            permanent_meta = store.promote_temp_to_permanent("test_promote")
            
            assert permanent_meta["name"] == "test_promote"
            assert permanent_meta["slug"] == "test_promote"
            assert permanent_meta["language"] == "python"
            assert permanent_meta["promoted_from_temp"] is True
            
            # Проверяем, что постоянный файл создан
            permanent_path = Path(permanent_meta["path"])
            assert permanent_path.exists()
            assert permanent_path.read_text() == "print('Hello')"
            
            # Проверяем, что временный файл удален
            temp_path = Path(temp_meta["path"])
            assert not temp_path.exists()
            
            # Проверяем, что скрипт удален из временного индекса
            temp_scripts = store.list_temp()
            assert len(temp_scripts) == 0

    def test_cleanup_old_temp_scripts(self):
        """Тест очистки старых временных скриптов."""
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ScriptStore()
            store.root = Path(temp_dir)
            store.temp_root = Path(temp_dir) / "tmp"
            store.temp_root.mkdir(exist_ok=True)
            store.temp_index_path = store.temp_root / "index.json"
            store.temp_max_age_days = 1  # Устанавливаем короткий период для теста
            store._write_temp_index({})

            # Создаем старый временный скрипт
            old_date = datetime.utcnow() - timedelta(days=2)
            old_meta = {
                "name": "old_script",
                "slug": "old_script",
                "language": "python",
                "path": str(store.temp_root / "old_script.py"),
                "created_at": old_date.isoformat() + "Z",
                "updated_at": old_date.isoformat() + "Z",
                "is_temp": True,
            }
            
            # Создаем файл и добавляем в индекс
            old_path = Path(old_meta["path"])
            old_path.write_text("print('old')")
            temp_index = store._read_temp_index()
            temp_index["old_script"] = old_meta
            store._write_temp_index(temp_index)
            
            # Создаем новый временный скрипт
            new_meta = store.save_temp("new_script", "python", "print('new')")
            
            # Очищаем старые скрипты
            cleaned_scripts = store.cleanup_old_temp_scripts()
            
            assert len(cleaned_scripts) == 1
            assert cleaned_scripts[0]["name"] == "old_script"
            
            # Проверяем, что старый файл удален
            assert not old_path.exists()
            
            # Проверяем, что новый файл остался
            new_path = Path(new_meta["path"])
            assert new_path.exists()
            
            # Проверяем, что в индексе остался только новый скрипт
            temp_scripts = store.list_temp()
            assert len(temp_scripts) == 1
            assert temp_scripts[0]["name"] == "new_script"

    def test_list_temp_scripts(self):
        """Тест получения списка временных скриптов."""
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ScriptStore()
            store.root = Path(temp_dir)
            store.temp_root = Path(temp_dir) / "tmp"
            store.temp_root.mkdir(exist_ok=True)
            store.temp_index_path = store.temp_root / "index.json"
            store._write_temp_index({})

            # Создаем несколько временных скриптов
            meta1 = store.save_temp("script1", "python", "print('1')")
            meta2 = store.save_temp("script2", "bash", "echo '2'")
            
            # Получаем список
            temp_scripts = store.list_temp()
            
            assert len(temp_scripts) == 2
            names = {s["name"] for s in temp_scripts}
            assert "script1" in names
            assert "script2" in names
