"""
Интеграционные тесты для Shell MCP Server.

Проверяет:
- Интеграцию между компонентами
- End-to-end сценарии
- Работу с Docker
- Полный цикл выполнения кода
"""

import json
import tempfile
from pathlib import Path

import pytest

from shell_mcp.services.docker_executor import DockerExecutor
from shell_mcp.services.script_store import ScriptStore


class TestDockerIntegration:
    """Интеграционные тесты с Docker."""

    @pytest.mark.integration
    def test_docker_executor_full_workflow(self):
        """Тест полного рабочего процесса DockerExecutor."""
        executor = DockerExecutor()

        # Простой Python код
        result = executor.run(
            code="print('Hello from Docker!')", language="python", timeout=30
        )

        assert result.exit_code == 0
        assert "Hello from Docker!" in result.stdout
        assert result.timed_out is False
        assert result.image == "python:3.11"
        assert result.network_enabled is True

    @pytest.mark.integration
    def test_docker_executor_with_dependencies(self):
        """Тест выполнения кода с зависимостями."""
        executor = DockerExecutor()

        result = executor.run(
            code="""
import requests
import json

# Простой HTTP запрос
try:
    response = requests.get('https://httpbin.org/json', timeout=5)
    data = response.json()
    print(f"Status: {response.status_code}")
    print(f"Data keys: {list(data.keys())}")
except Exception as e:
    print(f"Error: {e}")
""",
            language="python",
            dependencies=["requests"],
            timeout=60,
        )

        assert result.exit_code == 0
        assert "Status: 200" in result.stdout
        assert "Data keys:" in result.stdout

    @pytest.mark.integration
    def test_docker_executor_with_artifacts(self):
        """Тест создания и экспорта артефактов."""
        executor = DockerExecutor()

        code = """
import os
import json
import csv

# Создаем директорию для артефактов
os.makedirs('artifacts', exist_ok=True)

# Создаем JSON файл
data = {"message": "Hello World", "timestamp": "2024-01-01"}
with open('artifacts/data.json', 'w') as f:
    json.dump(data, f)

# Создаем CSV файл
with open('artifacts/data.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Name', 'Age'])
    writer.writerow(['Alice', 30])
    writer.writerow(['Bob', 25])

# Создаем текстовый файл
with open('artifacts/readme.txt', 'w') as f:
    f.write('This is a test artifact')

print('Artifacts created successfully')
"""

        with tempfile.TemporaryDirectory() as temp_dir:
            result = executor.run(
                code=code, language="python", out_artifacts_path=temp_dir, timeout=30
            )

            assert result.exit_code == 0
            assert "Artifacts created successfully" in result.stdout
            assert len(result.artifacts) == 3

            # Проверяем, что артефакты были скопированы
            artifacts_dir = Path(temp_dir)
            assert (artifacts_dir / "data.json").exists()
            assert (artifacts_dir / "data.csv").exists()
            assert (artifacts_dir / "readme.txt").exists()

            # Проверяем содержимое JSON файла
            with open(artifacts_dir / "data.json") as f:
                json_data = json.load(f)
                assert json_data["message"] == "Hello World"

    @pytest.mark.integration
    def test_docker_executor_timeout(self):
        """Тест таймаута выполнения."""
        executor = DockerExecutor()

        result = executor.run(
            code="import time; time.sleep(10)", language="python", timeout=2
        )

        assert result.timed_out is True
        assert result.exit_code is None
        assert "timed out" in result.stderr.lower()

    @pytest.mark.integration
    def test_docker_executor_network_disabled(self):
        """Тест выполнения с отключенной сетью."""
        executor = DockerExecutor(default_network=False)

        result = executor.run(
            code="import urllib.request; urllib.request.urlopen('http://example.com')",
            language="python",
            timeout=10,
        )

        # Должен завершиться с ошибкой из-за отсутствия сети
        assert result.exit_code != 0
        assert result.network_enabled is False

    @pytest.mark.integration
    def test_docker_executor_different_languages(self):
        """Тест выполнения кода на разных языках."""
        executor = DockerExecutor()

        # Python
        python_result = executor.run(
            code="print('Hello from Python!')", language="python", timeout=30
        )
        assert python_result.exit_code == 0
        assert "Hello from Python!" in python_result.stdout

        # Bash
        bash_result = executor.run(
            code="echo 'Hello from Bash!'", language="bash", timeout=30
        )
        assert bash_result.exit_code == 0
        assert "Hello from Bash!" in bash_result.stdout

        # Node.js
        node_result = executor.run(
            code="console.log('Hello from Node.js!');", language="node", timeout=30
        )
        assert node_result.exit_code == 0
        assert "Hello from Node.js!" in node_result.stdout

    @pytest.mark.integration
    def test_docker_executor_resource_limits(self):
        """Тест ограничений ресурсов."""
        executor = DockerExecutor()

        # Тест с ограничением памяти
        result = executor.run(
            code="print('Memory limited execution')",
            language="python",
            memory="256m",
            timeout=30,
        )

        assert result.exit_code == 0
        assert "Memory limited execution" in result.stdout

        # Тест с ограничением CPU
        result = executor.run(
            code="print('CPU limited execution')",
            language="python",
            cpus="0.5",
            timeout=30,
        )

        assert result.exit_code == 0
        assert "CPU limited execution" in result.stdout


class TestScriptStoreIntegration:
    """Интеграционные тесты ScriptStore."""

    def test_script_store_full_workflow(self):
        """Тест полного рабочего процесса ScriptStore."""
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ScriptStore()
            store.root = Path(temp_dir)
            store.index_path = store.root / "index.json"
            store._write_index({})

            # 1. Сохраняем несколько скриптов
            python_code = "print('Hello Python!')"
            bash_code = "echo 'Hello Bash!'"

            store.save("python-script", "python", python_code)
            store.save("bash-script", "bash", bash_code)

            # 2. Проверяем список скриптов
            scripts = store.list()
            assert len(scripts) == 2

            # 3. Получаем скрипты по имени
            python_retrieved = store.get("python-script")
            bash_retrieved = store.get("bash-script")

            assert python_retrieved["name"] == "python-script"
            assert bash_retrieved["name"] == "bash-script"

            # 4. Проверяем содержимое файлов
            python_path = Path(python_retrieved["path"])
            bash_path = Path(bash_retrieved["path"])

            assert python_path.read_text() == python_code
            assert bash_path.read_text() == bash_code

            # 5. Удаляем один скрипт
            deleted_meta = store.delete("python-script")
            assert deleted_meta["name"] == "python-script"
            assert not python_path.exists()

            # 6. Проверяем, что остался только один скрипт
            scripts = store.list()
            assert len(scripts) == 1
            assert scripts[0]["name"] == "bash-script"

    def test_script_store_collision_handling(self):
        """Тест обработки коллизий имен."""
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ScriptStore()
            store.root = Path(temp_dir)
            store.index_path = store.root / "index.json"
            store._write_index({})

            # Сохраняем скрипт
            meta1 = store.save("test-script", "python", "print('First')")
            assert meta1["slug"] == "test-script"

            # Сохраняем скрипт с тем же именем (перезаписывает)
            meta2 = store.save("test-script", "python", "print('Second')")
            assert meta2["slug"] == "test-script"

            # Проверяем, что содержимое обновилось
            script_path = Path(meta2["path"])
            assert script_path.read_text() == "print('Second')"

            # Сохраняем скрипт с похожим именем
            meta3 = store.save("test script", "python", "print('Third')")
            assert meta3["slug"] == "test-script-2"

    def test_script_store_case_insensitive_search(self):
        """Тест поиска без учета регистра."""
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ScriptStore()
            store.root = Path(temp_dir)
            store.index_path = store.root / "index.json"
            store._write_index({})

            # Сохраняем скрипт с заглавными буквами
            store.save("Test Script", "python", "print('Hello')")

            # Ищем с разным регистром
            meta1 = store.get("test script")
            meta2 = store.get("TEST SCRIPT")
            meta3 = store.get("Test Script")

            assert meta1["name"] == "Test Script"
            assert meta2["name"] == "Test Script"
            assert meta3["name"] == "Test Script"


class TestMCPToolsIntegration:
    """Интеграционные тесты MCP инструментов."""

    @pytest.mark.integration
    def test_run_code_simple_integration(self):
        """Тест интеграции run_code_simple."""
        from shell_mcp.services.docker_executor import DockerExecutor

        executor = DockerExecutor()

        # Выполняем простой код
        result = executor.run(
            code="print('Hello from MCP!')", language="python", timeout=30
        )

        assert result.exit_code == 0
        assert "Hello from MCP!" in result.stdout
        assert result.timed_out is False

    @pytest.mark.integration
    def test_run_code_simple_with_script_path(self):
        """Тест run_code_simple с путем к скрипту."""
        from shell_mcp.services.docker_executor import DockerExecutor

        executor = DockerExecutor()

        # Создаем временный файл
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("print('Hello from file!')")
            temp_path = f.name

        try:
            # Читаем содержимое файла
            with open(temp_path) as f:
                file_content = f.read()

            result = executor.run(code=file_content, language="python", timeout=30)

            assert result.exit_code == 0
            assert "Hello from file!" in result.stdout
        finally:
            Path(temp_path).unlink()

    @pytest.mark.integration
    def test_run_code_multi_step_integration(self):
        """Тест интеграции run_code_multi_step."""
        from shell_mcp.services.docker_executor import DockerExecutor

        executor = DockerExecutor()

        # Выполняем многошаговый код
        result = executor.run(
            code="""import math
result = math.sqrt(16)
print(f'Square root of 16 is {result}')""",
            language="python",
            timeout=30,
        )

        assert result.exit_code == 0
        assert "Square root of 16 is 4.0" in result.stdout

    @pytest.mark.integration
    def test_saved_scripts_workflow(self):
        """Тест полного рабочего процесса с сохраненными скриптами."""
        from shell_mcp.services.docker_executor import DockerExecutor
        from shell_mcp.services.script_store import ScriptStore

        executor = DockerExecutor()
        store = ScriptStore()

        # 1. Сохраняем скрипт
        result = executor.run(
            code="print('Hello from saved script!')",
            language="python",
            timeout=30,
        )

        # Сохраняем в store
        meta = store.save("hello-script", "python", "print('Hello from saved script!')")
        assert meta["name"] == "hello-script"

        # 2. Получаем список скриптов
        scripts = store.list()
        hello_script = None
        for script in scripts:
            if script["name"] == "hello-script":
                hello_script = script
                break

        assert hello_script is not None
        assert hello_script["language"] == "python"

        # 3. Выполняем сохраненный скрипт
        script_meta = store.get("hello-script")
        script_path = Path(script_meta["path"])
        with open(script_path) as f:
            script_code = f.read()

        result = executor.run(
            code=script_code,
            language="python",
            timeout=30,
        )

        assert result.exit_code == 0
        assert "Hello from saved script!" in result.stdout

        # 4. Удаляем скрипт
        deleted_meta = store.delete("hello-script")
        assert deleted_meta["name"] == "hello-script"

        # 5. Проверяем, что скрипт удален
        scripts = store.list()
        hello_script = None
        for script in scripts:
            if script["name"] == "hello-script":
                hello_script = script
                break

        assert hello_script is None

    @pytest.mark.integration
    def test_meta_tools_integration(self):
        """Тест интеграции мета-инструментов."""
        import importlib.metadata as _meta

        from shell_mcp.config import get_settings
        from shell_mcp.services.docker_executor import ensure_docker_available

        # Проверяем здоровье
        try:
            ensure_docker_available()
            health_ok = True
            health_message = "ok"
        except Exception as e:
            health_ok = False
            health_message = str(e)

        assert isinstance(health_ok, bool)
        assert isinstance(health_message, str)

        # Получаем информацию о версии
        s = get_settings()
        try:
            ver = _meta.version("shell-mcp")
        except Exception:
            ver = "0.0.0"

        assert isinstance(ver, str)
        assert s.DEFAULT_IMAGE is not None
        assert s.CONTAINER_WORKDIR is not None
        assert isinstance(s.DEFAULT_NETWORK, bool)


class TestEndToEndScenarios:
    """End-to-end сценарии."""

    @pytest.mark.integration
    def test_data_processing_pipeline(self):
        """Тест пайплайна обработки данных."""
        from shell_mcp.services.docker_executor import DockerExecutor

        executor = DockerExecutor()

        # Создаем пайплайн обработки данных
        code = """import json, csv, os
os.makedirs('artifacts', exist_ok=True)
data = [{'name': 'Alice', 'age': 30}, {'name': 'Bob', 'age': 25}]
with open('artifacts/data.json', 'w') as f: json.dump(data, f)
with open('artifacts/data.csv', 'w', newline='') as f:
    writer = csv.writer(f); writer.writerow(['Name', 'Age'])
    for item in data: writer.writerow([item['name'], item['age']])
print('Data processing completed')"""

        with tempfile.TemporaryDirectory() as temp_dir:
            result = executor.run(
                code=code,
                language="python",
                out_artifacts_path=temp_dir,
                timeout=60,
            )

            assert result.exit_code == 0
            assert "Data processing completed" in result.stdout
            assert len(result.artifacts) == 2

            # Проверяем созданные файлы
            artifacts_dir = Path(temp_dir)
            assert (artifacts_dir / "data.json").exists()
            assert (artifacts_dir / "data.csv").exists()

            # Проверяем содержимое JSON
            with open(artifacts_dir / "data.json") as f:
                json_data = json.load(f)
                assert len(json_data) == 2
                assert json_data[0]["name"] == "Alice"

    @pytest.mark.integration
    def test_web_scraping_scenario(self):
        """Тест сценария веб-скрапинга."""
        from shell_mcp.services.docker_executor import DockerExecutor

        executor = DockerExecutor()

        # Простой веб-скрапинг
        code = """
import requests
import json
import os

os.makedirs('artifacts', exist_ok=True)

try:
    # Простой HTTP запрос
    response = requests.get('https://httpbin.org/json', timeout=10)
    data = response.json()

    # Сохраняем результат
    with open('artifacts/scraped_data.json', 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Scraping completed. Status: {response.status_code}")
    print(f"Data keys: {list(data.keys())}")

except Exception as e:
    print(f"Scraping failed: {e}")
    with open('artifacts/error.log', 'w') as f:
        f.write(str(e))
"""

        with tempfile.TemporaryDirectory() as temp_dir:
            result = executor.run(
                code=code,
                language="python",
                dependencies=["requests"],
                out_artifacts_path=temp_dir,
                timeout=60,
            )

            assert result.exit_code == 0
            assert "Scraping completed" in result.stdout
            assert "Status: 200" in result.stdout

            # Проверяем созданные файлы
            artifacts_dir = Path(temp_dir)
            assert (artifacts_dir / "scraped_data.json").exists()

            # Проверяем содержимое
            with open(artifacts_dir / "scraped_data.json") as f:
                json_data = json.load(f)
                assert isinstance(json_data, dict)

    @pytest.mark.integration
    def test_machine_learning_preprocessing(self):
        """Тест предобработки данных для машинного обучения."""
        from shell_mcp.services.docker_executor import DockerExecutor

        executor = DockerExecutor()

        # Предобработка данных
        code = """
import pandas as pd
import numpy as np
import json
import os

os.makedirs('artifacts', exist_ok=True)

# Создаем тестовые данные
data = {
    'name': ['Alice', 'Bob', 'Charlie', 'David'],
    'age': [25, 30, 35, 28],
    'salary': [50000, 60000, 70000, 55000],
    'department': ['IT', 'HR', 'IT', 'Finance']
}

df = pd.DataFrame(data)

# Предобработка
df['age_group'] = pd.cut(df['age'], bins=[0, 30, 40, 100], labels=['Young', 'Middle', 'Senior'])
df['salary_normalized'] = (df['salary'] - df['salary'].mean()) / df['salary'].std()

# Сохраняем результаты
df.to_csv('artifacts/processed_data.csv', index=False)
df.describe().to_csv('artifacts/data_summary.csv')

# Статистика
stats = {
    'total_records': len(df),
    'avg_age': df['age'].mean(),
    'avg_salary': df['salary'].mean(),
    'departments': df['department'].value_counts().to_dict()
}

with open('artifacts/stats.json', 'w') as f:
    json.dump(stats, f, indent=2)

print("Data preprocessing completed")
print(f"Processed {len(df)} records")
"""

        with tempfile.TemporaryDirectory() as temp_dir:
            result = executor.run(
                code=code,
                language="python",
                dependencies=["pandas", "numpy"],
                out_artifacts_path=temp_dir,
                timeout=60,
            )

            assert result.exit_code == 0
            assert "Data preprocessing completed" in result.stdout
            assert len(result.artifacts) == 3

            # Проверяем созданные файлы
            artifacts_dir = Path(temp_dir)
            assert (artifacts_dir / "processed_data.csv").exists()
            assert (artifacts_dir / "data_summary.csv").exists()
            assert (artifacts_dir / "stats.json").exists()

            # Проверяем статистику
            with open(artifacts_dir / "stats.json") as f:
                stats = json.load(f)
                assert stats["total_records"] == 4
                assert "avg_age" in stats
                assert "departments" in stats
