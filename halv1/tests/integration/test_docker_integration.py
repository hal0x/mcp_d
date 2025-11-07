import pytest

from executor import create_executor
from .docker_utils import check_docker_available


@pytest.mark.skipif(not check_docker_available(), reason="Docker недоступен")
class TestDockerIntegration:
    """Интеграционные тесты с реальным Docker executor."""

    @pytest.fixture
    def executor(self):
        return create_executor("docker", "venv")

    def test_simple_code_execution(self, executor):
        """Тест простого выполнения кода."""
        result = executor.execute("print('Hello World')")
        assert result.stdout.strip() == "Hello World"
        assert result.returncode == 0

    def test_file_operations(self, executor):
        """Тест файловых операций внутри контейнера."""
        code = (
            "with open('test.txt', 'w') as f:\n"
            "    f.write('test content')\n"
            "print('File created')\n"
        )
        result = executor.execute(code)
        assert result.stdout.strip() == "File created"
        assert "test.txt" in result.files
        assert result.files["test.txt"].decode() == "test content"

    def test_network_access(self, executor):
        """Тест сетевого доступа: допускает ошибку сети в ограниченных окружениях."""
        code = (
            "import urllib.request, json\n"
            "try:\n"
            "    resp = urllib.request.urlopen('http://httpbin.org/ip', timeout=10)\n"
            "    data = json.loads(resp.read().decode())\n"
            "    print(f\"IP: {data['origin']}\")\n"
            "except Exception as e:\n"
            "    print(f'Network error: {e}')\n"
        )
        result = executor.execute(code)
        out = result.stdout or ""
        assert ("IP:" in out) or ("Network error:" in out)

    def test_multi_step_execution(self, executor):
        """Тест многошагового выполнения простых выражений."""
        steps = [
            "x = 5",
            "y = 10",
            "print(f'Sum: {x + y}')",
        ]
        result = executor.execute_multi_step(steps)
        assert "Sum: 15" in (result.stdout or "")

