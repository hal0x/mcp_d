"""Тесты для ArtifactsReader."""

import tempfile
from pathlib import Path

import pytest

from memory_mcp.memory.artifacts_reader import ArtifactsReader, ArtifactMetadata


@pytest.fixture
def temp_artifacts_dir():
    """Создает временную директорию с артифактами для тестов."""
    with tempfile.TemporaryDirectory() as tmpdir:
        artifacts_dir = Path(tmpdir) / "artifacts"
        artifacts_dir.mkdir()

        # Создаем поддиректории
        (artifacts_dir / "chat_contexts").mkdir()
        (artifacts_dir / "now_summaries").mkdir()
        (artifacts_dir / "reports").mkdir()
        (artifacts_dir / "smart_aggregation_state").mkdir()

        # Создаем тестовые файлы
        (artifacts_dir / "chat_contexts" / "test_chat.md").write_text(
            "# Контекст чата: Test Chat\n\nОсновные темы: криптовалюты, блокчейн, торговля."
        )
        (artifacts_dir / "now_summaries" / "test_now.md").write_text(
            "# NOW Summary\n\nАктуальные события за последние 24 часа."
        )
        (artifacts_dir / "reports" / "test_report.md").write_text(
            "# Отчет\n\nДетальный анализ данных."
        )
        (artifacts_dir / "smart_aggregation_state" / "test.json").write_text(
            '{"chat_name": "test", "last_aggregation_time": "2024-01-01T00:00:00Z"}'
        )

        yield artifacts_dir


def test_artifacts_reader_scan(temp_artifacts_dir):
    """Тест сканирования директории артифактов."""
    reader = ArtifactsReader(artifacts_dir=temp_artifacts_dir)
    artifacts = reader.scan_artifacts_directory()

    assert len(artifacts) == 4
    assert all(isinstance(a, ArtifactMetadata) for a in artifacts)

    # Проверяем типы артифактов
    types = {a.artifact_type for a in artifacts}
    assert "chat_context" in types
    assert "now_summary" in types
    assert "report" in types
    assert "aggregation_state" in types


def test_artifacts_reader_read_file(temp_artifacts_dir):
    """Тест чтения файла артифакта."""
    reader = ArtifactsReader(artifacts_dir=temp_artifacts_dir)
    file_path = str(temp_artifacts_dir / "chat_contexts" / "test_chat.md")

    content = reader.read_artifact_file(file_path)
    assert content is not None
    assert "криптовалюты" in content
    assert "блокчейн" in content


def test_artifacts_reader_search(temp_artifacts_dir):
    """Тест поиска по артифактам."""
    reader = ArtifactsReader(artifacts_dir=temp_artifacts_dir)
    reader.scan_artifacts_directory()

    results = reader.search_artifacts("криптовалюты", limit=10)
    assert len(results) > 0
    assert any("криптовалюты" in r.content.lower() for r in results)

    # Проверяем, что результаты отсортированы по score
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_artifacts_reader_search_by_type(temp_artifacts_dir):
    """Тест поиска с фильтром по типу."""
    reader = ArtifactsReader(artifacts_dir=temp_artifacts_dir)
    reader.scan_artifacts_directory()

    results = reader.search_artifacts(
        "тест", artifact_types=["chat_context"], limit=10
    )
    assert len(results) > 0
    assert all(r.artifact_type == "chat_context" for r in results)


def test_artifacts_reader_get_metadata(temp_artifacts_dir):
    """Тест получения метаданных артифакта."""
    reader = ArtifactsReader(artifacts_dir=temp_artifacts_dir)
    file_path = str(temp_artifacts_dir / "chat_contexts" / "test_chat.md")

    metadata = reader.get_artifact_metadata(file_path)
    assert metadata is not None
    assert metadata.artifact_type == "chat_context"
    assert metadata.file_name == "test_chat.md"
    assert metadata.size > 0

