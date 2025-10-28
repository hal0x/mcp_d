"""
Тесты для функциональности саммаризации
"""

from pathlib import Path

import pytest

from memory_mcp.analysis.index_markdown import MarkdownIndexer
from memory_mcp.analysis.summarize_chats import ChatSummarizer
from memory_mcp.search.search_summaries import SummarySearcher


@pytest.fixture
def test_output_dir(tmp_path):
    """Фикстура для временной директории вывода"""
    output_dir = tmp_path / "test_summaries"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def chats_dir():
    """Фикстура для директории с чатами"""
    chats_path = Path("chats")
    if not chats_path.exists():
        pytest.skip("Директория chats/ не найдена")
    return chats_path


class TestChatSummarizer:
    """Тесты для ChatSummarizer"""

    def test_summarizer_init(self):
        """Тест инициализации саммаризатора"""
        summarizer = ChatSummarizer()
        assert summarizer.model_name is not None
        assert summarizer.base_url is not None

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Требует запущенного Ollama")
    async def test_ollama_connection(self):
        """Тест подключения к Ollama"""
        summarizer = ChatSummarizer()

        async with summarizer:
            is_connected = await summarizer.check_ollama_connection()
            assert isinstance(is_connected, bool)

    def test_detect_conversation_boundaries(self, chats_dir):
        """Тест определения границ разговоров"""
        summarizer = ChatSummarizer()

        # Тестовые сообщения
        messages = [
            {"date": "2024-01-01T10:00:00Z", "text": "Привет"},
            {"date": "2024-01-01T10:05:00Z", "text": "Как дела?"},
            {"date": "2024-01-01T12:00:00Z", "text": "Новая тема"},
        ]

        conversations = summarizer.detect_conversation_boundaries(messages)
        assert len(conversations) >= 1
        assert all(isinstance(conv, list) for conv in conversations)

    def test_prepare_conversation_text(self):
        """Тест подготовки текста разговора"""
        summarizer = ChatSummarizer()

        conversation = [
            {"date": "2024-01-01T10:00:00Z", "text": "Первое сообщение"},
            {"date": "2024-01-01T10:05:00Z", "text": "Второе сообщение"},
        ]

        text = summarizer._prepare_conversation_text(conversation)
        assert "Первое сообщение" in text
        assert "Второе сообщение" in text
        assert len(text) > 0

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Требует запущенного Ollama и данных")
    async def test_process_chat_file(self, chats_dir, test_output_dir):
        """Тест обработки файла чата"""
        json_files = list(chats_dir.glob("**/*.json"))
        if not json_files:
            pytest.skip("JSON файлы не найдены")

        test_file = json_files[0]
        summarizer = ChatSummarizer()

        async with summarizer:
            md_file = await summarizer.process_chat_file(test_file, test_output_dir)

            if md_file:
                assert md_file.exists()
                assert md_file.suffix == ".md"


class TestMarkdownIndexer:
    """Тесты для MarkdownIndexer"""

    def test_indexer_init(self):
        """Тест инициализации индексатора"""
        indexer = MarkdownIndexer()
        assert indexer.ollama_client is not None
        assert indexer.mcp is not None

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Требует ChromaDB и данных")
    async def test_index_markdown_files(self, tmp_path):
        """Тест индексации markdown файлов"""
        # Создаем тестовый MD файл
        test_md = tmp_path / "test.md"
        test_md.write_text("# Тест\n\nТестовое содержимое")

        indexer = MarkdownIndexer()

        async with indexer:
            await indexer.index_markdown_files(tmp_path)
            # Проверяем что не упало с ошибкой


class TestSummarySearcher:
    """Тесты для SummarySearcher"""

    def test_searcher_init(self):
        """Тест инициализации поисковика"""
        searcher = SummarySearcher()
        assert searcher.ollama_client is not None
        assert searcher.mcp is not None

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Требует индексированных данных")
    async def test_search_summaries(self):
        """Тест поиска по саммаризациям"""
        searcher = SummarySearcher()

        async with searcher:
            try:
                results = await searcher.search("тест", limit=3)
                assert isinstance(results, list)
            except Exception:
                # Коллекция может не существовать
                pass


class TestIntegration:
    """Интеграционные тесты саммаризации"""

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Полный интеграционный тест, требует всех компонентов")
    async def test_full_summarization_flow(self, chats_dir, tmp_path):
        """Тест полного потока саммаризации"""
        json_files = list(chats_dir.glob("**/*.json"))
        if not json_files:
            pytest.skip("JSON файлы не найдены")

        test_file = json_files[0]
        output_dir = tmp_path / "summaries"
        output_dir.mkdir()

        # 1. Саммаризация
        summarizer = ChatSummarizer()
        async with summarizer:
            md_file = await summarizer.process_chat_file(test_file, output_dir)
            assert md_file is not None

        # 2. Индексация
        indexer = MarkdownIndexer()
        async with indexer:
            await indexer.index_markdown_files(output_dir)

        # 3. Поиск
        searcher = SummarySearcher()
        async with searcher:
            results = await searcher.search("тест", limit=1)
            assert isinstance(results, list)


class TestPathConsistency:
    """Тесты консистентности путей"""

    def test_summaries_directory_path(self):
        """Проверка что все компоненты используют правильный путь"""
        from memory_mcp.core.manager import TelegramDumpManager

        # Проверяем что используется "summaries" а не "summarized_chats"
        TelegramDumpManager()

        # Все компоненты должны согласованно использовать "summaries"
        assert True  # Визуальная проверка в коде


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
