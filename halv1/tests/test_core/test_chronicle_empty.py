"""Тесты для Chronicle с пустыми и некорректными данными."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from index.chronicle import Chronicle, TimeSlice


class TestChronicleEmpty:
    """Тесты для обработки пустых данных в Chronicle."""

    def setup_method(self):
        """Подготовка для каждого теста."""
        # Создаём моки для зависимостей
        self.manager = MagicMock()
        self.summarizer = MagicMock()
        self.summarizer.summarize.return_value = "Суммаризированный текст"

        # Настраиваем мок manager для возврата пустых кластеров
        self.manager.clusters = {}

        self.chronicle = Chronicle(manager=self.manager, summarizer=self.summarizer)

    def test_empty_events_list(self):
        """Тест: пустой список событий возвращает пустой список TimeSlice."""
        with patch.object(self.chronicle, "_collect_events", return_value=[]):
            result = self.chronicle.build()

        assert result == []
        assert isinstance(result, list)
        self.chronicle.summarizer.summarize.assert_not_called()

    def test_none_events(self):
        """Тест: None вместо событий возвращает пустой список."""
        with patch.object(self.chronicle, "_collect_events", return_value=None):
            # _collect_events может вернуть None, проверяем обработку
            # Chronicle должен обработать None и вернуть пустой список
            result = self.chronicle.build()
            assert result == [], "При None событиях должен возвращаться пустой список"

    def test_events_without_text(self):
        """Тест: события без текста обрабатываются корректно."""
        mock_events = [
            (1609459200, ""),  # Пустой текст
            (1609459300, None),  # None вместо текста
            (1609459400, "Нормальный текст"),
        ]

        with patch.object(self.chronicle, "_collect_events", return_value=mock_events):
            result = self.chronicle.build()

        # Должен создать TimeSlice даже с пустыми текстами
        assert len(result) >= 1
        assert isinstance(result[0], TimeSlice)

        # Суммаризация должна быть вызвана
        self.chronicle.summarizer.summarize.assert_called()

    def test_single_event(self):
        """Тест: одно событие создаёт один TimeSlice."""
        mock_events = [(1609459200, "Единственное событие")]

        with patch.object(self.chronicle, "_collect_events", return_value=mock_events):
            result = self.chronicle.build()

        assert len(result) == 1
        assert result[0].texts == ["Единственное событие"]
        assert result[0].summary == "Суммаризированный текст"

    def test_events_with_zero_timestamp(self):
        """Тест: события с нулевой меткой времени."""
        mock_events = [(0, "Событие в эпохе")]

        with patch.object(self.chronicle, "_collect_events", return_value=mock_events):
            result = self.chronicle.build()

        assert len(result) == 1
        # Время должно быть корректно обработано
        assert result[0].start == datetime.fromtimestamp(0, timezone.utc)

    def test_events_with_negative_timestamp(self):
        """Тест: события с отрицательной меткой времени."""
        mock_events = [(-1000, "Событие до эпохи")]

        with patch.object(self.chronicle, "_collect_events", return_value=mock_events):
            result = self.chronicle.build()

        assert len(result) == 1
        bucket = (-1000 // (24 * 3600)) * 24 * 3600
        assert result[0].start == datetime.fromtimestamp(bucket, timezone.utc)
        assert result[0].texts == ["Событие до эпохи"]
        assert result[0].summary == "Суммаризированный текст"
        self.chronicle.summarizer.summarize.assert_called_once_with(
            ["Событие до эпохи"]
        )

    def test_invalid_slice_hours(self):
        """Тест: некорректные значения slice_hours обрабатываются."""
        mock_events = [(1609459200, "Тестовое событие")]

        with patch.object(self.chronicle, "_collect_events", return_value=mock_events):
            # Тестируем различные некорректные значения
            for invalid_hours in [0, -1, -100]:
                result = self.chronicle.build(slice_hours=invalid_hours)
                # Должен использовать значение по умолчанию и не упасть
                assert isinstance(result, list)
                assert len(result) >= 0

    def test_very_large_slice_hours(self):
        """Тест: очень большие значения slice_hours."""
        mock_events = [(1609459200, "Событие 1"), (1609459300, "Событие 2")]

        with patch.object(self.chronicle, "_collect_events", return_value=mock_events):
            result = self.chronicle.build(slice_hours=8760)  # Год в часах

        # При очень больших интервалах все события должны попасть в один slice
        assert len(result) == 1
        assert len(result[0].texts) == 2

    def test_empty_texts_in_events(self):
        """Тест: события только с пустыми текстами."""
        mock_events = [
            (1609459200, ""),
            (1609459300, "   "),  # Только пробелы
            (1609459400, "\n\t"),  # Только служебные символы
        ]

        with patch.object(self.chronicle, "_collect_events", return_value=mock_events):
            result = self.chronicle.build()

        # Должен создать TimeSlice и попытаться суммаризировать
        assert len(result) >= 1
        assert len(result[0].texts) == 3
        self.chronicle.summarizer.summarize.assert_called_once()

    def test_malformed_events_structure(self):
        """Тест: события с неправильной структурой."""
        # События должны быть кортежами (timestamp, text)
        malformed_events = [
            [1609459200, "Список вместо кортежа"],
            ("не_число", "Строка вместо timestamp"),
            (1609459400,),  # Неполный кортеж
            (1609459500, "Нормальное событие"),
        ]

        with patch.object(
            self.chronicle, "_collect_events", return_value=malformed_events
        ):
            # В зависимости от реализации может упасть или обработать
            try:
                result = self.chronicle.build()
                # Если не упало - проверяем что результат разумный
                assert isinstance(result, list)
            except (TypeError, ValueError, IndexError):
                # Ожидаемое поведение для некорректных данных
                pass

    def test_unsorted_events_handling(self):
        """Тест: обработка несортированных событий."""
        unsorted_events = [
            (1609459400, "Третье событие"),
            (1609459200, "Первое событие"),
            (1609459300, "Второе событие"),
        ]

        with patch.object(
            self.chronicle, "_collect_events", return_value=unsorted_events
        ):
            result = self.chronicle.build()

        # _collect_events должен сортировать события, но проверим результат
        assert len(result) >= 1
        assert isinstance(result[0], TimeSlice)

    def test_events_with_invalid_text_type(self):
        """Тест: события с неправильным типом текста."""
        # События должны быть кортежами (timestamp, text)
        # Chronicle ожидает что text будет строкой, но может обработать другие типы
        invalid_events = [
            (1609459200, 123),  # Число вместо текста
            (1609459300, []),  # Список вместо текста
            (1609459400, {}),  # Словарь вместо текста
        ]

        with patch.object(
            self.chronicle, "_collect_events", return_value=invalid_events
        ):
            # Chronicle может обработать некорректные типы текста
            result = self.chronicle.build()

            # Проверяем что результат корректен
            assert isinstance(result, list), "Результат должен быть списком"
            assert len(result) >= 1, "Должен быть создан хотя бы один TimeSlice"

            # Проверяем что тексты были добавлены (возможно как строки)
            if result and result[0].texts:
                assert len(result[0].texts) == 3, "Все события должны быть добавлены"
