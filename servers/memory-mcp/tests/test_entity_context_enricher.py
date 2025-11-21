"""Unit-тесты для EntityContextEnricher."""

import pytest
from unittest.mock import MagicMock, patch

from memory_mcp.search.entity_context_enricher import EntityContextEnricher
from memory_mcp.analysis.entities.entity_dictionary import EntityDictionary
from memory_mcp.analysis.entities.entity_extraction import EntityExtractor


@pytest.fixture
def mock_entity_dictionary():
    """Создает мок словаря сущностей."""
    dictionary = MagicMock(spec=EntityDictionary)
    dictionary.build_entity_profile.return_value = {
        "entity_type": "persons",
        "value": "Дуров",
        "normalized_value": "дуров",
        "description": "Основатель Telegram",
        "aliases": ["дуров", "pavel durov"],
        "mention_count": 50,
        "chats": ["test_chat"],
        "chat_counts": {"test_chat": 50},
        "first_seen": "2024-01-01T00:00:00",
        "last_seen": "2024-12-01T00:00:00",
        "contexts": [
            {"content": "Дуров объявил о новом функционале", "timestamp": "2024-12-01T00:00:00"}
        ],
        "context_count": 1,
        "related_entities": [
            {"label": "Telegram", "weight": 0.9, "entity_type": "organizations"}
        ],
        "importance": 0.8,
    }
    dictionary.get_entity_description.return_value = "Основатель Telegram"
    dictionary._normalize_entity_value.return_value = "дуров"
    return dictionary


@pytest.fixture
def mock_entity_extractor():
    """Создает мок экстрактора сущностей."""
    extractor = MagicMock(spec=EntityExtractor)
    extractor.extract_entities.return_value = {
        "persons": ["Дуров"],
        "organizations": [],
    }
    return extractor


@pytest.fixture
def mock_graph():
    """Создает мок графа."""
    graph = MagicMock()
    graph.get_neighbors.return_value = [
        ("entity-telegram", {"type": "MENTIONS", "weight": 0.9})
    ]
    graph.graph = MagicMock()
    graph.graph.nodes = {
        "entity-telegram": {"type": "Entity", "label": "Telegram", "entity_type": "organizations"}
    }
    return graph


@pytest.fixture
def entity_enricher(mock_entity_dictionary, mock_entity_extractor, mock_graph):
    """Создает обогатитель контекста сущностей."""
    return EntityContextEnricher(
        entity_dictionary=mock_entity_dictionary,
        entity_extractor=mock_entity_extractor,
        graph=mock_graph,
    )


def test_extract_entities_from_query(entity_enricher):
    """Тест извлечения сущностей из запроса."""
    entities = entity_enricher.extract_entities_from_query("Дуров и Telegram")
    
    assert len(entities) > 0
    assert any(e.get("value") == "Дуров" for e in entities)


def test_enrich_query_with_entity_context(entity_enricher):
    """Тест обогащения запроса контекстом сущностей."""
    query = "Дуров"
    enriched = entity_enricher.enrich_query_with_entity_context(query)
    
    assert enriched != query
    assert "Дуров" in enriched
    assert "контекст" in enriched.lower()


def test_expand_query_with_related_entities(entity_enricher):
    """Тест расширения запроса связанными сущностями."""
    query = "Дуров"
    expanded = entity_enricher.expand_query_with_related_entities(query)
    
    # Запрос должен быть расширен связанными сущностями
    assert len(expanded) >= len(query)


def test_get_related_entities(entity_enricher):
    """Тест получения связанных сущностей."""
    related = entity_enricher.get_related_entities("persons", "Дуров", limit=5)
    
    # Должны быть найдены связанные сущности через граф
    assert isinstance(related, list)


def test_find_semantically_similar_entities(entity_enricher):
    """Тест поиска семантически похожих сущностей."""
    # Мокаем EntityVectorStore и EmbeddingService
    with patch.object(
        entity_enricher, "entity_vector_store", create=True
    ) as mock_store, patch.object(
        entity_enricher, "embedding_service", create=True
    ) as mock_embedding:
        mock_store.available.return_value = True
        mock_embedding.available.return_value = True
        mock_embedding.embed.return_value = [0.1] * 384  # Мок эмбеддинга
        
        from memory_mcp.mcp.schema import EntitySearchResult
        mock_store.search_entities.return_value = [
            EntitySearchResult(
                entity_id="entity-telegram",
                score=0.85,
                entity_type="organizations",
                value="Telegram",
                importance=0.7,
                mention_count=100,
                payload={
                    "entity_type": "organizations",
                    "value": "Telegram",
                    "description": "Мессенджер",
                }
            )
        ]
        
        similar = entity_enricher.find_semantically_similar_entities("мессенджер", limit=5)
        
        assert isinstance(similar, list)


def test_get_entity_profile_summary(entity_enricher):
    """Тест получения сводки профиля сущности."""
    summary = entity_enricher.get_entity_profile_summary("persons", "Дуров")
    
    assert summary is not None
    assert summary.get("entity_type") == "persons"
    assert summary.get("value") == "Дуров"
    assert "description" in summary

