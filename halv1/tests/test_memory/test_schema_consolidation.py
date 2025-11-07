"""Tests for schema consolidation of recent episodes."""

from datetime import datetime, timedelta, timezone

from memory.schemas import Episode, consolidate


class TestSchemaConsolidation:
    def test_consolidates_last_seven_days_only(self) -> None:
        now = datetime.now(timezone.utc)
        episodes = [
            Episode(id="1", text="buy milk", timestamp=now - timedelta(days=1)),
            Episode(id="2", text="buy milk", timestamp=now - timedelta(days=2)),
            Episode(id="3", text="buy bread", timestamp=now - timedelta(days=3)),
            # Old event should be ignored
            Episode(id="4", text="buy milk", timestamp=now - timedelta(days=8)),
        ]

        result = consolidate(episodes, now=now)

        # Two schemas: one for milk, one for bread
        assert len(result.schemas) == 2

        milk_schema = next(s for s in result.schemas if s.schema_summary == "buy milk")
        assert {e.id for e in milk_schema.episodes} == {"1", "2"}

        bread_schema = next(
            s for s in result.schemas if s.schema_summary == "buy bread"
        )
        assert [e.id for e in bread_schema.episodes] == ["3"]

        # Old episode not present in mapping
        assert "4" not in result.episode_to_schema

    def test_groups_similar_texts_via_embeddings(self) -> None:
        now = datetime.now(timezone.utc)

        episodes = [
            Episode(id="1", text="buy milk", timestamp=now - timedelta(hours=1)),
            Episode(id="2", text="purchase milk", timestamp=now - timedelta(hours=2)),
            Episode(id="3", text="buy bread", timestamp=now - timedelta(hours=3)),
        ]

        def dummy_embed(text: str) -> list[float]:
            return [1.0, 0.0] if "milk" in text else [0.0, 1.0]

        result = consolidate(episodes, now=now, embed=dummy_embed, threshold=0.8)

        assert len(result.schemas) == 2

        milk_schema = next(s for s in result.schemas if "milk" in s.schema_summary)
        assert {e.id for e in milk_schema.episodes} == {"1", "2"}

        bread_schema = next(s for s in result.schemas if "bread" in s.schema_summary)
        assert {e.id for e in bread_schema.episodes} == {"3"}
