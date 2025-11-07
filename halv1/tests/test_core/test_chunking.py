import pathlib
import sys
from datetime import UTC, datetime, timedelta

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

# Import the function under test
from index.chunking import chunk_messages


# Simple token counter used in tests – 4 chars per token (rounded up)
def _token_counter(s: str) -> int:
    return max(1, len(s) // 4)


def _make_msg(
    msg_id: int,
    ts_offset_min: int,
    author: str = "A",
    text_len: int = 50,
    reply_to: int | None = None,
    chat_id: str = "chat1",
) -> dict[str, object]:
    return {
        "chat_id": chat_id,
        "message_id": msg_id,
        "author": author,
        "text": "x" * text_len,
        "timestamp": datetime.now(UTC) + timedelta(minutes=ts_offset_min),
        "reply_to": reply_to,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_basic_merge_by_gap() -> None:
    msgs = [
        _make_msg(1, 0),
        _make_msg(2, 1),
        _make_msg(3, 2),
    ]
    chunks = chunk_messages(msgs, token_counter=_token_counter)
    assert len(chunks) == 1
    assert chunks[0]["message_ids"] == [1, 2, 3]
    assert chunks[0]["authors"] == ["A"]


def test_split_by_max_tokens() -> None:
    # Each message ~200 chars -> 50 tokens. max_chunk_tokens=100.
    msgs = [
        _make_msg(1, 0, text_len=200),
        _make_msg(2, 1, text_len=200),
        _make_msg(3, 2, text_len=200),
    ]
    chunks = chunk_messages(msgs, max_chunk_tokens=100, token_counter=_token_counter)
    assert len(chunks) == 2
    # First two merged, third separate
    assert chunks[0]["message_ids"] == [1, 2]
    assert chunks[1]["message_ids"] == [3]


def test_merge_by_reply_chain() -> None:
    msgs = [
        _make_msg(1, 0),
        _make_msg(2, 8, reply_to=1),  # gap > max_gap but reply_to inside chunk
    ]
    chunks = chunk_messages(msgs, token_counter=_token_counter)
    assert len(chunks) == 1
    assert chunks[0]["message_ids"] == [1, 2]


def test_stop_pattern_breaks() -> None:
    msgs = [
        _make_msg(1, 0),
        _make_msg(2, 1),
        _make_msg(3, 2, text_len=10, reply_to=None),
    ]
    # Override third message to contain stop pattern
    msgs[2]["text"] = "#новая тема: something"
    chunks = chunk_messages(msgs, token_counter=_token_counter)
    assert len(chunks) == 2
    assert chunks[0]["message_ids"] == [1, 2]
    assert chunks[1]["message_ids"] == [3]


def test_same_author_only() -> None:
    msgs = [
        _make_msg(1, 0, author="A"),
        _make_msg(2, 1, author="B"),
    ]
    chunks = chunk_messages(msgs, same_author_only=True, token_counter=_token_counter)
    assert len(chunks) == 2
    assert chunks[0]["message_ids"] == [1]
    assert chunks[1]["message_ids"] == [2]


def test_metadata_ids_range() -> None:
    msgs = [
        _make_msg(10, 0),
        _make_msg(20, 1),
        _make_msg(30, 2),
    ]
    chunks = chunk_messages(msgs, token_counter=_token_counter)
    assert len(chunks) == 1
    assert chunks[0]["message_ids"] == [10, 20, 30]


def test_has_links_multiple_domains() -> None:
    msgs = [
        _make_msg(1, 0),
        _make_msg(2, 1),
    ]
    msgs[0]["text"] = "see https://example.com"
    msgs[1]["text"] = "and https://openai.com"
    chunks = chunk_messages(msgs, token_counter=_token_counter)
    assert len(chunks) == 1
    assert chunks[0]["meta"]["has_links"] is True


def test_duplicates_across_gaps_are_skipped() -> None:
    msgs = [
        _make_msg(1, 0),
        _make_msg(2, 1),
        _make_msg(2, 100),
        _make_msg(3, 101),
    ]
    chunks = chunk_messages(msgs, token_counter=_token_counter)
    assert len(chunks) == 2
    assert chunks[0]["message_ids"] == [1, 2]
    assert chunks[1]["message_ids"] == [3]


def test_duplicates_in_different_chats_are_kept() -> None:
    msgs = [
        _make_msg(1, 0, chat_id="chat1"),
        _make_msg(1, 1, chat_id="chat2"),
    ]
    chunks = chunk_messages(msgs, token_counter=_token_counter)
    assert len(chunks) == 2
    assert chunks[0]["message_ids"] == [1]
    assert chunks[1]["message_ids"] == [1]


def test_duplicate_across_chunks_skipped() -> None:
    msgs = [
        _make_msg(1, 0),
        _make_msg(2, 10),
        _make_msg(1, 20),
    ]
    chunks = chunk_messages(msgs, token_counter=_token_counter)
    assert len(chunks) == 2
    assert chunks[0]["message_ids"] == [1]
    assert chunks[1]["message_ids"] == [2]


def test_has_links_metadata() -> None:
    msgs = [_make_msg(1, 0, text_len=20), _make_msg(2, 1, text_len=20)]
    msgs[0]["text"] = "check https://a.com"
    msgs[1]["text"] = "and https://b.com"
    chunks = chunk_messages(msgs, token_counter=_token_counter)
    assert chunks[0]["meta"]["has_links"] is True


def test_split_by_time_gap() -> None:
    msgs = [_make_msg(1, 0), _make_msg(2, 5)]
    chunks = chunk_messages(msgs, token_counter=_token_counter)
    assert len(chunks) == 2
    assert chunks[0]["message_ids"] == [1]
    assert chunks[1]["message_ids"] == [2]


def test_consecutive_duplicates_skipped() -> None:
    msgs = [_make_msg(1, 0), _make_msg(1, 1), _make_msg(2, 2)]
    chunks = chunk_messages(msgs, token_counter=_token_counter)
    assert len(chunks) == 1
    assert chunks[0]["message_ids"] == [1, 2]


def test_empty_input_returns_empty_list() -> None:
    assert chunk_messages([], token_counter=_token_counter) == []


def test_different_chats_create_new_chunks() -> None:
    m1 = _make_msg(1, 0)
    m2 = _make_msg(2, 1)
    m2["chat_id"] = "chat2"
    chunks = chunk_messages([m1, m2], token_counter=_token_counter)
    assert len(chunks) == 2
    assert chunks[0]["chat_id"] == "chat1"
    assert chunks[1]["chat_id"] == "chat2"
