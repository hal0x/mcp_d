import datetime as dt
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from index.preprocess import merge_short_messages, normalize_text


def test_normalize_text_strips_html_emoji_and_tags() -> None:
    raw = "<b>Hello</b> 🌟 [source: tg]"
    assert normalize_text(raw) == "Hello"


def test_merge_short_messages_within_window() -> None:
    msgs = [
        {"timestamp": dt.datetime(2024, 1, 1, 12, 0, 0), "text": "hi"},
        {"timestamp": dt.datetime(2024, 1, 1, 12, 0, 30), "text": "there"},
        {"timestamp": dt.datetime(2024, 1, 1, 12, 5, 0), "text": "long message"},
    ]
    merged = merge_short_messages(msgs, delta=dt.timedelta(minutes=1), max_length=5)
    assert len(merged) == 2
    assert merged[0]["text"] == "hi there"
    assert merged[1]["text"] == "long message"
