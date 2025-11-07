import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from bot.utils import split_long


def test_short_text() -> None:
    assert split_long("hello", 10) == ["hello"]


def test_long_text_simple_split() -> None:
    text = "a" * 3600
    parts = split_long(text, 3500)
    assert parts == ["a" * 3500, "a" * 100]


def test_split_on_newline() -> None:
    text = "a" * 3000 + "\n" + "b" * 3000
    parts = split_long(text, 3500)
    assert parts == ["a" * 3000, "b" * 3000]


def test_split_on_space() -> None:
    text = "a" * 3499 + " " + "b" * 100
    parts = split_long(text, 3500)
    assert parts == ["a" * 3499, "b" * 100]
