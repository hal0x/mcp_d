import pathlib

from core.utils.json_io import load_json, save_json


def test_load_json_decode_error(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{bad")
    assert load_json(path, {"x": 1}) == {"x": 1}


def test_save_and_load_round_trip(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "data.json"
    data = {"a": [1, 2], "b": {"c": "d"}}
    save_json(path, data, indent=2)
    assert load_json(path, {}) == data
