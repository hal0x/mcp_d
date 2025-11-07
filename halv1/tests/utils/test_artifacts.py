import base64

from utils.artifacts import write_artifact_files


def test_write_artifact_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    files = {"data/file.txt": b"hello"}
    encoded = write_artifact_files(files)
    expected = {"data/file.txt": base64.b64encode(b"hello").decode()}
    assert encoded == expected
    assert (tmp_path / "data" / "file.txt").read_bytes() == b"hello"
