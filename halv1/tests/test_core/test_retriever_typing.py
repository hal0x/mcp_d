from mypy import api


def test_retriever_type_checking() -> None:
    stdout, stderr, _ = api.run(["--follow-imports=skip", "retriever/retriever.py"])
    # Ensure the protocol provides the `_embed` attribute without mypy errors
    assert "attr-defined" not in stdout + stderr, stdout + stderr
