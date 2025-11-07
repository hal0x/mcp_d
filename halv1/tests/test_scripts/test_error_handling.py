import pytest


def test_example_handles_name_error(capsys: pytest.CaptureFixture[str]) -> None:
    try:
        print(undefined_variable)  # type: ignore[name-defined]
    except NameError as e:
        print("Handled error:", e)

    captured = capsys.readouterr()
    assert "Handled error: name 'undefined_variable' is not defined" in captured.out
