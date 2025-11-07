import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


def make_event(
    event_id: str, days_offset: int, value: float, *, frozen: bool = False
) -> dict[str, object]:
    now = datetime.now(timezone.utc)
    ts = now + timedelta(days=days_offset)
    return {
        "id": event_id,
        "timestamp": ts.isoformat(),
        "value": value,
        "frozen": frozen,
    }


def run_cli(
    input_path: Path, output_path: Path, dry_run: bool
) -> subprocess.CompletedProcess[str]:
    args = [
        sys.executable,
        str(Path("scripts") / "consolidate.py"),
        "--k",
        "1",
        "--threshold",
        "5.0",
        "--input",
        str(input_path),
        "--output",
        str(output_path),
    ]
    if dry_run:
        args.append("--dry-run")
    return subprocess.run(args, check=True, text=True)


def test_cli_dry_run_does_not_write(tmp_path: Path) -> None:
    input_path = tmp_path / "events.json"
    output_path = tmp_path / "out.json"
    events = [make_event("0", 0, 1.0)]
    input_path.write_text(json.dumps(events))

    run_cli(input_path, output_path, dry_run=True)

    assert not output_path.exists()


def test_cli_respects_frozen_flag(tmp_path: Path) -> None:
    input_path = tmp_path / "events.json"
    output_path = tmp_path / "out.json"
    events = [
        make_event("frozen", -6, 1.0, frozen=True),
        make_event("low", -6, 1.0),
        make_event("high", 0, 10.0),
    ]
    input_path.write_text(json.dumps(events))

    run_cli(input_path, output_path, dry_run=False)

    data = json.loads(output_path.read_text())
    ids = {item["id"] for item in data}
    assert "frozen" in ids
    assert "low" not in ids
    assert "high" in ids
