from __future__ import annotations

import base64
from pathlib import Path


def write_artifact_files(files: dict[str, bytes]) -> dict[str, str]:
    """Persist ``files`` to disk and return base64-encoded contents.

    Parameters
    ----------
    files:
        Mapping of file paths to their raw byte contents.

    Returns
    -------
    dict[str, str]
        Mapping of file paths to base64-encoded strings.
    """
    encoded: dict[str, str] = {}
    for name, data in files.items():
        path = Path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        encoded[name] = base64.b64encode(data).decode()
    return encoded
