#!/usr/bin/env python3
"""Scaffold a new MCP server from the fastmcp-basic template.

Usage:
  python scripts/scaffold_mcp.py <cli_name> [--app "Human App Name"] [--dest ./my-server]

Example:
  python scripts/scaffold_mcp.py jira-mcp --app "Jira MCP Server" --dest ./jira-mcp
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "templates" / "fastmcp-basic"


def copy_and_replace(src: Path, dst: Path, replacements: dict[str, str]) -> None:
    if src.is_dir():
        # Replace __PKG__ in directory names
        rel = src.relative_to(TEMPLATE)
        parts = [replacements.get(p, p) for p in rel.parts]
        dst_dir = dst.joinpath(*parts)
        dst_dir.mkdir(parents=True, exist_ok=True)
        for child in src.iterdir():
            copy_and_replace(child, dst, replacements)
        return

    # File
    rel = src.relative_to(TEMPLATE)
    out_parts = [replacements.get(p, p) for p in rel.parts]
    out_path = dst.joinpath(*out_parts)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    text_exts = {".py", ".md", ".toml", ".txt"}
    if src.suffix in text_exts:
        content = src.read_text(encoding="utf-8")
        for k, v in replacements.items():
            content = content.replace(k, v)
        out_path.write_text(content, encoding="utf-8")
    else:
        shutil.copy2(src, out_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cli", help="CLI and package base name (snake/kebab ok)")
    ap.add_argument("--app", default=None, help="Human readable app name")
    ap.add_argument("--dest", default=None, help="Destination directory (default: ./<cli>)")
    args = ap.parse_args()

    cli_raw = args.cli.strip()
    cli = cli_raw.replace(" ", "-")
    pkg = re.sub(r"[^a-zA-Z0-9_]", "_", cli_raw).lower()
    app = args.app or cli.replace("-", " ").title()

    dest = Path(args.dest or f"./{cli}").resolve()
    if dest.exists() and any(dest.iterdir()):
        raise SystemExit(f"Destination not empty: {dest}")

    replacements = {
        "__CLI__": cli,
        "__PKG__": pkg,
        "__APP_NAME__": app,
    }

    copy_and_replace(TEMPLATE, dest, replacements)
    print(f"Scaffolded MCP server to: {dest}")
    print("Next steps:")
    print(f"  cd {dest}")
    print("  uv sync    # or: pip install -e .")
    print(f"  uv run {cli} --print-config")
    print(f"  uv run mcp dev src/{pkg}/server.py")


if __name__ == "__main__":
    main()

