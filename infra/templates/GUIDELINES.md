# MCP Server Guidelines

These conventions standardize new MCP servers across this workspace for predictable structure, easy extension, and clean ops.

## Project Layout

- `pyproject.toml` – package metadata, dependencies, entry points
- `src/<package>/server.py` – server entry point (FastMCP preferred)
- `src/<package>/tools/` – tool implementations grouped by domain
- `src/<package>/resources/` – resource providers (if any)
- `src/<package>/services/` – domain/service logic (no MCP specifics)
- `src/<package>/config.py` – config via environment (Pydantic BaseSettings)
- `README.md` – usage, configuration, examples

Optional:
- `tests/` (if you already test in this repo, mirror patterns)
- `Makefile` or `tasks/` for common dev commands

## Technology Choices

- Default to `mcp.server.fastmcp.FastMCP` with `@mcp.tool()` for simplicity.
- Use low-level `mcp.server.Server` only when you need custom protocol control.
- Prefer `uv` for execution and dependency management, or `pip` if your team standardizes on it.

## Tool Design

- Naming: `snake_case` verbs with clear domain prefixes when useful (e.g., `order_create`, `market_ticker`).
- Arguments: define a Pydantic model per tool to enforce types and generate JSON schema.
- Returns: structured JSON (dict/list) for machine consumption; include concise human text if helpful.
- Errors: raise `ValueError` for user errors; catch-and-wrap internal exceptions into friendly messages.
- Idempotency: tools should be safe to retry unless the domain forbids it (document when not).

## Config & Secrets

- Use a `BaseSettings` class (Pydantic v2 `pydantic-settings`) to read env vars.
- Document required env vars in `README.md` and provide `.env.example` where applicable.
- Never log secrets. Log only prefixes where it helps debugging.

## Logging & Observability

- Initialize a module-level logger in `server.py`.
- Log: tool name, arguments (sanitized), latency, and outcome.
- Consider adding a `health` and `version` tool in every server.

## Packaging & Entrypoint

- Expose a console script in `pyproject.toml`:
  - `[project.scripts] <name> = "<package>.server:main"`
- Support both stdio (default) and optional `streamable-http` via CLI flags.

## CLI Conventions

- `--host`, `--port` for HTTP transport (FastMCP supports `transport="streamable-http"`).
- `--print-config` to dump effective configuration and exit.
- `--log-level` to adjust verbosity.

## Testing

- Unit-test services without MCP.
- If adding protocol tests, prefer MCP Inspector flows (`uv run mcp dev ...`) and narrow tests around tools you add.

## Documentation

- Include: quick start, configuration, example tool calls (payloads, outputs), and Claude Desktop config snippet.
- Keep examples copy-pasteable. Show both stdio and HTTP run modes if supported.

## Example Tool Pattern (FastMCP)

```py
from pydantic import BaseModel
from mcp.server.fastmcp import FastMCP

mcp = FastMCP(name="example")

class TickerArgs(BaseModel):
    symbol: str

@mcp.tool()
def market_ticker(args: TickerArgs) -> dict:
    """Get current price for a symbol."""
    price = ...  # call service
    return {"symbol": args.symbol, "price": price}

def main() -> None:
    mcp.run()
```

## When To Use Low-Level Server

- You need custom `list_tools` or `call_tool` dispatching logic
- Streaming patterns outside FastMCP’s convenience
- Full control over transport lifecycle

Mirror patterns from `binance-mcp` when you need this flexibility.
