# __APP_NAME__

Minimal MCP server scaffold following the workspace guidelines.

## Install

- Create venv and install:

```
uv sync
```

Or with pip:

```
pip install -e .
```

## Run (stdio)

```
uv run __CLI__
```

Debug with MCP Inspector:

```
uv run mcp dev src/__PKG__/server.py
```

## Run (HTTP, optional)

```
uv run __CLI__ --host 127.0.0.1 --port 3001
```

Then configure Claude Desktop transport `streamable-http`.

## Configure

Environment variables are defined in `src/__PKG__/config.py`. Print effective config:

```
uv run __CLI__ --print-config
```

## Tools

- `health` — quick check that services are reachable
- `version` — package version and mode
- `example_echo` — sample tool implemented in `tools/example.py`

## Claude Desktop

Add to `claude_desktop_config.json`:

```
{
  "mcpServers": {
    "__CLI__": {
      "command": "uv",
      "args": ["run", "__CLI__"]
    }
  }
}
```

