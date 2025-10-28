#!/usr/bin/env python3
"""
Простой HTTP MCP сервер для тестирования.
Универсальный шаблон для любого MCP сервиса.
"""

import logging
import os

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

PORT = int(os.getenv("PORT", "8000"))
HOST = os.getenv("HOST", "0.0.0.0")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
SERVER_NAME = os.getenv("SERVER_NAME", "mcp-server")

# Создаем FastAPI приложение
app = FastAPI(title=SERVER_NAME, version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": f"Welcome to {SERVER_NAME}", "status": "running"}


@app.get("/health")
async def health():
    return {"status": "ok", "server_name": SERVER_NAME}


# Простой MCP endpoint для тестирования
@app.post("/mcp")
async def mcp_endpoint(request: Request):
    """Простой MCP endpoint для тестирования."""
    try:
        body = await request.json()
        method = body.get("method", "")

        logger.info(f"MCP request: {method}")

        if method == "initialize":
            return {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": SERVER_NAME, "version": "1.0.0"},
            }

        elif method == "tools/list":
            return {
                "tools": [
                    {
                        "name": "health",
                        "description": f"Check {SERVER_NAME} health status",
                    },
                    {
                        "name": "version",
                        "description": f"Get {SERVER_NAME} version information",
                    },
                    {
                        "name": "demo_tool",
                        "description": f"Demo tool for {SERVER_NAME}",
                    },
                ]
            }

        elif method == "tools/call":
            tool_name = body.get("params", {}).get("name", "")

            if tool_name == "health":
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Server {SERVER_NAME} is healthy and running",
                        }
                    ]
                }

            elif tool_name == "version":
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Server: {SERVER_NAME}, Version: 1.0.0, Transport: HTTP",
                        }
                    ]
                }

            elif tool_name == "demo_tool":
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": f"This is a demo tool from {SERVER_NAME}. Replace with actual functionality.",
                        }
                    ]
                }

            else:
                return {
                    "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"}
                }

        else:
            return {"error": {"code": -32601, "message": f"Unknown method: {method}"}}

    except Exception as e:
        logger.error(f"MCP endpoint error: {e}")
        return {"error": {"code": -32603, "message": f"Internal error: {str(e)}"}}


if __name__ == "__main__":
    logger.info(f"Starting {SERVER_NAME} HTTP server on {HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT, log_level=LOG_LEVEL.lower())
