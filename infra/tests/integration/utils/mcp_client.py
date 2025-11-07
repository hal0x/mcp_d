"""
Обновленный MCP клиент с поддержкой правильного протокола
"""
import httpx
import json
import logging
from typing import Any, Dict, Optional
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class MCPClient:
    """Клиент для работы с MCP серверами через HTTP"""
    
    def __init__(self, base_url: str, service_name: str = "unknown"):
        self.base_url = base_url
        self.service_name = service_name
        self.client: Optional[httpx.AsyncClient] = None
        self.session_id: Optional[str] = None
    
    async def __aenter__(self):
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=30.0,
            follow_redirects=True
        )
        # Пытаемся инициализировать сессию
        try:
            await self._initialize_session()
        except Exception as e:
            logger.warning(f"Failed to initialize session for {self.service_name}: {e}")
        return self
    
    async def __aexit__(self, *args):
        if self.client:
            await self.client.aclose()
    
    async def _initialize_session(self) -> bool:
        """Инициализация MCP сессии"""
        try:
            # Пробуем инициализировать сессию через initialize method
            response = await self.client.post(
                "/mcp",
                json={
                    "jsonrpc": "2.0",
                    "id": 0,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "clientInfo": {
                            "name": "integration-test-client",
                            "version": "1.0.0"
                        }
                    }
                },
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream"
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                if "result" in result:
                    logger.info(f"Session initialized for {self.service_name}")
                    # Извлекаем session ID из headers или response
                    self.session_id = response.headers.get("x-session-id")
                    return True
        except Exception as e:
            logger.debug(f"Session initialization failed: {e}")
        
        return False
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Вызов MCP tool"""
        try:
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream"
            }
            
            if self.session_id:
                headers["x-session-id"] = self.session_id
            
            response = await self.client.post(
                "/mcp",
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {
                        "name": tool_name,
                        "arguments": arguments
                    }
                },
                headers=headers
            )
            
            if response.status_code == 200:
                result = response.json()
                if "result" in result:
                    return result["result"]
                elif "error" in result:
                    raise Exception(f"MCP error: {result['error']}")
                else:
                    return result
            else:
                logger.error(f"HTTP error {response.status_code}: {response.text[:200]}")
                raise Exception(f"HTTP error {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error calling tool {tool_name}: {e}")
            raise
    
    async def list_tools(self) -> list:
        """Получение списка доступных инструментов"""
        try:
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream"
            }
            
            if self.session_id:
                headers["x-session-id"] = self.session_id
            
            response = await self.client.post(
                "/mcp",
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/list"
                },
                headers=headers
            )
            
            if response.status_code == 200:
                result = response.json()
                if "result" in result:
                    return result["result"].get("tools", [])
                elif "error" in result:
                    logger.warning(f"MCP error listing tools: {result['error']}")
                    return []
            
            return []
                
        except Exception as e:
            logger.error(f"Error listing tools: {e}")
            return []
    
    async def health_check(self) -> bool:
        """Проверка доступности сервиса"""
        try:
            # Пробуем простой GET запрос
            response = await self.client.get("/", timeout=5.0)
            # Даже 404 означает, что сервис доступен
            return response.status_code in [200, 404]
        except Exception:
            return False
