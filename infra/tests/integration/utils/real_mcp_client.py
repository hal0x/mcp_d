"""
Реальный MCP клиент для интеграционных тестов
"""
import asyncio
import json
import logging
from typing import Any, Dict, List, Optional
import httpx
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class RealMCPClient:
    """Клиент для работы с реальными MCP серверами"""
    
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
        
        # Попытка инициализации сессии
        try:
            await self._initialize_session()
        except Exception as e:
            logger.warning(f"Failed to initialize MCP session for {self.service_name}: {e}")
        
        return self
    
    async def __aexit__(self, *args):
        if self.client:
            await self.client.aclose()
    
    async def _initialize_session(self) -> bool:
        """Инициализация MCP сессии"""
        try:
            # Пробуем инициализировать сессию
            response = await self.client.post(
                "/mcp",
                json={
                    "jsonrpc": "2.0",
                    "id": "init-1",
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {
                            "roots": {"listChanged": True},
                            "sampling": {}
                        },
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
                logger.info(f"MCP session initialized for {self.service_name}: {result}")
                
                # Извлекаем session ID из headers или response
                self.session_id = response.headers.get("x-session-id") or response.headers.get("session-id")
                
                # Отправляем initialized notification
                await self.client.post(
                    "/mcp",
                    json={
                        "jsonrpc": "2.0",
                        "method": "notifications/initialized"
                    },
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json, text/event-stream"
                    }
                )
                
                return True
        except Exception as e:
            logger.debug(f"MCP session initialization failed for {self.service_name}: {e}")
        
        return False
    
    async def list_tools(self) -> List[Dict[str, Any]]:
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
                    "id": "list-tools-1",
                    "method": "tools/list"
                },
                headers=headers
            )
            
            if response.status_code == 200:
                result = response.json()
                if "result" in result and "tools" in result["result"]:
                    return result["result"]["tools"]
                elif "error" in result:
                    logger.warning(f"MCP error listing tools for {self.service_name}: {result['error']}")
            else:
                logger.warning(f"HTTP error listing tools for {self.service_name}: {response.status_code} - {response.text[:200]}")
            
            return []
                
        except Exception as e:
            logger.error(f"Error listing tools for {self.service_name}: {e}")
            return []
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Optional[Dict[str, Any]]:
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
                    "id": f"call-{tool_name}-1",
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
                    logger.error(f"MCP error calling {tool_name} on {self.service_name}: {result['error']}")
                    return None
                else:
                    return result
            else:
                logger.error(f"HTTP error calling {tool_name} on {self.service_name}: {response.status_code} - {response.text[:200]}")
                return None
                
        except Exception as e:
            logger.error(f"Error calling tool {tool_name} on {self.service_name}: {e}")
            return None
    
    async def health_check(self) -> bool:
        """Проверка доступности сервиса"""
        try:
            # Простая проверка доступности
            response = await self.client.get("/", timeout=5.0)
            # Даже 404 означает, что сервис доступен
            return response.status_code in [200, 404]
        except Exception:
            return False


class StdioMCPClient:
    """Клиент для работы с MCP сервисами через stdio (внутри Docker)"""
    
    def __init__(self, service_name: str, container_name: str):
        self.service_name = service_name
        self.container_name = container_name
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, *args):
        pass
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Вызов MCP tool через docker exec"""
        try:
            # Создаем JSON-RPC запрос
            request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments
                }
            }
            
            # Выполняем команду через docker exec
            cmd = [
                "docker", "exec", "-i", self.container_name,
                "python", "-c", 
                f"""
import json
import sys
from mcp_server import create_server

# Создаем сервер
server = create_server()

# Отправляем запрос
request = {json.dumps(request)}
print(json.dumps(request))
"""
            ]
            
            # Это упрощенная версия - в реальности нужно использовать subprocess
            # и правильно обрабатывать stdio протокол
            logger.info(f"Would execute: {' '.join(cmd)}")
            
            # Для демонстрации возвращаем None
            return None
            
        except Exception as e:
            logger.error(f"Error calling tool {tool_name} via stdio on {self.service_name}: {e}")
            return None
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """Получение списка инструментов через stdio"""
        # Аналогично call_tool, но для list_tools
        return []
    
    async def health_check(self) -> bool:
        """Проверка доступности через docker ps"""
        try:
            import subprocess
            result = subprocess.run(
                ["docker", "ps", "--filter", f"name={self.container_name}", "--format", "{{.Status}}"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return "Up" in result.stdout
        except Exception:
            return False
