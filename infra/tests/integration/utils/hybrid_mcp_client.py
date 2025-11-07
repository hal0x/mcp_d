"""
MCP клиент с использованием официального SDK
"""
import asyncio
import json
import logging
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager

# Попробуем импортировать MCP SDK
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.client.sse import sse_client
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("MCP SDK не найден, используем fallback")

import httpx

logger = logging.getLogger(__name__)


class MCPSDKClient:
    """Клиент с использованием официального MCP SDK"""
    
    def __init__(self, base_url: str, service_name: str = "unknown"):
        self.base_url = base_url
        self.service_name = service_name
        self.session: Optional[ClientSession] = None
    
    async def __aenter__(self):
        if not MCP_AVAILABLE:
            raise ImportError("MCP SDK недоступен")
        
        try:
            # Пробуем подключиться через SSE
            self.session = await sse_client(f"{self.base_url}/mcp").__aenter__()
            
            # Инициализируем сессию
            await self.session.initialize()
            
            logger.info(f"MCP SDK session initialized for {self.service_name}")
            return self
            
        except Exception as e:
            logger.error(f"Failed to initialize MCP SDK session for {self.service_name}: {e}")
            raise
    
    async def __aexit__(self, *args):
        if self.session:
            try:
                await self.session.__aexit__(*args)
            except Exception as e:
                logger.error(f"Error closing MCP SDK session: {e}")
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """Получение списка инструментов через MCP SDK"""
        if not self.session:
            return []
        
        try:
            result = await self.session.list_tools()
            return [tool.model_dump() for tool in result.tools]
        except Exception as e:
            logger.error(f"Error listing tools via MCP SDK: {e}")
            return []
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Вызов инструмента через MCP SDK"""
        if not self.session:
            return None
        
        try:
            result = await self.session.call_tool(tool_name, arguments)
            return result.model_dump()
        except Exception as e:
            logger.error(f"Error calling tool {tool_name} via MCP SDK: {e}")
            return None
    
    async def health_check(self) -> bool:
        """Проверка доступности"""
        return self.session is not None


class DirectHTTPClient:
    """Прямой HTTP клиент для тестирования endpoints"""
    
    def __init__(self, base_url: str, service_name: str = "unknown"):
        self.base_url = base_url
        self.service_name = service_name
        self.client: Optional[httpx.AsyncClient] = None
    
    async def __aenter__(self):
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=30.0,
            follow_redirects=True
        )
        return self
    
    async def __aexit__(self, *args):
        if self.client:
            await self.client.aclose()
    
    async def test_direct_endpoints(self) -> Dict[str, Any]:
        """Тестирование прямых HTTP endpoints"""
        results = {}
        
        # Список endpoints для тестирования
        endpoints_to_test = [
            "/health",
            "/healthz", 
            "/meta/health",
            "/status",
            "/ping",
            "/openapi.json",
            "/docs",
            "/market/ticker/price?symbol=BTCUSDT",
            "/market/klines?symbol=BTCUSDT&interval=1h&limit=1",
            "/account/info",
            "/server/time"
        ]
        
        for endpoint in endpoints_to_test:
            try:
                response = await self.client.get(endpoint, timeout=10.0)
                results[endpoint] = {
                    "status_code": response.status_code,
                    "content_type": response.headers.get("content-type", ""),
                    "content_length": len(response.content),
                    "content_preview": response.text[:200] if response.text else ""
                }
                
                if response.status_code == 200:
                    logger.info(f"✅ {endpoint}: {response.status_code}")
                else:
                    logger.info(f"❌ {endpoint}: {response.status_code}")
                    
            except Exception as e:
                results[endpoint] = {
                    "error": str(e),
                    "status_code": None
                }
                logger.info(f"❌ {endpoint}: {e}")
        
        return results
    
    async def health_check(self) -> bool:
        """Проверка доступности"""
        try:
            response = await self.client.get("/", timeout=5.0)
            return response.status_code in [200, 404]
        except Exception:
            return False


class HybridMCPClient:
    """Гибридный клиент, который пробует разные подходы"""
    
    def __init__(self, base_url: str, service_name: str = "unknown"):
        self.base_url = base_url
        self.service_name = service_name
        self.working_client = None
        self.client_type = None
    
    async def __aenter__(self):
        # Пробуем разные подходы в порядке предпочтения
        
        # 1. Пробуем MCP SDK
        if MCP_AVAILABLE:
            try:
                sdk_client = MCPSDKClient(self.base_url, self.service_name)
                await sdk_client.__aenter__()
                self.working_client = sdk_client
                self.client_type = "MCP_SDK"
                logger.info(f"Using MCP SDK for {self.service_name}")
                return self
            except Exception as e:
                logger.info(f"MCP SDK failed for {self.service_name}: {e}")
        
        # 2. Пробуем прямые HTTP endpoints
        try:
            http_client = DirectHTTPClient(self.base_url, self.service_name)
            await http_client.__aenter__()
            
            # Тестируем endpoints
            results = await http_client.test_direct_endpoints()
            working_endpoints = [ep for ep, result in results.items() 
                               if isinstance(result, dict) and result.get("status_code") == 200]
            
            if working_endpoints:
                self.working_client = http_client
                self.client_type = "DIRECT_HTTP"
                logger.info(f"Using Direct HTTP for {self.service_name}, working endpoints: {working_endpoints}")
                return self
            else:
                await http_client.__aexit__(None, None, None)
                
        except Exception as e:
            logger.info(f"Direct HTTP failed for {self.service_name}: {e}")
        
        # 3. Fallback к простому HTTP клиенту
        try:
            from .real_mcp_client import RealMCPClient
            fallback_client = RealMCPClient(self.base_url, self.service_name)
            await fallback_client.__aenter__()
            self.working_client = fallback_client
            self.client_type = "FALLBACK_HTTP"
            logger.info(f"Using Fallback HTTP for {self.service_name}")
            return self
        except Exception as e:
            logger.error(f"All client types failed for {self.service_name}: {e}")
            raise
    
    async def __aexit__(self, *args):
        if self.working_client:
            await self.working_client.__aexit__(*args)
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """Получение списка инструментов"""
        if not self.working_client:
            return []
        
        if hasattr(self.working_client, 'list_tools'):
            return await self.working_client.list_tools()
        return []
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Вызов инструмента"""
        if not self.working_client:
            return None
        
        if hasattr(self.working_client, 'call_tool'):
            return await self.working_client.call_tool(tool_name, arguments)
        return None
    
    async def health_check(self) -> bool:
        """Проверка доступности"""
        if not self.working_client:
            return False
        
        if hasattr(self.working_client, 'health_check'):
            return await self.working_client.health_check()
        return False
    
    def get_client_info(self) -> Dict[str, Any]:
        """Информация о используемом клиенте"""
        return {
            "service_name": self.service_name,
            "client_type": self.client_type,
            "working": self.working_client is not None
        }
