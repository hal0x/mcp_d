"""Scraper service for managing Bright Data MCP integration."""

import os
import json
import logging
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
import aiohttp

logger = logging.getLogger(__name__)


class ScraperService:
    """Service for managing web scraping through Bright Data MCP."""
    
    def __init__(self):
        self.bright_data_url = os.getenv("BRIGHT_DATA_MCP_URL", "http://bright-data-mcp:8083")
        self.api_token = os.getenv("BRIGHT_DATA_API_TOKEN")
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._session: Optional[aiohttp.ClientSession] = None
        self._task_history: List[Dict[str, Any]] = []
        
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session
    
    async def close(self):
        """Close HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def scrape_url(
        self, 
        url: str, 
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Scrape a single URL using Bright Data MCP."""
        if not self.api_token:
            raise ValueError("BRIGHT_DATA_API_TOKEN not configured")
        
        # Check cache first
        cache_key = f"{url}:{json.dumps(options or {}, sort_keys=True)}"
        if cache_key in self._cache:
            logger.info(f"Returning cached result for {url}")
            return self._cache[cache_key]
        
        session = await self._get_session()
        task_id = f"scrape_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self._task_history)}"
        
        try:
            # Prepare request payload
            payload = {
                "jsonrpc": "2.0",
                "id": task_id,
                "method": "scrape_url",
                "params": {
                    "url": url,
                    "options": options or {},
                    "api_token": self.api_token
                }
            }
            
            logger.info(f"Scraping URL: {url}")
            
            async with session.post(
                f"{self.bright_data_url}/mcp",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Bright Data MCP error: {response.status} - {error_text}")
                
                result = await response.json()
                
                if "error" in result:
                    raise Exception(f"Bright Data MCP error: {result['error']}")
                
                # Extract scraping result
                scrape_result = result.get("result", {})
                
                # Cache the result
                self._cache[cache_key] = scrape_result
                
                # Add to task history
                self._task_history.append({
                    "task_id": task_id,
                    "url": url,
                    "timestamp": datetime.now().isoformat(),
                    "status": "completed",
                    "result": scrape_result
                })
                
                return scrape_result
                
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            # Add failed task to history
            self._task_history.append({
                "task_id": task_id,
                "url": url,
                "timestamp": datetime.now().isoformat(),
                "status": "failed",
                "error": str(e)
            })
            raise
    
    async def scrape_urls_batch(
        self, 
        urls: List[str], 
        options: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Scrape multiple URLs in batch."""
        if not urls:
            return []
        
        logger.info(f"Starting batch scraping of {len(urls)} URLs")
        
        # Process URLs concurrently with rate limiting
        semaphore = asyncio.Semaphore(5)  # Limit concurrent requests
        
        async def scrape_single(url: str) -> Dict[str, Any]:
            async with semaphore:
                try:
                    return await self.scrape_url(url, options)
                except Exception as e:
                    logger.error(f"Failed to scrape {url}: {e}")
                    return {
                        "url": url,
                        "error": str(e),
                        "status": "failed"
                    }
        
        tasks = [scrape_single(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to error results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "url": urls[i],
                    "error": str(result),
                    "status": "failed"
                })
            else:
                processed_results.append(result)
        
        logger.info(f"Completed batch scraping: {len(processed_results)} results")
        return processed_results
    
    async def scrape_search_results(
        self, 
        query: str, 
        search_engine: str = "google",
        limit: int = 10,
        options: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Scrape search results for a query."""
        if not self.api_token:
            raise ValueError("BRIGHT_DATA_API_TOKEN not configured")
        
        session = await self._get_session()
        task_id = f"search_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self._task_history)}"
        
        try:
            payload = {
                "jsonrpc": "2.0",
                "id": task_id,
                "method": "scrape_search_results",
                "params": {
                    "query": query,
                    "search_engine": search_engine,
                    "limit": limit,
                    "options": options or {},
                    "api_token": self.api_token
                }
            }
            
            logger.info(f"Scraping search results for: {query}")
            
            async with session.post(
                f"{self.bright_data_url}/mcp",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Bright Data MCP error: {response.status} - {error_text}")
                
                result = await response.json()
                
                if "error" in result:
                    raise Exception(f"Bright Data MCP error: {result['error']}")
                
                search_results = result.get("result", [])
                
                # Add to task history
                self._task_history.append({
                    "task_id": task_id,
                    "query": query,
                    "timestamp": datetime.now().isoformat(),
                    "status": "completed",
                    "result_count": len(search_results)
                })
                
                return search_results
                
        except Exception as e:
            logger.error(f"Error scraping search results for {query}: {e}")
            # Add failed task to history
            self._task_history.append({
                "task_id": task_id,
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "status": "failed",
                "error": str(e)
            })
            raise
    
    def get_scraping_status(self, task_id: Optional[str] = None) -> Dict[str, Any]:
        """Get status of scraping tasks."""
        if task_id:
            # Find specific task
            for task in self._task_history:
                if task["task_id"] == task_id:
                    return {
                        "task_id": task_id,
                        "status": task["status"],
                        "timestamp": task["timestamp"],
                        "error": task.get("error")
                    }
            return {"error": f"Task {task_id} not found"}
        
        # Return overall status
        total_tasks = len(self._task_history)
        completed_tasks = len([t for t in self._task_history if t["status"] == "completed"])
        failed_tasks = len([t for t in self._task_history if t["status"] == "failed"])
        
        return {
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "failed_tasks": failed_tasks,
            "success_rate": completed_tasks / total_tasks if total_tasks > 0 else 0,
            "cache_size": len(self._cache)
        }
    
    def get_scraping_history(
        self, 
        limit: Optional[int] = None,
        status_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get scraping task history."""
        history = self._task_history.copy()
        
        # Filter by status if specified
        if status_filter:
            history = [task for task in history if task["status"] == status_filter]
        
        # Limit results if specified
        if limit:
            history = history[-limit:]  # Get most recent tasks
        
        return history
    
    def clear_cache(self) -> Dict[str, Any]:
        """Clear scraping cache."""
        cache_size = len(self._cache)
        self._cache.clear()
        return {
            "status": "success",
            "cleared_items": cache_size
        }
