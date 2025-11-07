"""Кастомные assertion helpers"""
import pytest
from typing import Dict, Any, List

class MCPAssertions:
    """Вспомогательные функции для проверок в тестах"""
    
    @staticmethod
    def assert_valid_mcp_response(response: Dict[str, Any], expected_keys: List[str] = None):
        """Проверка валидности ответа MCP сервиса"""
        assert isinstance(response, dict), "Response should be a dictionary"
        
        if expected_keys:
            for key in expected_keys:
                assert key in response, f"Response should contain key '{key}'"
    
    @staticmethod
    def assert_price_data(price_data: Dict[str, Any], symbol: str):
        """Проверка данных о цене"""
        assert "symbol" in price_data, "Price data should contain symbol"
        assert price_data["symbol"] == symbol, f"Symbol should be {symbol}"
        assert "price" in price_data, "Price data should contain price"
        
        price = float(price_data["price"])
        assert price > 0, "Price should be positive"
    
    @staticmethod
    def assert_backtest_metrics(metrics: Dict[str, Any]):
        """Проверка метрик бэктестинга"""
        required_metrics = ["total_trades", "profit_factor", "win_rate"]
        
        for metric in required_metrics:
            assert metric in metrics, f"Metrics should contain {metric}"
        
        assert metrics["total_trades"] >= 0, "Total trades should be non-negative"
        assert metrics["profit_factor"] >= 0, "Profit factor should be non-negative"
        assert 0 <= metrics["win_rate"] <= 1, "Win rate should be between 0 and 1"
    
    @staticmethod
    def assert_alert_data(alert: Dict[str, Any]):
        """Проверка данных алерта"""
        required_fields = ["id", "symbol", "condition", "message"]
        
        for field in required_fields:
            assert field in alert, f"Alert should contain {field}"
        
        assert alert["symbol"].endswith("USDT"), "Symbol should end with USDT"
        assert len(alert["condition"]) > 0, "Condition should not be empty"
        assert len(alert["message"]) > 0, "Message should not be empty"
    
    @staticmethod
    def assert_memory_entity(entity: Dict[str, Any], entity_type: str):
        """Проверка сущности в memory"""
        assert "entity_type" in entity, "Entity should contain entity_type"
        assert entity["entity_type"] == entity_type, f"Entity type should be {entity_type}"
        assert "entity_id" in entity, "Entity should contain entity_id"
        assert "data" in entity, "Entity should contain data"
    
    @staticmethod
    def assert_service_health(health_status: bool, service_name: str):
        """Проверка здоровья сервиса"""
        assert health_status, f"{service_name} should be healthy"
