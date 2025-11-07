"""Сбор метрик из integration тестов"""
import time
import json
import argparse
from typing import Dict, Any, List
from datetime import datetime

class TestMetricsCollector:
    def __init__(self):
        self.metrics: Dict[str, Any] = {
            "tests_run": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "avg_response_time": {},
            "service_availability": {},
            "test_duration": {},
            "timestamp": datetime.now().isoformat()
        }
    
    def record_test_result(self, test_name: str, passed: bool, duration: float):
        """Запись результата теста"""
        self.metrics["tests_run"] += 1
        if passed:
            self.metrics["tests_passed"] += 1
        else:
            self.metrics["tests_failed"] += 1
        
        self.metrics["test_duration"][test_name] = duration
    
    def record_service_response_time(self, service_name: str, response_time: float):
        """Запись времени отклика сервиса"""
        if service_name not in self.metrics["avg_response_time"]:
            self.metrics["avg_response_time"][service_name] = []
        
        self.metrics["avg_response_time"][service_name].append(response_time)
    
    def record_service_availability(self, service_name: str, is_available: bool):
        """Запись доступности сервиса"""
        self.metrics["service_availability"][service_name] = is_available
    
    def calculate_averages(self):
        """Расчет средних значений"""
        for service, times in self.metrics["avg_response_time"].items():
            if times:
                self.metrics["avg_response_time"][service] = sum(times) / len(times)
    
    def export_prometheus(self) -> str:
        """Экспорт метрик в формате Prometheus"""
        self.calculate_averages()
        
        prometheus_metrics = f"""
# HELP mcp_integration_tests_total Total integration tests run
# TYPE mcp_integration_tests_total counter
mcp_integration_tests_total {self.metrics["tests_run"]}

# HELP mcp_integration_tests_passed Integration tests passed
# TYPE mcp_integration_tests_passed counter
mcp_integration_tests_passed {self.metrics["tests_passed"]}

# HELP mcp_integration_tests_failed Integration tests failed
# TYPE mcp_integration_tests_failed counter
mcp_integration_tests_failed {self.metrics["tests_failed"]}

# HELP mcp_integration_test_success_rate Integration test success rate
# TYPE mcp_integration_test_success_rate gauge
mcp_integration_test_success_rate {self.metrics["tests_passed"] / max(self.metrics["tests_run"], 1)}
"""
        
        # Добавляем метрики времени отклика сервисов
        for service, avg_time in self.metrics["avg_response_time"].items():
            prometheus_metrics += f"""
# HELP mcp_service_response_time_seconds Average response time for service
# TYPE mcp_service_response_time_seconds gauge
mcp_service_response_time_seconds{{service="{service}"}} {avg_time}
"""
        
        # Добавляем метрики доступности сервисов
        for service, is_available in self.metrics["service_availability"].items():
            prometheus_metrics += f"""
# HELP mcp_service_availability Service availability status
# TYPE mcp_service_availability gauge
mcp_service_availability{{service="{service}"}} {1 if is_available else 0}
"""
        
        return prometheus_metrics
    
    def save_report(self, filepath: str):
        """Сохранение отчета в JSON"""
        self.calculate_averages()
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def print_summary(self):
        """Вывод сводки метрик"""
        self.calculate_averages()
        
        print("=== Integration Tests Metrics Summary ===")
        print(f"Tests run: {self.metrics['tests_run']}")
        print(f"Tests passed: {self.metrics['tests_passed']}")
        print(f"Tests failed: {self.metrics['tests_failed']}")
        print(f"Success rate: {self.metrics['tests_passed'] / max(self.metrics['tests_run'], 1):.2%}")
        print()
        
        print("Service Response Times:")
        for service, avg_time in self.metrics["avg_response_time"].items():
            print(f"  {service}: {avg_time:.3f}s")
        print()
        
        print("Service Availability:")
        for service, is_available in self.metrics["service_availability"].items():
            status = "✓" if is_available else "✗"
            print(f"  {service}: {status}")

def main():
    """CLI для работы с метриками"""
    parser = argparse.ArgumentParser(description="Integration tests metrics collector")
    parser.add_argument("--export-prometheus", action="store_true", help="Export metrics in Prometheus format")
    parser.add_argument("--save-report", type=str, help="Save report to JSON file")
    parser.add_argument("--print-summary", action="store_true", help="Print metrics summary")
    
    args = parser.parse_args()
    
    # Создаем коллектор с примерными данными
    collector = TestMetricsCollector()
    
    # Примерные данные для демонстрации
    collector.record_test_result("test_backtesting_full_flow", True, 2.5)
    collector.record_test_result("test_tradingview_price_sync", True, 1.8)
    collector.record_test_result("test_halv1_integration", True, 3.2)
    collector.record_test_result("test_health_cascade", False, 5.0)
    
    collector.record_service_response_time("binance-mcp", 0.5)
    collector.record_service_response_time("tradingview-mcp", 0.8)
    collector.record_service_response_time("memory-mcp", 0.3)
    collector.record_service_response_time("shell-mcp", 1.2)
    collector.record_service_response_time("backtesting-mcp", 2.1)
    
    collector.record_service_availability("binance-mcp", True)
    collector.record_service_availability("tradingview-mcp", True)
    collector.record_service_availability("memory-mcp", True)
    collector.record_service_availability("shell-mcp", True)
    collector.record_service_availability("backtesting-mcp", True)
    
    if args.export_prometheus:
        print(collector.export_prometheus())
    
    if args.save_report:
        collector.save_report(args.save_report)
        print(f"Report saved to {args.save_report}")
    
    if args.print_summary:
        collector.print_summary()

if __name__ == "__main__":
    main()
