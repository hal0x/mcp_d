#!/usr/bin/env python3
"""
Тест сетевого доступа в Docker executor.
"""

import os
import sys
sys.path.append('.')

import subprocess
import pytest

from executor.docker_executor import DockerExecutor

def test_network_access():
    """Тестирует сетевой доступ в Docker executor."""
    print("🌐 Тестирование сетевого доступа в Docker executor...")
    
    # Проверяем Docker
    try:
        result = subprocess.run(["docker", "info"], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ Docker не доступен")
            pytest.skip("Docker не доступен")
        print("✅ Docker доступен")
    except FileNotFoundError:
        print("❌ Docker не установлен")
        pytest.skip("Docker не установлен")
    
    # Тестируем сеть в контейнере
    test_code = '''
import socket
import urllib.request
import json

try:
    # DNS тест
    ip = socket.gethostbyname("google.com")
    print(f"DNS: google.com -> {ip}")
    
    # HTTP тест
    response = urllib.request.urlopen("http://httpbin.org/ip", timeout=10)
    data = json.loads(response.read().decode())
    print(f"HTTP: {data}")
    
    print("✅ Сеть в Docker executor работает!")
except Exception as e:
    print(f"❌ Ошибка сети в Docker executor: {e}")
'''
    
    try:
        result = subprocess.run([
            "docker", "run", "--rm", "--network", "host",
            "python:3.11-slim", "python", "-c", test_code
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Docker executor может использовать сеть")
            print(result.stdout)
        else:
            print("❌ Docker executor не может использовать сеть")
            print(result.stderr)
            assert False, "Docker executor не может использовать сеть"
    except subprocess.TimeoutExpired:
        print("❌ Docker тест превысил timeout")
        assert False, "Docker тест превысил timeout"
    except Exception as e:
        print(f"❌ Ошибка Docker теста: {e}")
        assert False, f"Ошибка Docker теста: {e}"
    
    print("✅ Сетевой доступ в Docker executor работает")

def main():
    """Основная функция."""
    print("🚀 Тестирование Docker executor с сетевым доступом\n")
    
    # Устанавливаем переменные окружения для тестирования
    os.environ["DOCKER_ALLOW_INTERNET"] = "true"
    os.environ["DOCKER_NETWORK_MODE"] = "host"
    
    success = test_network_access()
    
    print("\n" + "="*50)
    if success:
        print("🎉 Тест пройден! Docker executor может использовать сеть.")
        print("✅ HALv1 готов к работе с интернет-функциями")
    else:
        print("❌ Тест не пройден. Проверьте настройки Docker.")
        print("\n🔧 Рекомендации:")
        print("1. Убедитесь, что Docker запущен")
        print("2. Проверьте права доступа к Docker")
        print("3. Попробуйте запустить: ./run_docker.sh")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
