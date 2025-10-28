#!/usr/bin/env python3
"""Скрипт для добавления operation_id ко всем FastAPI endpoints."""

import re

# Читаем файл
with open('src/api.py', 'r') as f:
    lines = f.readlines()

new_lines = []
i = 0
while i < len(lines):
    line = lines[i]
    
    # Ищем строки с декораторами @app.get/post/delete
    if re.match(r'^\s*@app\.(get|post|delete)\("([^"]+)"\),?\s*$', line):
        decorator_type = re.search(r'@app\.(get|post|delete)', line).group(1)
        path = re.search(r'@app\.(get|post|delete)\("([^"]+)"\)', line).group(2)
        
        # Проверяем есть ли уже operation_id
        if 'operation_id' not in line:
            # Генерируем красивое имя из пути
            operation_id = path.replace('/', '_').replace('-', '_').strip('_')
            
            # Улучшаем имена
            operation_id = operation_id.replace('pro_scanner_profiles', 'list_profiles')
            operation_id = operation_id.replace('pro_momentum_scan', 'scan_momentum')
            operation_id = operation_id.replace('pro_mean_revert_scan', 'scan_mean_revert')
            operation_id = operation_id.replace('pro_breakout_scan', 'scan_breakout')
            operation_id = operation_id.replace('pro_volume', 'scan_volume_profile')
            operation_id = operation_id.replace('pro_backtest', 'run_backtest')
            operation_id = operation_id.replace('pro_metrics', 'get_scanner_metrics')
            operation_id = operation_id.replace('pro_snapshot', 'get_metrics_snapshot')
            operation_id = operation_id.replace('pro_signals', 'get_recent_signals')
            operation_id = operation_id.replace('pro_feedback', 'submit_feedback')
            operation_id = operation_id.replace('pro_cache', 'clear_cache')
            operation_id = operation_id.replace('pro_results', 'get_recent_results')
            operation_id = operation_id.replace('pro_status', 'get_scheduler_status')
            operation_id = operation_id.replace('derivatives', 'get_derivatives_context')
            operation_id = operation_id.replace('top_gainers', 'get_top_gainers')
            operation_id = operation_id.replace('multi_changes', 'get_multi_changes')
            operation_id = operation_id.replace('top_losers', 'get_top_losers')
            operation_id = operation_id.replace('bollinger_batch', 'scan_bollinger')
            operation_id = operation_id.replace('coin_batch', 'analyze_coins')
            operation_id = operation_id.replace('candles_batch', 'scan_candles')
            operation_id = operation_id.replace('patterns_batch', 'scan_patterns')
            operation_id = operation_id.replace('trend_breakout', 'scan_trend_breakout')
            operation_id = operation_id.replace('pullback', 'scan_pullback')
            operation_id = operation_id.replace('unified', 'scan_unified')
            operation_id = operation_id.replace('strategy', 'find_strategy_candidates')
            operation_id = operation_id.replace('scan_strategy', 'scan_strategy')
            operation_id = operation_id.replace('smart', 'scan_smart')
            
            # Заменяем текущую строку
            # Исправляем путь если он уже начинается со слэша
            clean_path = path if path.startswith('/') else '/' + path
            new_line = f'    @app.{decorator_type}("{clean_path}", operation_id="{operation_id}")\n'
            new_lines.append(new_line)
        else:
            new_lines.append(line)
    else:
        new_lines.append(line)
    
    i += 1

# Записываем измененный файл
with open('src/api.py', 'w') as f:
    f.writelines(new_lines)

print('✅ Добавлены operation_id ко всем endpoints')

