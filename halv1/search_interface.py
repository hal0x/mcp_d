#!/usr/bin/env python3
"""
Простой веб-интерфейс для поиска по индексу канала "Вселенная Плюс"
Использует Flask для создания веб-приложения
"""

from flask import Flask, render_template, request, jsonify
import json
from datetime import datetime
from pathlib import Path
import sys

# Добавляем текущий каталог в путь для импорта
sys.path.append(str(Path(__file__).parent))

from create_search_index import TelegramChannelSearcher

app = Flask(__name__)

# Инициализация поисковой системы
searcher = None

def init_searcher():
    """Инициализация поисковой системы"""
    global searcher
    try:
        searcher = TelegramChannelSearcher('./db')
        searcher.load_model()
        searcher.load_index()
        return True
    except Exception as e:
        print(f"Ошибка инициализации поисковой системы: {e}")
        return False

@app.route('/')
def index():
    """Главная страница"""
    return render_template('search.html')

@app.route('/search', methods=['POST'])
def search():
    """API для поиска"""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        top_k = int(data.get('top_k', 10))
        
        if not query:
            return jsonify({'error': 'Пустой поисковый запрос'})
            
        if not searcher:
            return jsonify({'error': 'Поисковая система не инициализирована'})
            
        # Выполняем поиск
        results = searcher.search(query, top_k)
        
        # Форматируем результаты
        formatted_results = []
        for result in results:
            formatted_result = {
                'message_id': result['message_id'],
                'text': result['text'],
                'original_text': result['original_text'],
                'date': result['date'],
                'chat': result['chat'],
                'channel': result['channel'],
                'file': result['file'],
                'line': result['line'],
                'score': round(result['score'], 3),
                'preview': result['text'][:200] + '...' if len(result['text']) > 200 else result['text']
            }
            formatted_results.append(formatted_result)
            
        return jsonify({
            'query': query,
            'total_results': len(formatted_results),
            'results': formatted_results
        })
        
    except Exception as e:
        return jsonify({'error': f'Ошибка поиска: {str(e)}'})

@app.route('/search_by_date', methods=['POST'])
def search_by_date():
    """API для поиска по датам"""
    try:
        data = request.get_json()
        start_date = data.get('start_date', '')
        end_date = data.get('end_date', '')
        query = data.get('query', '').strip()
        top_k = int(data.get('top_k', 10))
        
        if not start_date or not end_date:
            return jsonify({'error': 'Укажите диапазон дат'})
            
        if not searcher:
            return jsonify({'error': 'Поисковая система не инициализирована'})
            
        # Выполняем поиск по датам
        results = searcher.search_by_date_range(start_date, end_date, query, top_k)
        
        # Форматируем результаты
        formatted_results = []
        for result in results:
            formatted_result = {
                'message_id': result['message_id'],
                'text': result['text'],
                'original_text': result['original_text'],
                'date': result['date'],
                'chat': result['chat'],
                'channel': result['channel'],
                'file': result['file'],
                'line': result['line'],
                'score': round(result['score'], 3),
                'preview': result['text'][:200] + '...' if len(result['text']) > 200 else result['text']
            }
            formatted_results.append(formatted_result)
            
        return jsonify({
            'start_date': start_date,
            'end_date': end_date,
            'query': query,
            'total_results': len(formatted_results),
            'results': formatted_results
        })
        
    except Exception as e:
        return jsonify({'error': f'Ошибка поиска по датам: {str(e)}'})

@app.route('/stats')
def stats():
    """API для получения статистики индекса"""
    try:
        if not searcher or not searcher.index:
            return jsonify({'error': 'Индекс не загружен'})
            
        # Подключаемся к базе данных для получения статистики
        import sqlite3
        conn = sqlite3.connect(searcher.db_path)
        cursor = conn.cursor()
        
        # Общее количество документов
        cursor.execute('SELECT COUNT(*) FROM documents')
        total_docs = cursor.fetchone()[0]
        
        # Количество документов по датам
        cursor.execute('''
            SELECT DATE(date) as date, COUNT(*) as count 
            FROM documents 
            GROUP BY DATE(date) 
            ORDER BY date DESC 
            LIMIT 10
        ''')
        daily_stats = [{'date': row[0], 'count': row[1]} for row in cursor.fetchall()]
        
        # Статистика по каналам
        cursor.execute('''
            SELECT channel, COUNT(*) as count 
            FROM documents 
            GROUP BY channel 
            ORDER BY count DESC
        ''')
        channel_stats = [{'channel': row[0], 'count': row[1]} for row in cursor.fetchall()]
        
        conn.close()
        
        return jsonify({
            'total_documents': total_docs,
            'index_size': searcher.index.ntotal,
            'embedding_dimension': searcher.index.d,
            'daily_stats': daily_stats,
            'channel_stats': channel_stats
        })
        
    except Exception as e:
        return jsonify({'error': f'Ошибка получения статистики: {str(e)}'})

if __name__ == '__main__':
    # Создаем каталог для шаблонов
    templates_dir = Path(__file__).parent / 'templates'
    templates_dir.mkdir(exist_ok=True)
    
    # Создаем HTML шаблон
    html_template = '''<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Поиск по каналу "Вселенная Плюс"</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
        }
        .search-container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }
        .search-form {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }
        .search-input {
            flex: 1;
            padding: 12px;
            border: 2px solid #ddd;
            border-radius: 5px;
            font-size: 16px;
        }
        .search-input:focus {
            outline: none;
            border-color: #667eea;
        }
        .search-button {
            padding: 12px 24px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
        }
        .search-button:hover {
            background: #5a6fd8;
        }
        .date-search {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            align-items: center;
        }
        .date-input {
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        .results {
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            padding: 20px;
        }
        .result-item {
            border-bottom: 1px solid #eee;
            padding: 15px 0;
        }
        .result-item:last-child {
            border-bottom: none;
        }
        .result-meta {
            color: #666;
            font-size: 14px;
            margin-bottom: 5px;
        }
        .result-text {
            margin: 10px 0;
            line-height: 1.6;
        }
        .result-score {
            background: #e8f4fd;
            color: #1976d2;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: bold;
        }
        .loading {
            text-align: center;
            padding: 20px;
            color: #666;
        }
        .error {
            background: #ffebee;
            color: #c62828;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .stats {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
        }
        .stat-item {
            text-align: center;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 5px;
        }
        .stat-number {
            font-size: 24px;
            font-weight: bold;
            color: #667eea;
        }
        .stat-label {
            color: #666;
            margin-top: 5px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔍 Поиск по всем каналам</h1>
        <p>Семантический поиск по контенту всех каналов в базе данных</p>
    </div>

    <div class="stats" id="stats">
        <h3>📊 Статистика индекса</h3>
        <div class="stats-grid" id="statsGrid">
            <div class="loading">Загрузка статистики...</div>
        </div>
    </div>

    <div class="search-container">
        <h3>🔍 Поиск по тексту</h3>
        <div class="search-form">
            <input type="text" id="searchInput" class="search-input" 
                   placeholder="Введите поисковый запрос (например: SpaceX, бактерии, нейронаука)">
            <button onclick="performSearch()" class="search-button">Поиск</button>
        </div>
        
        <h3>📅 Поиск по датам</h3>
        <div class="date-search">
            <label>С:</label>
            <input type="date" id="startDate" class="date-input">
            <label>По:</label>
            <input type="date" id="endDate" class="date-input">
            <input type="text" id="dateSearchInput" class="search-input" 
                   placeholder="Опциональный текстовый запрос">
            <button onclick="performDateSearch()" class="search-button">Поиск по датам</button>
        </div>
    </div>

    <div class="results" id="results">
        <div style="text-align: center; color: #666; padding: 40px;">
            Введите поисковый запрос для начала поиска
        </div>
    </div>

    <script>
        // Загружаем статистику при загрузке страницы
        loadStats();

        function loadStats() {
            fetch('/stats')
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        document.getElementById('statsGrid').innerHTML = 
                            '<div class="error">Ошибка загрузки статистики: ' + data.error + '</div>';
                        return;
                    }
                    
                    const statsHtml = `
                        <div class="stat-item">
                            <div class="stat-number">${data.total_documents}</div>
                            <div class="stat-label">Документов</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-number">${data.index_size}</div>
                            <div class="stat-label">Векторов в индексе</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-number">${data.embedding_dimension}</div>
                            <div class="stat-label">Размерность эмбеддингов</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-number">${data.channel_stats.length}</div>
                            <div class="stat-label">Каналов</div>
                        </div>
                    `;
                    document.getElementById('statsGrid').innerHTML = statsHtml;
                })
                .catch(error => {
                    document.getElementById('statsGrid').innerHTML = 
                        '<div class="error">Ошибка загрузки статистики: ' + error + '</div>';
                });
        }

        function performSearch() {
            const query = document.getElementById('searchInput').value.trim();
            if (!query) {
                alert('Введите поисковый запрос');
                return;
            }
            
            document.getElementById('results').innerHTML = '<div class="loading">Поиск...</div>';
            
            fetch('/search', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    query: query,
                    top_k: 20
                })
            })
            .then(response => response.json())
            .then(data => {
                displayResults(data);
            })
            .catch(error => {
                document.getElementById('results').innerHTML = 
                    '<div class="error">Ошибка поиска: ' + error + '</div>';
            });
        }

        function performDateSearch() {
            const startDate = document.getElementById('startDate').value;
            const endDate = document.getElementById('endDate').value;
            const query = document.getElementById('dateSearchInput').value.trim();
            
            if (!startDate || !endDate) {
                alert('Укажите диапазон дат');
                return;
            }
            
            document.getElementById('results').innerHTML = '<div class="loading">Поиск по датам...</div>';
            
            fetch('/search_by_date', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    start_date: startDate,
                    end_date: endDate,
                    query: query,
                    top_k: 20
                })
            })
            .then(response => response.json())
            .then(data => {
                displayResults(data);
            })
            .catch(error => {
                document.getElementById('results').innerHTML = 
                    '<div class="error">Ошибка поиска по датам: ' + error + '</div>';
            });
        }

        function displayResults(data) {
            if (data.error) {
                document.getElementById('results').innerHTML = 
                    '<div class="error">Ошибка: ' + data.error + '</div>';
                return;
            }
            
            if (data.results.length === 0) {
                document.getElementById('results').innerHTML = 
                    '<div style="text-align: center; color: #666; padding: 40px;">Результаты не найдены</div>';
                return;
            }
            
            let html = '<h3>Результаты поиска (' + data.total_results + ')</h3>';
            
            data.results.forEach((result, index) => {
                html += `
                    <div class="result-item">
                        <div class="result-meta">
                            <span class="result-score">Сходство: ${result.score}</span>
                            <span style="margin-left: 15px;">📺 ${result.channel}</span>
                            <span style="margin-left: 15px;">📅 ${result.date}</span>
                            <span style="margin-left: 15px;">📁 ${result.file}</span>
                            <span style="margin-left: 15px;">🆔 ${result.message_id}</span>
                        </div>
                        <div class="result-text">${result.preview}</div>
                    </div>
                `;
            });
            
            document.getElementById('results').innerHTML = html;
        }

        // Поиск по Enter
        document.getElementById('searchInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                performSearch();
            }
        });

        document.getElementById('dateSearchInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                performDateSearch();
            }
        });
    </script>
</body>
</html>'''
    
    with open(templates_dir / 'search.html', 'w', encoding='utf-8') as f:
        f.write(html_template)
    
    # Инициализируем поисковую систему
    if init_searcher():
        print("Поисковая система инициализирована успешно!")
        print("Запускаем веб-сервер на http://localhost:5000")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Ошибка инициализации поисковой системы!")
        print("Убедитесь, что индекс создан с помощью create_search_index.py")
