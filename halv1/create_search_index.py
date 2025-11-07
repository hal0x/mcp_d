#!/usr/bin/env python3
"""
Скрипт для создания поискового индекса из JSON файлов канала "Вселенная Плюс"
Поддерживает семантический поиск через нейронные сети
"""

import json
import os
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from pathlib import Path

class TelegramChannelIndexer:
    def __init__(self, db_dir: str = "./db", model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Инициализация индексатора
        
        Args:
            db_dir: Путь к каталогу с папками каналов
            model_name: Название модели для создания эмбеддингов
        """
        self.db_dir = Path(db_dir)
        self.model_name = model_name
        self.model = None
        self.index = None
        self.documents = []
        self.metadata = []
        
        # Создаем базу данных для метаданных в корне проекта
        self.db_path = Path("search_index.db")
        self.embeddings_path = Path("embeddings.pkl")
        self.faiss_index_path = Path("faiss_index.bin")
        
    def load_model(self):
        """Загрузка модели для создания эмбеддингов"""
        print(f"Загружаем модель {self.model_name}...")
        self.model = SentenceTransformer(self.model_name)
        print("Модель загружена успешно!")
        
    def clean_text(self, text: str) -> str:
        """Очистка и нормализация текста"""
        if not text:
            return ""
            
        # Убираем markdown разметку
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Жирный текст
        text = re.sub(r'\*(.*?)\*', r'\1', text)     # Курсив
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)  # Ссылки
        
        # Убираем лишние пробелы и переносы строк
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Убираем упоминания канала
        text = re.sub(r'@\w+', '', text)
        
        return text.strip()
        
    def extract_content_from_json(self, json_file: Path, channel_name: str = None) -> List[Dict[str, Any]]:
        """Извлечение контента из JSON файла"""
        documents = []
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                        
                    try:
                        data = json.loads(line)
                        
                        # Пропускаем пустые сообщения
                        if not data.get('text', '').strip():
                            continue
                            
                        # Очищаем текст
                        clean_text = self.clean_text(data['text'])
                        if len(clean_text) < 10:  # Пропускаем слишком короткие сообщения
                            continue
                            
                        # Определяем название канала
                        chat_name = data.get('chat', channel_name or json_file.parent.name)
                        
                        document = {
                            'id': data.get('id'),
                            'text': clean_text,
                            'original_text': data.get('text', ''),
                            'date': data.get('date'),
                            'chat': chat_name,
                            'channel': channel_name or json_file.parent.name,
                            'file': json_file.name,
                            'line': line_num
                        }
                        
                        documents.append(document)
                        
                    except json.JSONDecodeError as e:
                        print(f"Ошибка парсинга JSON в {json_file}:{line_num}: {e}")
                        continue
                        
        except Exception as e:
            print(f"Ошибка чтения файла {json_file}: {e}")
            
        return documents
        
    def load_all_documents(self) -> List[Dict[str, Any]]:
        """Загрузка всех документов из JSON файлов во всех папках каналов"""
        all_documents = []
        
        # Находим все папки каналов в db/raw
        raw_dir = self.db_dir / "raw"
        if not raw_dir.exists():
            print(f"Каталог {raw_dir} не найден!")
            return []
            
        channel_dirs = [d for d in raw_dir.iterdir() if d.is_dir()]
        print(f"Найдено {len(channel_dirs)} папок каналов")
        
        for channel_dir in channel_dirs:
            print(f"\nОбрабатываем канал: {channel_dir.name}")
            
            # Находим все JSON файлы в папке канала
            json_files = list(channel_dir.glob("*.json"))
            print(f"  Найдено {len(json_files)} JSON файлов")
            
            channel_documents = 0
            for json_file in json_files:
                documents = self.extract_content_from_json(json_file, channel_dir.name)
                all_documents.extend(documents)
                channel_documents += len(documents)
                
            print(f"  Извлечено {channel_documents} документов из канала {channel_dir.name}")
            
        print(f"\nВсего документов из всех каналов: {len(all_documents)}")
        return all_documents
        
    def create_embeddings(self, documents: List[Dict[str, Any]]) -> np.ndarray:
        """Создание эмбеддингов для документов"""
        if not self.model:
            self.load_model()
            
        texts = [doc['text'] for doc in documents]
        print(f"Создаем эмбеддинги для {len(texts)} документов...")
        
        # Создаем эмбеддинги батчами для экономии памяти
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(batch_texts, show_progress_bar=True)
            all_embeddings.append(batch_embeddings)
            
        embeddings = np.vstack(all_embeddings)
        print(f"Создано эмбеддингов размерности: {embeddings.shape}")
        
        return embeddings
        
    def create_faiss_index(self, embeddings: np.ndarray):
        """Создание FAISS индекса для быстрого поиска"""
        dimension = embeddings.shape[1]
        
        # Создаем индекс с L2 расстоянием
        self.index = faiss.IndexFlatL2(dimension)
        
        # Нормализуем эмбеддинги для косинусного сходства
        faiss.normalize_L2(embeddings)
        
        # Добавляем эмбеддинги в индекс
        self.index.add(embeddings.astype('float32'))
        
        print(f"FAISS индекс создан с {self.index.ntotal} векторами")
        
    def save_to_database(self, documents: List[Dict[str, Any]], embeddings: np.ndarray):
        """Сохранение метаданных в SQLite базу данных"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Создаем таблицу
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message_id INTEGER,
                text TEXT,
                original_text TEXT,
                date TEXT,
                chat TEXT,
                channel TEXT,
                file TEXT,
                line INTEGER,
                embedding_id INTEGER
            )
        ''')
        
        # Создаем индекс для быстрого поиска
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_text ON documents(text)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_date ON documents(date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_chat ON documents(chat)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_channel ON documents(channel)')
        
        # Вставляем данные
        for i, doc in enumerate(documents):
            cursor.execute('''
                INSERT INTO documents 
                (message_id, text, original_text, date, chat, channel, file, line, embedding_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                doc['id'],
                doc['text'],
                doc['original_text'],
                doc['date'],
                doc['chat'],
                doc['channel'],
                doc['file'],
                doc['line'],
                i
            ))
            
        conn.commit()
        conn.close()
        
        print(f"Метаданные сохранены в базу данных: {self.db_path}")
        
    def save_embeddings(self, embeddings: np.ndarray):
        """Сохранение эмбеддингов в файл"""
        with open(self.embeddings_path, 'wb') as f:
            pickle.dump(embeddings, f)
        print(f"Эмбеддинги сохранены: {self.embeddings_path}")
        
    def save_faiss_index(self):
        """Сохранение FAISS индекса"""
        if self.index:
            faiss.write_index(self.index, str(self.faiss_index_path))
            print(f"FAISS индекс сохранен: {self.faiss_index_path}")
            
    def build_index(self):
        """Основной метод для создания индекса"""
        print("Начинаем создание поискового индекса...")
        
        # 1. Загружаем все документы
        documents = self.load_all_documents()
        if not documents:
            print("Не найдено документов для индексации!")
            return
            
        # 2. Создаем эмбеддинги
        embeddings = self.create_embeddings(documents)
        
        # 3. Создаем FAISS индекс
        self.create_faiss_index(embeddings)
        
        # 4. Сохраняем все данные
        self.save_to_database(documents, embeddings)
        self.save_embeddings(embeddings)
        self.save_faiss_index()
        
        print("Индекс создан успешно!")
        print(f"Статистика:")
        print(f"  - Документов: {len(documents)}")
        print(f"  - Размерность эмбеддингов: {embeddings.shape[1]}")
        print(f"  - База данных: {self.db_path}")
        print(f"  - Эмбеддинги: {self.embeddings_path}")
        print(f"  - FAISS индекс: {self.faiss_index_path}")


class TelegramChannelSearcher:
    def __init__(self, db_dir: str = "./db", model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """Инициализация поисковой системы"""
        self.db_dir = Path(db_dir)
        self.model_name = model_name
        self.model = None
        self.index = None
        
        # Пути к файлам в корне проекта
        self.db_path = Path("search_index.db")
        self.embeddings_path = Path("embeddings.pkl")
        self.faiss_index_path = Path("faiss_index.bin")
        
    def load_model(self):
        """Загрузка модели"""
        if not self.model:
            print(f"Загружаем модель {self.model_name}...")
            self.model = SentenceTransformer(self.model_name)
            
    def load_index(self):
        """Загрузка индекса"""
        if not self.index and self.faiss_index_path.exists():
            print("Загружаем FAISS индекс...")
            self.index = faiss.read_index(str(self.faiss_index_path))
            
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Поиск по запросу"""
        if not self.model:
            self.load_model()
        if not self.index:
            self.load_index()
            
        if not self.index:
            print("Индекс не найден! Сначала создайте индекс.")
            return []
            
        # Создаем эмбеддинг для запроса
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Поиск в FAISS индексе
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        # Получаем метаданные из базы данных
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # Пустой результат
                continue
                
            cursor.execute('''
                SELECT message_id, text, original_text, date, chat, channel, file, line
                FROM documents WHERE embedding_id = ?
            ''', (int(idx),))
            
            row = cursor.fetchone()
            if row:
                result = {
                    'message_id': row[0],
                    'text': row[1],
                    'original_text': row[2],
                    'date': row[3],
                    'chat': row[4],
                    'channel': row[5],
                    'file': row[6],
                    'line': row[7],
                    'score': float(score)
                }
                results.append(result)
                
        conn.close()
        return results
        
    def search_by_date_range(self, start_date: str, end_date: str, query: str = None, top_k: int = 10) -> List[Dict[str, Any]]:
        """Поиск по диапазону дат с опциональным текстовым запросом"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Поиск по датам
        cursor.execute('''
            SELECT message_id, text, original_text, date, chat, channel, file, line, embedding_id
            FROM documents 
            WHERE date BETWEEN ? AND ?
            ORDER BY date DESC
        ''', (start_date, end_date))
        
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return []
            
        # Если есть текстовый запрос, фильтруем результаты
        if query:
            if not self.model:
                self.load_model()
            if not self.index:
                self.load_index()
                
            # Создаем эмбеддинг для запроса
            query_embedding = self.model.encode([query])
            faiss.normalize_L2(query_embedding)
            
            # Поиск среди найденных документов
            filtered_results = []
            for row in rows:
                embedding_id = row[7]
                if embedding_id is not None:
                    # Получаем эмбеддинг документа
                    doc_embedding = np.array([self.index.reconstruct(embedding_id)])
                    faiss.normalize_L2(doc_embedding)
                    
                    # Вычисляем сходство
                    similarity = np.dot(query_embedding[0], doc_embedding[0])
                    
                    if similarity > 0.3:  # Порог сходства
                        result = {
                            'message_id': row[0],
                            'text': row[1],
                            'original_text': row[2],
                            'date': row[3],
                            'chat': row[4],
                            'channel': row[5],
                            'file': row[6],
                            'line': row[7],
                            'score': float(similarity)
                        }
                        filtered_results.append(result)
                        
            # Сортируем по сходству
            filtered_results.sort(key=lambda x: x['score'], reverse=True)
            return filtered_results[:top_k]
        else:
            # Возвращаем все результаты по датам
            results = []
            for row in rows:
                result = {
                    'message_id': row[0],
                    'text': row[1],
                    'original_text': row[2],
                    'date': row[3],
                    'chat': row[4],
                    'channel': row[5],
                    'file': row[6],
                    'line': row[7],
                    'score': 1.0
                }
                results.append(result)
            return results[:top_k]


def main():
    """Основная функция"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Создание поискового индекса для всех каналов в ./db')
    parser.add_argument('--db-dir', default='./db', help='Путь к каталогу с папками каналов')
    parser.add_argument('--model', default='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', 
                       help='Модель для создания эмбеддингов')
    parser.add_argument('--action', choices=['index', 'search'], default='index',
                       help='Действие: создание индекса или поиск')
    parser.add_argument('--query', help='Поисковый запрос')
    parser.add_argument('--top-k', type=int, default=10, help='Количество результатов')
    
    args = parser.parse_args()
    
    if args.action == 'index':
        # Создаем индекс
        indexer = TelegramChannelIndexer(args.db_dir, args.model)
        indexer.build_index()
        
    elif args.action == 'search':
        if not args.query:
            print("Укажите поисковый запрос с помощью --query")
            return
            
        # Выполняем поиск
        searcher = TelegramChannelSearcher(args.db_dir, args.model)
        results = searcher.search(args.query, args.top_k)
        
        print(f"\nРезультаты поиска для запроса: '{args.query}'")
        print("=" * 50)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Сообщение ID: {result['message_id']}")
            print(f"   Канал: {result['channel']}")
            print(f"   Дата: {result['date']}")
            print(f"   Файл: {result['file']}")
            print(f"   Сходство: {result['score']:.3f}")
            print(f"   Текст: {result['text'][:200]}...")
            print("-" * 30)


if __name__ == "__main__":
    main()
