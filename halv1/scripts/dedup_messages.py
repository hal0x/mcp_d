#!/usr/bin/env python3
"""
Скрипт для удаления дубликатов сообщений по полю 'id' в чатах.

Анализирует все JSON файлы в db/raw и удаляет дубликаты сообщений,
оставляя только первое вхождение каждого уникального ID.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Set
import argparse

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MessageDeduplicator:
    """Класс для удаления дубликатов сообщений по полю 'id'."""
    
    def __init__(self, raw_dir: str = "db/raw"):
        self.raw_dir = Path(raw_dir)
        self.stats = {
            'total_files': 0,
            'processed_files': 0,
            'total_messages': 0,
            'duplicates_removed': 0,
            'unique_messages': 0,
            'errors': 0
        }
    
    def deduplicate_chat(self, chat_dir: Path) -> Dict[str, int]:
        """Удаляет дубликаты в одном чате."""
        chat_stats = {
            'files_processed': 0,
            'messages_before': 0,
            'messages_after': 0,
            'duplicates_removed': 0,
            'errors': 0
        }
        
        # Собираем все сообщения из всех файлов чата
        all_messages = []
        json_files = list(chat_dir.glob("*.json"))
        
        if not json_files:
            logger.warning(f"Нет JSON файлов в чате: {chat_dir}")
            return chat_stats
        
        logger.info(f"Обрабатываем чат: {chat_dir.name} ({len(json_files)} файлов)")
        
        # Читаем все сообщения
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        
                        try:
                            message = json.loads(line)
                            if 'id' in message:
                                all_messages.append((json_file, line_num, message))
                                chat_stats['messages_before'] += 1
                        except json.JSONDecodeError as e:
                            logger.warning(f"Ошибка JSON в {json_file}:{line_num}: {e}")
                            chat_stats['errors'] += 1
                
                chat_stats['files_processed'] += 1
                
            except Exception as e:
                logger.error(f"Ошибка чтения файла {json_file}: {e}")
                chat_stats['errors'] += 1
        
        if not all_messages:
            logger.warning(f"Нет сообщений в чате: {chat_dir}")
            return chat_stats
        
        # Находим дубликаты по полю 'id'
        seen_ids: Set[str] = set()
        unique_messages = []
        duplicates_count = 0
        
        for file_path, line_num, message in all_messages:
            message_id = str(message['id'])
            
            if message_id in seen_ids:
                duplicates_count += 1
                logger.debug(f"Дубликат найден: {message_id} в {file_path}:{line_num}")
            else:
                seen_ids.add(message_id)
                unique_messages.append((file_path, line_num, message))
        
        chat_stats['duplicates_removed'] = duplicates_count
        chat_stats['messages_after'] = len(unique_messages)
        
        if duplicates_count > 0:
            logger.info(f"Найдено {duplicates_count} дубликатов в чате {chat_dir.name}")
            
            # Перезаписываем файлы с уникальными сообщениями
            self._rewrite_chat_files(chat_dir, unique_messages)
        else:
            logger.info(f"Дубликаты не найдены в чате {chat_dir.name}")
        
        return chat_stats
    
    def _rewrite_chat_files(self, chat_dir: Path, unique_messages: List[tuple]):
        """Перезаписывает файлы чата с уникальными сообщениями."""
        # Группируем сообщения по файлам
        files_content: Dict[Path, List[Dict]] = {}
        
        for file_path, line_num, message in unique_messages:
            if file_path not in files_content:
                files_content[file_path] = []
            files_content[file_path].append(message)
        
        # Перезаписываем каждый файл
        for file_path, messages in files_content.items():
            try:
                # Создаем резервную копию
                backup_path = file_path.with_suffix('.json.backup')
                if not backup_path.exists():
                    file_path.rename(backup_path)
                
                # Записываем уникальные сообщения
                with open(file_path, 'w', encoding='utf-8') as f:
                    for message in messages:
                        json.dump(message, f, ensure_ascii=False)
                        f.write('\n')
                
                logger.info(f"Перезаписан файл: {file_path} ({len(messages)} сообщений)")
                
            except Exception as e:
                logger.error(f"Ошибка записи файла {file_path}: {e}")
                # Восстанавливаем из резервной копии
                if backup_path.exists():
                    backup_path.rename(file_path)
    
    def deduplicate_all(self, dry_run: bool = False) -> Dict[str, int]:
        """Удаляет дубликаты во всех чатах."""
        if not self.raw_dir.exists():
            logger.error(f"Директория {self.raw_dir} не существует")
            return self.stats
        
        chat_dirs = [d for d in self.raw_dir.iterdir() if d.is_dir()]
        self.stats['total_files'] = len(chat_dirs)
        
        logger.info(f"Найдено {len(chat_dirs)} чатов для обработки")
        
        if dry_run:
            logger.info("РЕЖИМ ТЕСТИРОВАНИЯ - изменения не будут сохранены")
        
        for chat_dir in chat_dirs:
            try:
                chat_stats = self.deduplicate_chat(chat_dir)
                
                # Обновляем общую статистику
                self.stats['processed_files'] += chat_stats['files_processed']
                self.stats['total_messages'] += chat_stats['messages_before']
                self.stats['duplicates_removed'] += chat_stats['duplicates_removed']
                self.stats['unique_messages'] += chat_stats['messages_after']
                self.stats['errors'] += chat_stats['errors']
                
            except Exception as e:
                logger.error(f"Ошибка обработки чата {chat_dir}: {e}")
                self.stats['errors'] += 1
        
        return self.stats
    
    def print_stats(self):
        """Выводит статистику обработки."""
        print("\n" + "="*60)
        print("СТАТИСТИКА ДЕДУПЛИКАЦИИ СООБЩЕНИЙ")
        print("="*60)
        print(f"Обработано чатов: {self.stats['processed_files']}/{self.stats['total_files']}")
        print(f"Всего сообщений: {self.stats['total_messages']}")
        print(f"Уникальных сообщений: {self.stats['unique_messages']}")
        print(f"Дубликатов удалено: {self.stats['duplicates_removed']}")
        print(f"Ошибок: {self.stats['errors']}")
        
        if self.stats['total_messages'] > 0:
            duplicate_percent = (self.stats['duplicates_removed'] / self.stats['total_messages']) * 100
            print(f"Процент дубликатов: {duplicate_percent:.2f}%")
        
        print("="*60)


def main():
    """Основная функция."""
    parser = argparse.ArgumentParser(description='Удаление дубликатов сообщений по полю id')
    parser.add_argument('--raw-dir', default='db/raw', help='Путь к директории с сырыми данными')
    parser.add_argument('--chat', help='Обработать только указанный чат')
    parser.add_argument('--dry-run', action='store_true', help='Режим тестирования без сохранения изменений')
    parser.add_argument('--verbose', '-v', action='store_true', help='Подробный вывод')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    deduplicator = MessageDeduplicator(args.raw_dir)
    
    if args.chat:
        # Обработка одного чата
        chat_dir = Path(args.raw_dir) / args.chat
        if not chat_dir.exists():
            logger.error(f"Чат {args.chat} не найден в {args.raw_dir}")
            return 1
        
        logger.info(f"Обрабатываем чат: {args.chat}")
        chat_stats = deduplicator.deduplicate_chat(chat_dir)
        
        print(f"\nРезультаты для чата '{args.chat}':")
        print(f"Файлов обработано: {chat_stats['files_processed']}")
        print(f"Сообщений до: {chat_stats['messages_before']}")
        print(f"Сообщений после: {chat_stats['messages_after']}")
        print(f"Дубликатов удалено: {chat_stats['duplicates_removed']}")
        print(f"Ошибок: {chat_stats['errors']}")
        
    else:
        # Обработка всех чатов
        stats = deduplicator.deduplicate_all(args.dry_run)
        deduplicator.print_stats()
    
    return 0


if __name__ == "__main__":
    exit(main())
