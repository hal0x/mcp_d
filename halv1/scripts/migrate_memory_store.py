#!/usr/bin/env python3
"""
Скрипт для миграции с MemoryStore на UnifiedMemory.

Использование:
    python scripts/migrate_memory_store.py --dry-run  # Показать изменения без применения
    python scripts/migrate_memory_store.py --apply    # Применить изменения
    python scripts/migrate_memory_store.py --check    # Проверить текущее состояние
"""

import argparse
import os
import re
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict


class MemoryStoreMigrator:
    """Мигратор для замены MemoryStore на UnifiedMemory."""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.changes: List[Tuple[str, str, str]] = []  # (file, old, new)
        
    def find_files(self) -> List[Path]:
        """Находит все Python файлы, которые используют MemoryStore."""
        files = []
        for py_file in self.project_root.rglob("*.py"):
            if self._uses_memory_store(py_file):
                files.append(py_file)
        return files
    
    def _uses_memory_store(self, file_path: Path) -> bool:
        """Проверяет, использует ли файл MemoryStore."""
        try:
            content = file_path.read_text(encoding="utf-8")
            return "MemoryStore" in content
        except Exception:
            return False
    
    def analyze_file(self, file_path: Path) -> List[Tuple[str, str, str]]:
        """Анализирует файл и возвращает список изменений."""
        changes = []
        content = file_path.read_text(encoding="utf-8")
        
        # 1. Замена импортов
        if "from memory import MemoryStore" in content:
            new_content = content.replace(
                "from memory import MemoryStore",
                "from memory import UnifiedMemory"
            )
            changes.append((str(file_path), content, new_content))
            content = new_content
        
        if "from memory.memory_store import MemoryStore" in content:
            new_content = content.replace(
                "from memory.memory_store import MemoryStore",
                "from memory import UnifiedMemory"
            )
            changes.append((str(file_path), content, new_content))
            content = new_content
        
        # 2. Замена создания экземпляров
        if "MemoryStore(" in content:
            new_content = content.replace("MemoryStore(", "UnifiedMemory(")
            changes.append((str(file_path), content, new_content))
            content = new_content
        
        # 3. Замена в комментариях и строках (опционально)
        if "MemoryStore" in content and "UnifiedMemory" not in content:
            new_content = content.replace("MemoryStore", "UnifiedMemory")
            changes.append((str(file_path), content, new_content))
        
        return changes
    
    def check_parameters(self, file_path: Path) -> List[str]:
        """Проверяет, нужно ли обновлять параметры конструктора."""
        issues = []
        content = file_path.read_text(encoding="utf-8")
        
        # Проверяем использование старых параметров MemoryStore
        if "long_term_path=" in content:
            issues.append(f"Использует long_term_path= (заменить на path=)")
        
        if "episode_graph_path=" in content:
            issues.append(f"Использует episode_graph_path= (заменить на path=)")
        
        if "embeddings_client=" in content:
            issues.append(f"Проверить параметр embeddings_client=")
        
        return issues
    
    def migrate_file(self, file_path: Path, dry_run: bool = True) -> bool:
        """Мигрирует один файл."""
        changes = self.analyze_file(file_path)
        issues = self.check_parameters(file_path)
        
        if not changes and not issues:
            return True
        
        print(f"\n📁 {file_path.relative_to(self.project_root)}")
        
        if issues:
            print("⚠️  Требует ручной проверки:")
            for issue in issues:
                print(f"   - {issue}")
        
        if changes:
            print("🔄 Изменения:")
            for file_path_str, old_content, new_content in changes:
                # Показываем diff
                old_lines = old_content.splitlines()
                new_lines = new_content.splitlines()
                
                for i, (old_line, new_line) in enumerate(zip(old_lines, new_lines)):
                    if old_line != new_line:
                        print(f"   {i+1:3d}: - {old_line}")
                        print(f"   {i+1:3d}: + {new_line}")
            
            if not dry_run:
                # Применяем изменения
                final_content = changes[-1][2]  # Берем последнее изменение
                file_path.write_text(final_content, encoding="utf-8")
                print("✅ Изменения применены")
            else:
                print("🔍 Dry run - изменения не применены")
        
        return True
    
    def run_migration(self, dry_run: bool = True) -> Dict[str, int]:
        """Запускает миграцию."""
        files = self.find_files()
        stats = {
            "total_files": len(files),
            "migrated": 0,
            "issues": 0,
            "skipped": 0
        }
        
        print(f"🔍 Найдено {len(files)} файлов с MemoryStore")
        
        for file_path in files:
            try:
                if self.migrate_file(file_path, dry_run):
                    stats["migrated"] += 1
                else:
                    stats["skipped"] += 1
            except Exception as e:
                print(f"❌ Ошибка в {file_path}: {e}")
                stats["issues"] += 1
        
        return stats
    
    def check_status(self) -> None:
        """Проверяет текущее состояние миграции."""
        files = self.find_files()
        
        print("📊 Статус миграции MemoryStore -> UnifiedMemory")
        print("=" * 50)
        
        for file_path in files:
            relative_path = file_path.relative_to(self.project_root)
            issues = self.check_parameters(file_path)
            
            status = "✅" if not issues else "⚠️"
            print(f"{status} {relative_path}")
            
            if issues:
                for issue in issues:
                    print(f"   - {issue}")
        
        print(f"\n📈 Итого: {len(files)} файлов требуют внимания")


def main():
    parser = argparse.ArgumentParser(description="Миграция MemoryStore -> UnifiedMemory")
    parser.add_argument("--dry-run", action="store_true", help="Показать изменения без применения")
    parser.add_argument("--apply", action="store_true", help="Применить изменения")
    parser.add_argument("--check", action="store_true", help="Проверить текущее состояние")
    parser.add_argument("--project-root", default=".", help="Корневая директория проекта")
    parser.add_argument("files", nargs="*", help="Конкретные файлы или директории для миграции")
    
    args = parser.parse_args()
    
    migrator = MemoryStoreMigrator(args.project_root)
    
    if args.check:
        migrator.check_status()
    elif args.dry_run or args.apply:
        if args.files:
            # Мигрируем конкретные файлы
            stats = {"total_files": 0, "migrated": 0, "issues": 0, "skipped": 0}
            for file_pattern in args.files:
                file_path = Path(file_pattern)
                if file_path.exists():
                    if file_path.is_file() and file_path.suffix == ".py":
                        stats["total_files"] += 1
                        if migrator.migrate_file(file_path, dry_run=args.dry_run):
                            stats["migrated"] += 1
                        else:
                            stats["skipped"] += 1
                    elif file_path.is_dir():
                        # Рекурсивно ищем Python файлы
                        for py_file in file_path.rglob("*.py"):
                            if migrator._uses_memory_store(py_file):
                                stats["total_files"] += 1
                                if migrator.migrate_file(py_file, dry_run=args.dry_run):
                                    stats["migrated"] += 1
                                else:
                                    stats["skipped"] += 1
            print(f"\n📊 Результат: {stats}")
        else:
            # Мигрируем все файлы
            if args.dry_run:
                print("🔍 DRY RUN - изменения не будут применены")
            else:
                print("⚠️  ПРИМЕНЕНИЕ ИЗМЕНЕНИЙ")
                confirm = input("Вы уверены? (yes/no): ")
                if confirm.lower() != "yes":
                    print("❌ Отменено")
                    return
            
            stats = migrator.run_migration(dry_run=args.dry_run)
            print(f"\n📊 Результат: {stats}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
