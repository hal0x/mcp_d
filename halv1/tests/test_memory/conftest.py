import sys
from pathlib import Path

# Добавляем корневую папку проекта в Python path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
