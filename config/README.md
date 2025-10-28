# Config Directory

- `mcp-services.yaml` — каталог MCP сервисов (без секретов), валидируется схемой `schema/mcp-services.schema.json`.
- JSON Schema находится в `config/schema/`, использовать `jsonschema` для проверки.
- Чувствительные данные (API токены, пароли) хранить только через переменные окружения, а не в YAML.

## Быстрая валидация

```bash
python - <<'PY'
import json, yaml, jsonschema
from pathlib import Path
base = Path('config')
schema = json.load((base / 'schema/mcp-services.schema.json').open())
data = yaml.safe_load((base / 'mcp-services.yaml').open())
jsonschema.validate(instance=data, schema=schema)
print('mcp-services.yaml is valid')
PY
```
