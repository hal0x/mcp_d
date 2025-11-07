import yaml

from main import load_config


def test_load_config_overrides_environment(monkeypatch, tmp_path):
    cfg = {
        "telegram": {"bot_token": "yaml_token"},
        "llm": {"api_key": "yaml_llm"},
        "embeddings": {"api_key": "yaml_emb"},
        "telethon": {
            "api_id": 1,
            "api_hash": "yaml_hash",
            "session": "yaml_session",
        },
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "env_token")
    monkeypatch.setenv("LLM_API_KEY", "env_llm")
    monkeypatch.setenv("EMBEDDINGS_API_KEY", "env_emb")
    monkeypatch.setenv("TELETHON_API_ID", "123")
    monkeypatch.setenv("TELETHON_API_HASH", "env_hash")
    monkeypatch.setenv("TELETHON_SESSION", "env_session")

    loaded = load_config(str(path))
    assert loaded["telegram"]["bot_token"] == "env_token"
    assert loaded["llm"]["api_key"] == "env_llm"
    assert loaded["embeddings"]["api_key"] == "env_emb"
    tele = loaded["telethon"]
    assert tele["api_id"] == 123
    assert tele["api_hash"] == "env_hash"
    assert tele["session"] == "env_session"


def test_load_config_env_missing_keeps_yaml(monkeypatch, tmp_path):
    cfg = {
        "telegram": {"bot_token": "yaml_token"},
        "llm": {"api_key": "yaml_llm"},
        "embeddings": {"api_key": "yaml_emb"},
        "telethon": {
            "api_id": 1,
            "api_hash": "yaml_hash",
            "session": "yaml_session",
        },
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("EMBEDDINGS_API_KEY", raising=False)
    monkeypatch.delenv("TELETHON_API_ID", raising=False)
    monkeypatch.delenv("TELETHON_API_HASH", raising=False)
    monkeypatch.delenv("TELETHON_SESSION", raising=False)

    loaded = load_config(str(path))
    assert loaded["telegram"]["bot_token"] == "yaml_token"
    assert loaded["llm"]["api_key"] == "yaml_llm"
    assert loaded["embeddings"]["api_key"] == "yaml_emb"
    tele = loaded["telethon"]
    assert tele["api_id"] == 1
    assert tele["api_hash"] == "yaml_hash"
    assert tele["session"] == "yaml_session"


def test_load_config_session_default(monkeypatch, tmp_path):
    cfg = {"telethon": {}}
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    monkeypatch.delenv("TELETHON_SESSION", raising=False)

    loaded = load_config(str(path))
    assert loaded["telethon"]["session"] == "user"
