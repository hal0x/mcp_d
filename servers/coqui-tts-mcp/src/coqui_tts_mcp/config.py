"""Configuration for Coqui TTS MCP server."""

from __future__ import annotations

import os
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Server settings."""

    model_config = SettingsConfigDict(
        env_prefix="COQUI_TTS_MCP_",
        case_sensitive=False,
    )

    # Server settings
    host: str = "0.0.0.0"
    port: int = 5002
    log_level: str = "INFO"

    # TTS settings
    model_name: str = "tts_models/en/ljspeech/tacotron2-DDC"
    device: Optional[str] = None  # Auto-detect if None
    output_dir: str = "/tmp/coqui_tts_output"
    cleanup_after_seconds: int = 300  # Clean up old files after 5 minutes

    # Metal/MPS settings
    use_mps: bool = True  # Use Metal Performance Shaders on Mac

    def get_device(self) -> str:
        """Get the device to use for TTS."""
        if self.device:
            return self.device

        import torch

        if self.use_mps and torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"


def get_settings() -> Settings:
    """Get server settings."""
    return Settings()




