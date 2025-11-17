"""Coqui TTS MCP Server - Main server implementation."""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional

import torch
from fastapi import HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel
from starlette.requests import Request
from starlette.responses import JSONResponse
from mcp.server.fastmcp import FastMCP

from .config import Settings, get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global TTS instance
_tts_instance: Optional[object] = None
_settings: Optional[Settings] = None


def initialize_tts(settings: Settings) -> object:
    """Initialize the TTS model."""
    global _tts_instance

    if _tts_instance is not None:
        return _tts_instance

    try:
        from TTS.api import TTS

        device = settings.get_device()
        logger.info(f"Initializing TTS model '{settings.model_name}' on device '{device}'")

        _tts_instance = TTS(
            model_name=settings.model_name,
            progress_bar=False,
        ).to(device)

        logger.info(f"TTS model initialized successfully on {device}")
        return _tts_instance
    except Exception as e:
        logger.error(f"Failed to initialize TTS model: {e}")
        raise


def cleanup_old_files(output_dir: Path, max_age_seconds: int) -> None:
    """Remove old audio files from output directory."""
    try:
        current_time = time.time()
        for file_path in output_dir.glob("*.wav"):
            if current_time - file_path.stat().st_mtime > max_age_seconds:
                try:
                    file_path.unlink()
                    logger.debug(f"Removed old file: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to remove {file_path}: {e}")
    except Exception as e:
        logger.warning(f"Error during cleanup: {e}")


def create_server(settings: Optional[Settings] = None) -> FastMCP:
    """Create and configure the Coqui TTS MCP server."""
    global _settings
    _settings = settings or get_settings()

    # Ensure output directory exists
    output_dir = Path(_settings.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize TTS
    initialize_tts(_settings)

    # Create FastMCP server
    mcp = FastMCP(
        name="Coqui TTS",
        instructions=(
            "Text-to-speech synthesis server using Coqui TTS. "
            "Supports Metal (MPS) acceleration on Mac M-series. "
            "Provides tools for synthesizing speech from text."
        ),
    )

    @mcp.tool()
    def synthesize_speech(
        text: str,
        model_name: Optional[str] = None,
    ) -> dict:
        """
        Synthesize speech from text using Coqui TTS.

        Args:
            text: The text to convert to speech
            model_name: Optional model name override

        Returns:
            Dictionary with file path and metadata
        """
        try:
            # Get TTS instance
            tts = _tts_instance
            if tts is None:
                raise RuntimeError("TTS model not initialized")

            # Generate unique filename
            output_dir = Path(_settings.output_dir)
            fname = output_dir / f"tts_{uuid.uuid4().hex}.wav"

            # Synthesize
            logger.info(f"Synthesizing text: {text[:50]}...")
            tts.tts_to_file(text=text, file_path=str(fname))

            # Cleanup old files periodically
            cleanup_old_files(output_dir, _settings.cleanup_after_seconds)

            return {
                "file_path": str(fname),
                "text": text,
                "model": _settings.model_name,
                "device": _settings.get_device(),
            }
        except Exception as e:
            logger.error(f"Error synthesizing speech: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @mcp.tool()
    def list_available_models() -> dict:
        """
        List available TTS models.

        Returns:
            Dictionary with available models
        """
        try:
            from TTS.utils.manage import ModelManager
            
            # Use ModelManager to get the list of available models
            manager = ModelManager()
            # Get TTS models (not vocoders or voice conversion)
            tts_models = []
            for model_name in manager.list_models():
                if model_name.startswith("tts_models/"):
                    tts_models.append(model_name)
            
            return {
                "available_models": tts_models,
                "current_model": _settings.model_name,
            }
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            # Return error in dict format instead of raising HTTPException for MCP tool
            return {
                "error": str(e),
                "available_models": [],
                "current_model": _settings.model_name,
            }

    @mcp.tool()
    def get_device_info() -> dict:
        """
        Get information about available devices (CPU, CUDA, MPS).

        Returns:
            Dictionary with device information
        """
        device_info = {
            "cpu_available": True,
            "cuda_available": torch.cuda.is_available(),
            "mps_available": torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else False,
            "current_device": _settings.get_device(),
        }

        if device_info["cuda_available"]:
            device_info["cuda_device_count"] = torch.cuda.device_count()
            device_info["cuda_device_name"] = torch.cuda.get_device_name(0)

        return device_info

    def convert_wav_to_mp3(wav_path: Path) -> Path:
        """Convert WAV file to MP3 using ffmpeg."""
        mp3_path = wav_path.with_suffix(".mp3")
        
        # Try to find ffmpeg in common locations
        ffmpeg_paths = [
            "/opt/homebrew/bin/ffmpeg",  # Homebrew on Apple Silicon
            "/usr/local/bin/ffmpeg",     # Homebrew on Intel Mac
            "/usr/bin/ffmpeg",           # System path
            "ffmpeg",                    # In PATH
        ]
        
        ffmpeg_cmd = None
        for path in ffmpeg_paths:
            if path == "ffmpeg":
                # Try to find in PATH
                import shutil
                found = shutil.which("ffmpeg")
                if found:
                    ffmpeg_cmd = found
                    break
            else:
                if Path(path).exists():
                    ffmpeg_cmd = path
                    break
        
        if not ffmpeg_cmd:
            raise FileNotFoundError("ffmpeg not found. Please install ffmpeg for MP3 support.")
        
        try:
            subprocess.run(
                [
                    ffmpeg_cmd,
                    "-i", str(wav_path),
                    "-codec:a", "libmp3lame",
                    "-qscale:a", "2",  # High quality
                    "-y",  # Overwrite output file
                    str(mp3_path),
                ],
                check=True,
                capture_output=True,
                timeout=30,
            )
            return mp3_path
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.decode() if e.stderr else str(e)
            logger.error(f"FFmpeg conversion failed: {error_msg}")
            raise
        except FileNotFoundError as e:
            logger.error(f"ffmpeg not found: {e}")
            raise

    # Register HTTP endpoints using custom_route (only available in streamable-http mode)
    try:
        # OpenAI Compatible TTS endpoint (for AnythingLLM and other clients)
        @mcp.custom_route("/v1/audio/speech", methods=["POST", "GET"])
        async def openai_compatible_tts(request: Request):
            """OpenAI-compatible TTS endpoint for AnythingLLM integration."""
            try:
                # Handle both POST (JSON body) and GET (query params) requests
                if request.method == "POST":
                    body = await request.json()
                    text = body.get("input") or body.get("text", "")
                    model = body.get("model", "tts-1")
                    voice = body.get("voice", "default")
                else:  # GET
                    text = request.query_params.get("input") or request.query_params.get("text", "")
                    model = request.query_params.get("model", "tts-1")
                    voice = request.query_params.get("voice", "default")
                
                if not text:
                    response = JSONResponse(
                        {"error": {"message": "Missing 'input' or 'text' parameter"}},
                        status_code=400
                    )
                    response.headers["Access-Control-Allow-Origin"] = "*"
                    return response
                
                result = synthesize_speech(text=text)
                wav_path = Path(result["file_path"])

                if not wav_path.exists():
                    response = JSONResponse(
                        {"error": {"message": "Generated file not found"}},
                        status_code=500
                    )
                    response.headers["Access-Control-Allow-Origin"] = "*"
                    return response

                # Convert WAV to MP3 for OpenAI compatibility
                try:
                    mp3_path = convert_wav_to_mp3(wav_path)
                except Exception as e:
                    logger.warning(f"MP3 conversion failed, returning WAV: {e}")
                    # Fallback to WAV if conversion fails
                    from starlette.responses import FileResponse as StarletteFileResponse
                    response = StarletteFileResponse(
                        path=str(wav_path),
                        media_type="audio/wav",
                        filename="speech.wav",
                    )
                    response.headers["Access-Control-Allow-Origin"] = "*"
                    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
                    response.headers["Access-Control-Allow-Headers"] = "*"
                    return response

                from starlette.responses import FileResponse as StarletteFileResponse
                response = StarletteFileResponse(
                    path=str(mp3_path),
                    media_type="audio/mpeg",
                    filename="speech.mp3",
                )
                # Add CORS headers for browser-based clients
                response.headers["Access-Control-Allow-Origin"] = "*"
                response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
                response.headers["Access-Control-Allow-Headers"] = "*"
                return response
            except Exception as e:
                logger.error(f"Error in OpenAI-compatible TTS: {e}", exc_info=True)
                response = JSONResponse(
                    {"error": {"message": str(e)}},
                    status_code=500
                )
                response.headers["Access-Control-Allow-Origin"] = "*"
                return response
        
        # Handle OPTIONS for CORS preflight
        @mcp.custom_route("/v1/audio/speech", methods=["OPTIONS"])
        async def openai_compatible_tts_options(request: Request):
            """Handle CORS preflight requests."""
            from starlette.responses import Response
            response = Response()
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "*"
            response.headers["Access-Control-Max-Age"] = "3600"
            return response

        @mcp.custom_route("/tts", methods=["GET"])
        async def http_synthesize(request: Request) -> FileResponse:
            """HTTP endpoint for TTS synthesis (simple GET with query parameter)."""
            text = request.query_params.get("text")
            if not text:
                from starlette.responses import JSONResponse
                return JSONResponse(
                    {"error": "Missing 'text' query parameter"},
                    status_code=400
                )
            
            result = synthesize_speech(text=text)
            file_path = Path(result["file_path"])

            if not file_path.exists():
                from starlette.responses import JSONResponse
                return JSONResponse(
                    {"error": "Generated file not found"},
                    status_code=404
                )

            from starlette.responses import FileResponse as StarletteFileResponse
            return StarletteFileResponse(
                path=str(file_path),
                media_type="audio/wav",
                filename=f"tts_{uuid.uuid4().hex}.wav",
            )

        @mcp.custom_route("/health", methods=["GET"])
        async def health(request: Request) -> JSONResponse:
            """Health check endpoint."""
            return JSONResponse({
                "status": "ok",
                "model": _settings.model_name,
                "device": _settings.get_device(),
                "tts_initialized": _tts_instance is not None,
            })

        @mcp.custom_route("/models", methods=["GET"])
        async def http_list_models(request: Request) -> JSONResponse:
            """HTTP endpoint to list available models."""
            return JSONResponse(list_available_models())

        @mcp.custom_route("/device", methods=["GET"])
        async def http_device_info(request: Request) -> JSONResponse:
            """HTTP endpoint to get device information."""
            return JSONResponse(get_device_info())
    except (AttributeError, TypeError) as e:
        # custom_route not available or not working, skip HTTP endpoints
        logger.debug(f"custom_route not available: {e}, skipping HTTP endpoint registration")
        pass

    return mcp


def main() -> None:
    """Main entry point for the Coqui TTS MCP server."""
    parser = argparse.ArgumentParser(description="Coqui TTS MCP Server")
    parser.add_argument(
        "--host",
        default=None,
        help="Host to bind to (default: from config or 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to bind to (default: from config or 5002)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="TTS model name (default: from config)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to use: cpu, cuda, or mps (default: auto-detect)",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        help="Logging level (default: from config or INFO)",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        default="streamable-http",
        help="Transport mode (default: streamable-http)",
    )

    args = parser.parse_args()

    # Load settings
    settings = get_settings()

    # Apply CLI overrides
    if args.host:
        settings.host = args.host
    if args.port:
        settings.port = args.port
    if args.model:
        settings.model_name = args.model
    if args.device:
        settings.device = args.device
    if args.log_level:
        settings.log_level = args.log_level.upper()
        logging.getLogger().setLevel(getattr(logging, settings.log_level, logging.INFO))

    # Create server
    mcp = create_server(settings)

    if args.transport == "streamable-http":
        logger.info(
            f"Starting Coqui TTS MCP server in {args.transport} mode on {settings.host}:{settings.port}"
        )
        mcp.settings.host = settings.host
        mcp.settings.port = settings.port
        mcp.run(transport=args.transport)
    else:
        logger.info("Starting Coqui TTS MCP server in stdio mode")
        mcp.run()


if __name__ == "__main__":
    main()

