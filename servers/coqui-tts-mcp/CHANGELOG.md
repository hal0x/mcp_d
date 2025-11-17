# Changelog

All notable changes to the Coqui TTS MCP server will be documented in this file.

## [0.1.0] - 2024-01-XX

### Added
- Initial release of Coqui TTS MCP server
- Support for text-to-speech synthesis using Coqui TTS
- Metal (MPS) acceleration support for Mac M-series
- HTTP endpoints for direct API access
- MCP tools: `synthesize_speech`, `list_available_models`, `get_device_info`
- Automatic cleanup of old audio files
- Configurable TTS models
- Support for stdio and HTTP transports

### Features
- FastAPI-based HTTP server
- FastMCP integration for MCP protocol
- Automatic device detection (CPU, CUDA, MPS)
- Configurable output directory and cleanup intervals
- Health check and device info endpoints



