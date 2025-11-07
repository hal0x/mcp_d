from enum import Enum


class Tool(str, Enum):
    """Enumeration of available planning tools."""

    SEARCH = "search"
    CODE = "code"
    FILE_IO = "file_io"
    SHELL = "shell"
    HTTP = "http"
