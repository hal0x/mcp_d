# metrics/__init__.py
import os
USE_NOOP = os.getenv("METRICS_NOOP", "").lower() in ("1", "true", "yes")

if USE_NOOP:
    from .noop import *  # noqa
else:
    from .registry import *  # noqa
