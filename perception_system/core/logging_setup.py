"""Process-wide logging: console + size-rotated file, UTC ISO-8601 timestamps."""
from __future__ import annotations

import logging
import logging.handlers
import sys
import time
from pathlib import Path
from typing import Optional, Union

_FMT = "%(asctime)s.%(msecs)03dZ %(levelname)-8s %(threadName)-13s %(name)-24s | %(message)s"
_DATEFMT = "%Y-%m-%dT%H:%M:%S"
_NOISY = ("ultralytics", "urllib3", "PIL", "matplotlib", "insightface", "onnxruntime", "streamlit")


def setup_logging(level: str = "INFO", log_dir: Optional[Union[str, Path]] = None,
                  filename: str = "system.log", max_bytes: int = 10 * 1024 * 1024,
                  backups: int = 10) -> None:
    """Configure the root logger once (idempotent: existing handlers are replaced)."""
    formatter = logging.Formatter(_FMT, _DATEFMT)
    formatter.converter = time.gmtime
    root = logging.getLogger()
    root.setLevel(level.upper())
    for handler in list(root.handlers):
        root.removeHandler(handler)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)
    if log_dir is not None:
        directory = Path(log_dir)
        directory.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            directory / filename, maxBytes=max_bytes, backupCount=backups, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    logging.captureWarnings(True)
    for name in _NOISY:
        logging.getLogger(name).setLevel(logging.WARNING)
