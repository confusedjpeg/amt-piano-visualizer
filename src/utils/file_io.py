"""
File I/O helpers — path utilities, temp directory management,
and run-ID generation.
"""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path


def generate_run_id() -> str:
    """Generate a unique run identifier based on the current UTC timestamp.

    Returns:
        A string like ``run_20260525_150204``.
    """
    now = datetime.now(timezone.utc)
    return f"run_{now.strftime('%Y%m%d_%H%M%S')}"


def create_run_directory(base_dir: Path, run_id: str) -> Path:
    """Create a per-run subdirectory under *base_dir*.

    Args:
        base_dir: The intermediate output directory (e.g. ``data/intermediate``).
        run_id: Unique run identifier.

    Returns:
        Path to the newly created run directory.
    """
    run_dir = Path(base_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def ensure_directory(path: Path) -> Path:
    """Create a directory (and parents) if it doesn't already exist.

    Args:
        path: Directory path to ensure exists.

    Returns:
        The same path, guaranteed to exist.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_copy(src: Path, dst: Path) -> Path:
    """Copy a file, creating the destination directory if needed.

    Args:
        src: Source file path.
        dst: Destination file path.

    Returns:
        The destination path.
    """
    ensure_directory(dst.parent)
    shutil.copy2(str(src), str(dst))
    return dst
