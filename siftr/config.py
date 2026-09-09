"""Where siftr keeps its index."""

from __future__ import annotations

import os
from pathlib import Path

APP_NAME = "siftr"


def default_db_path() -> Path:
    """The default index location.

    ``SIFTR_HOME`` overrides it, which is what the test suite uses to stay clear
    of a real library.
    """
    override = os.environ.get("SIFTR_HOME")
    if override:
        return Path(override).expanduser() / "index.db"
    return Path.home() / ".siftr" / "index.db"
