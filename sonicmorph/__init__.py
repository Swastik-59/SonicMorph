"""SonicMorph — Automated dataset pipeline for music AI training."""

from __future__ import annotations

__version__ = "1.0.0"

# Export common symbols for static analysis and imports
from .config import config
from .pipeline import Pipeline
from .database import init_db
