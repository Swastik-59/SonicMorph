"""Shared utility functions for the SonicMorph pipeline."""

from __future__ import annotations

import hashlib
import re
import uuid
from pathlib import Path
from typing import Any

try:
    import soundfile as sf
except Exception:  # pragma: no cover - optional dependency
    sf = None


def slugify(text: str) -> str:
    """Convert a human-readable string into a lowercase slug.

    Rules:
        * Lowercase the entire string.
        * Replace spaces, hyphens, and any non-alphanumeric character with
          underscores.
        * Collapse consecutive underscores into one.
        * Strip leading / trailing underscores.

    Examples:
        >>> slugify("The Beatles")
        'the_beatles'
        >>> slugify("  Pink Floyd  ")
        'pink_floyd'
    """
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def generate_id() -> str:
    """Return a new UUID4 hex string (32 characters, no hyphens)."""
    return uuid.uuid4().hex


def safe_filename(text: str) -> str:
    """Sanitise *text* so it is safe to use as a filename on all major OSes.

    Replaces or removes characters that are forbidden in Windows / macOS /
    Linux filenames.  Collapses whitespace and trims to 200 characters.
    """
    # Remove characters forbidden on Windows: \ / : * ? " < > |
    text = re.sub(r'[\\/:*?"<>|]', "", text)
    # Replace any remaining non-printable / control characters
    text = re.sub(r"[\x00-\x1f\x7f]", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Trim length (keep a reasonable limit)
    return text[:200] if text else "unnamed"


def get_audio_info(file_path: Path) -> dict[str, Any] | None:
    """Return basic audio metadata using *soundfile*.

    Returns a dict with keys ``duration``, ``sample_rate``, ``channels``,
    and ``frames``, or ``None`` if the file cannot be read.
    """
    if sf is None:
        return None
    try:
        info = sf.info(str(file_path))
        return {
            "duration": info.duration,
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "frames": info.frames,
        }
    except Exception:  # noqa: BLE001
        return None


def compute_file_hash(file_path: Path, algorithm: str = "sha256") -> str:
    """Compute the hex digest of *file_path* using the given hash algorithm.

    Reads the file in 64 KiB chunks to keep memory usage low for large
    audio files.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the algorithm is unsupported.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    try:
        hasher = hashlib.new(algorithm)
    except ValueError as exc:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}") from exc

    buf_size = 65536  # 64 KiB
    with file_path.open("rb") as fh:
        while True:
            data = fh.read(buf_size)
            if not data:
                break
            hasher.update(data)
    return hasher.hexdigest()


def ensure_dir(path: Path) -> Path:
    """Create *path* (and parents) if it does not exist, then return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def format_duration(seconds: float) -> str:
    """Format a duration in seconds as a human-readable string.

    Examples:
        >>> format_duration(225.4)
        '3m 45s'
        >>> format_duration(59.9)
        '0m 59s'
        >>> format_duration(3661)
        '61m 01s'
    """
    total = int(seconds)
    mins, secs = divmod(total, 60)
    return f"{mins}m {secs:02d}s"


def format_size(num_bytes: int) -> str:
    """Format a byte count as a human-readable string.

    Uses binary prefixes (KiB, MiB, GiB, TiB) for precision.

    Examples:
        >>> format_size(1536)
        '1.50 KiB'
        >>> format_size(1073741824)
        '1.00 GiB'
    """
    if num_bytes < 0:
        return f"-{format_size(-num_bytes)}"

    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(num_bytes) < 1024.0 or unit == "TiB":
            if unit == "B":
                return f"{num_bytes} B"
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0  # type: ignore[assignment]
    # Fallback (unreachable in practice)
    return f"{num_bytes:.2f} TiB"


def sanitize_artist_dir(artist_name: str) -> str:
    """Return a filesystem-safe directory name for an artist.

    This is intentionally different from :func:`slugify`; it preserves
    capitalisation and spaces where possible but strips characters that
    are unsafe for directory names on Windows.
    """
    # Remove characters forbidden on Windows
    name = re.sub(r'[\\/:*?"<>|]', "", artist_name)
    # Replace control characters
    name = re.sub(r"[\x00-\x1f\x7f]", "", name)
    # Collapse whitespace
    name = re.sub(r"\s+", " ", name).strip()
    # Ensure non-empty
    return name if name else "unknown_artist"
