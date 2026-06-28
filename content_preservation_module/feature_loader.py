import sqlite3
from pathlib import Path

from content_preservation_module.song_features import SongFeatures


class FeatureLoader:
    """
    Loads song features from the SonicMorph SQLite database.
    """

    VALID_MODES = {"major", "minor", "dorian", "mixolydian",
                   "lydian", "phrygian", "locrian"}

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"Database not found: {self.db_path}\n"
                "Check your DB_PATH setting."
            )

    # ── public ────────────────────────────────────────────
    def load(self, song_id: str) -> SongFeatures:
        """
        Load features for a single song by its ID.
        Raises ValueError if the song_id is not in the database.
        """
        row = self._fetch_row(song_id)
        return self._row_to_features(row, song_id)

    def exists(self, song_id: str) -> bool:
        """Return True if song_id has a features row."""
        try:
            self._fetch_row(song_id)
            return True
        except ValueError:
            return False

    # ── private ───────────────────────────────────────────
    def _fetch_row(self, song_id: str) -> tuple:
        conn = sqlite3.connect(str(self.db_path))
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT
                    duration,
                    tempo,
                    musical_key,
                    rms_mean,
                    spectral_centroid_mean
                FROM features
                WHERE song_id = ?
                """,
                (song_id,),
            )
            row = cur.fetchone()
        finally:
            conn.close()

        if row is None:
            raise ValueError(
                f"No features found for song_id='{song_id}'.\n"
                "Verify the ID exists in the features table."
            )
        return row

    def _row_to_features(self, row: tuple, song_id: str) -> SongFeatures:
        duration, tempo, musical_key, rms_mean, spectral_centroid = row

        # ── parse key / mode ──────────────────────────────
        key, mode = self._parse_musical_key(musical_key)

        # ── guard against NULL / zero tempo ──────────────
        tempo = float(tempo or 0)
        if tempo <= 0:
            raise ValueError(
                f"song_id='{song_id}' has invalid tempo={tempo}. "
                "Fix the database entry before packaging."
            )

        return SongFeatures(
            duration               = float(duration or 0),
            tempo                  = tempo,
            key                    = key,
            mode                   = mode,
            rms_mean               = float(rms_mean or 0),
            spectral_centroid_mean = float(spectral_centroid or 0),
        )

    def _parse_musical_key(self, musical_key: str | None) -> tuple[str, str]:
        """
        Parse 'F major', 'A minor', 'C', etc.
        Returns (key, mode) with safe fallbacks.
        """
        if not musical_key:
            return "C", "major"

        parts = musical_key.strip().split()
        key   = parts[0] if parts else "C"
        mode  = parts[1].lower() if len(parts) > 1 else "major"

        if mode not in self.VALID_MODES:
            # unknown mode token — default to major
            mode = "major"

        return key, mode