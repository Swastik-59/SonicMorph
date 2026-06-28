import sqlite3
from pathlib import Path

from content_preservation_module.song_features import (
    SongFeatures,
)


class FeatureLoader:
    """
    Loads song features from the SonicMorph SQLite database.
    """

    def __init__(self, db_path: str | Path):

        self.db_path = str(db_path)

    def load(self, song_id: str) -> SongFeatures:

        conn = sqlite3.connect(self.db_path)
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
            (song_id,)
        )

        row = cur.fetchone()

        conn.close()

        if row is None:
            raise ValueError(
                f"No features found for song_id={song_id}"
            )

        (
            duration,
            tempo,
            musical_key,
            rms_mean,
            spectral_centroid_mean,
        ) = row

        if musical_key:

            parts = musical_key.split()

            key = parts[0]

            if len(parts) > 1:
                mode = parts[1]
            else:
                mode = "major"

        else:

            key = "C"
            mode = "major"

        return SongFeatures(
            duration=float(duration),
            tempo=float(tempo),
            key=key,
            mode=mode,
            rms_mean=float(rms_mean or 0),
            spectral_centroid_mean=float(
                spectral_centroid_mean or 0
            ),
        )