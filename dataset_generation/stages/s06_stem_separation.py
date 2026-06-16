import logging
import sqlite3
import subprocess
import shutil

from pathlib import Path

from dataset_generation.config import DATASET_DIR
from dataset_generation.utils import ensure_dir, generate_id

logger = logging.getLogger(__name__)


def _find_vocal_stem(song_output_dir: Path):

    patterns = [
        "**/vocals.wav",
        "**/*vocals.wav",
        "**/*vocal*.wav",
    ]

    for pattern in patterns:

        matches = list(
            song_output_dir.glob(pattern)
        )

        if matches:
            return matches[0]

    return None


def _demucs_exists():

    return shutil.which(
        "demucs"
    ) is not None


def run(config, db_conn=None):

    logger.info(
        "Running stem separation stage"
    )

    if not _demucs_exists():

        logger.warning(
            "Demucs executable not found."
        )

        return True

    ds = Path(DATASET_DIR)

    stems_root = ensure_dir(
        ds / "stems"
    )

    db_path = ds / "sonicmorph.db"

    conn = sqlite3.connect(
        str(db_path)
    )

    cur = conn.cursor()

    cur.execute(
        """
        SELECT
            song_id,
            artist_id,
            file_path
        FROM songs
        WHERE status_separated='pending'
        AND is_duplicate=0
        """
    )

    rows = cur.fetchall()

    if not rows:

        logger.info(
            "No songs pending separation"
        )

        conn.close()

        return True

    model = (
        config.pipeline
        .get(
            "stem_separation",
            {},
        )
        .get(
            "model",
            "htdemucs_ft",
        )
        if hasattr(
            config,
            "pipeline",
        )
        else "htdemucs_ft"
    )

    for (
        song_id,
        artist_id,
        file_path,
    ) in rows:

        try:

            input_file = Path(
                file_path
            )

            if not input_file.exists():

                logger.warning(
                    "Missing input file: %s",
                    input_file,
                )

                continue

            #
            # Create song-specific output directory
            #

            song_output_dir = (
                stems_root
                / artist_id
                / song_id
            )

            ensure_dir(
                song_output_dir
            )

            cmd = [
                "demucs",
                "-n",
                model,
                "-o",
                str(song_output_dir),
                str(input_file),
            ]

            logger.info(
                "Running Demucs for %s",
                song_id,
            )

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7200,
            )

            if result.returncode != 0:

                error_text = (
                    result.stderr
                    or result.stdout
                )

                logger.error(
                    "Demucs failed for %s",
                    song_id,
                )

                cur.execute(
                    """
                    UPDATE songs
                    SET
                        status_separated='failed',
                        error_log=?
                    WHERE song_id=?
                    """,
                    (
                        error_text[:5000],
                        song_id,
                    ),
                )

                conn.commit()

                continue

            vocal_stem = (
                _find_vocal_stem(
                    song_output_dir
                )
            )

            if vocal_stem is None:

                logger.warning(
                    "No vocal stem found for %s",
                    song_id,
                )

                cur.execute(
                    """
                    UPDATE songs
                    SET
                        status_separated='failed',
                        error_log=?
                    WHERE song_id=?
                    """,
                    (
                        "Demucs completed but vocals.wav not found",
                        song_id,
                    ),
                )

                conn.commit()

                continue

            stem_id = generate_id()

            cur.execute(
                """
                INSERT OR REPLACE INTO stems
                (
                    stem_id,
                    song_id,
                    stem_type,
                    file_path
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    stem_id,
                    song_id,
                    "vocals",
                    str(vocal_stem),
                ),
            )

            cur.execute(
                """
                UPDATE songs
                SET status_separated='done'
                WHERE song_id=?
                """,
                (
                    song_id,
                ),
            )

            conn.commit()

            logger.info(
                "Created vocal stem for %s",
                song_id,
            )

        except subprocess.TimeoutExpired:

            logger.error(
                "Demucs timeout for %s",
                song_id,
            )

            cur.execute(
                """
                UPDATE songs
                SET
                    status_separated='failed',
                    error_log=?
                WHERE song_id=?
                """,
                (
                    "Demucs timeout",
                    song_id,
                ),
            )

            conn.commit()

        except Exception as exc:

            logger.exception(
                "Stem separation failed for %s",
                song_id,
            )

            cur.execute(
                """
                UPDATE songs
                SET
                    status_separated='failed',
                    error_log=?
                WHERE song_id=?
                """,
                (
                    str(exc),
                    song_id,
                ),
            )

            conn.commit()

    conn.close()

    return True