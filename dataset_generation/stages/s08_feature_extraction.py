import logging
import sqlite3

from pathlib import Path

from dataset_generation.config import DATASET_DIR
from dataset_generation.utils import generate_id

logger = logging.getLogger(__name__)

try:
    import librosa
    import numpy as np
except Exception:
    librosa = None
    np = None


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def run(config, db_conn=None):
    logger.info("Running feature extraction stage")

    ds = Path(DATASET_DIR)
    db_path = ds / "sonicmorph.db"

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    cur.execute(
        """
        SELECT song_id, file_path
        FROM songs
        WHERE status_features = 'pending'
        AND is_duplicate = 0
        """
    )

    rows = cur.fetchall()

    if not rows:
        logger.info("No songs pending feature extraction")
        conn.close()
        return True

    if librosa is None or np is None:

        logger.warning(
            "librosa/numpy unavailable; skipping feature extraction"
        )

        for song_id, _ in rows:
            cur.execute(
                """
                UPDATE songs
                SET status_features = 'skipped'
                WHERE song_id = ?
                """,
                (song_id,)
            )

        conn.commit()
        conn.close()
        return True

    for song_id, file_path in rows:

        try:

            logger.info(
                "Extracting features for %s",
                song_id,
            )

            y, sr = librosa.load(
                file_path,
                sr=None,
                mono=True,
            )

            duration = librosa.get_duration(
                y=y,
                sr=sr,
            )

            #
            # Tempo
            #

            tempo, _ = librosa.beat.beat_track(
                y=y,
                sr=sr,
            )

            tempo = np.asarray(tempo).flatten()

            tempo_value = (
                float(tempo[0])
                if len(tempo) > 0
                else 0.0
            )

            #
            # Core features
            #

            mfcc = librosa.feature.mfcc(
                y=y,
                sr=sr,
                n_mfcc=13,
            )

            chroma = librosa.feature.chroma_stft(
                y=y,
                sr=sr,
            )

            rms = librosa.feature.rms(
                y=y,
            )

            spec_cent = librosa.feature.spectral_centroid(
                y=y,
                sr=sr,
            )

            #
            # Krumhansl-Schmuckler key detection
            #

            musical_key = None
            confidence = None

            try:

                chroma_mean = np.mean(
                    chroma,
                    axis=1,
                )

                major_profile = np.array(
                    [
                        6.35, 2.23, 3.48, 2.33,
                        4.38, 4.09, 2.52, 5.19,
                        2.39, 3.66, 2.29, 2.88,
                    ]
                )

                minor_profile = np.array(
                    [
                        6.33, 2.68, 3.52, 5.38,
                        2.60, 3.53, 2.54, 4.75,
                        3.98, 2.69, 3.34, 3.17,
                    ]
                )

                def norm(v):
                    return (
                        (v - np.mean(v))
                        / (np.std(v) + 1e-9)
                    )

                chroma_n = norm(chroma_mean)
                major_n = norm(major_profile)
                minor_n = norm(minor_profile)

                best_corr = -1e9
                best_root = 0
                best_mode = "major"

                for root in range(12):

                    maj = np.roll(
                        major_n,
                        root,
                    )

                    minp = np.roll(
                        minor_n,
                        root,
                    )

                    corr_maj = float(
                        np.dot(chroma_n, maj)
                    )

                    corr_min = float(
                        np.dot(chroma_n, minp)
                    )

                    if corr_maj > best_corr:
                        best_corr = corr_maj
                        best_root = root
                        best_mode = "major"

                    if corr_min > best_corr:
                        best_corr = corr_min
                        best_root = root
                        best_mode = "minor"

                pitch_classes = [
                    "C",
                    "C#",
                    "D",
                    "D#",
                    "E",
                    "F",
                    "F#",
                    "G",
                    "G#",
                    "A",
                    "A#",
                    "B",
                ]

                musical_key = (
                    f"{pitch_classes[best_root]} "
                    f"{best_mode}"
                )

                confidence = (
                    (best_corr + 10.0)
                    / 40.0
                )

                confidence = max(
                    0.0,
                    min(1.0, confidence)
                )

            except Exception as exc:

                logger.warning(
                    "Key detection failed for %s: %s",
                    song_id,
                    exc,
                )

            #
            # Feature record
            #

            entry = {
                "feature_id": generate_id(),
                "song_id": song_id,
                "tempo": tempo_value,
                "musical_key": musical_key,
                "mfcc_mean": np.array2string(
                    np.mean(mfcc, axis=1),
                    separator=",",
                ),
                "mfcc_std": np.array2string(
                    np.std(mfcc, axis=1),
                    separator=",",
                ),
                "chroma_mean": np.array2string(
                    np.mean(chroma, axis=1),
                    separator=",",
                ),
                "rms_mean": _safe_float(
                    np.mean(rms)
                ),
                "rms_std": _safe_float(
                    np.std(rms)
                ),
                "spectral_centroid_mean": _safe_float(
                    np.mean(spec_cent)
                ),
                "spectral_centroid_std": _safe_float(
                    np.std(spec_cent)
                ),
                "duration": _safe_float(
                    duration
                ),
            }

            cur.execute(
                """
                INSERT INTO features
                (
                    feature_id,
                    song_id,
                    tempo,
                    musical_key,
                    mfcc_mean,
                    mfcc_std,
                    chroma_mean,
                    rms_mean,
                    rms_std,
                    spectral_centroid_mean,
                    spectral_centroid_std,
                    duration
                )
                VALUES
                (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    entry["feature_id"],
                    entry["song_id"],
                    entry["tempo"],
                    entry["musical_key"],
                    entry["mfcc_mean"],
                    entry["mfcc_std"],
                    entry["chroma_mean"],
                    entry["rms_mean"],
                    entry["rms_std"],
                    entry["spectral_centroid_mean"],
                    entry["spectral_centroid_std"],
                    entry["duration"],
                ),
            )

            cur.execute(
                """
                UPDATE songs
                SET status_features = 'done'
                WHERE song_id = ?
                """,
                (song_id,)
            )

            conn.commit()

        except Exception as exc:

            logger.exception(
                "Feature extraction failed for %s: %s",
                song_id,
                exc,
            )

            cur.execute(
                """
                UPDATE songs
                SET
                    status_features = 'failed',
                    error_log = ?
                WHERE song_id = ?
                """,
                (
                    str(exc),
                    song_id,
                ),
            )

            conn.commit()

    conn.close()

    return True