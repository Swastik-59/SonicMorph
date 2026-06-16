import json
import logging
import sqlite3
import subprocess

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dataset_generation.config import DATASET_DIR

logger = logging.getLogger(__name__)


def _extract_loudnorm_json(text: str):
    """
    Robust extraction of loudnorm JSON block from ffmpeg output.
    """

    start = text.find("{")

    if start == -1:
        return None

    brace_count = 0

    for i in range(start, len(text)):

        if text[i] == "{":
            brace_count += 1

        elif text[i] == "}":
            brace_count -= 1

            if brace_count == 0:
                try:
                    return json.loads(text[start : i + 1])
                except Exception:
                    return None

    return None


def _verify_output(path: str) -> bool:

    try:

        p = Path(path)

        return (
            p.exists()
            and p.is_file()
            and p.stat().st_size > 1024
        )

    except Exception:
        return False


def _ffmpeg_normalize(
    in_path: str,
    out_path: str,
    target_lufs: float,
) -> bool:

    try:

        #
        # PASS 1 - ANALYZE
        #

        analyze_cmd = [
            "ffmpeg",
            "-hide_banner",
            "-nostats",
            "-y",
            "-i",
            in_path,
            "-af",
            f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11:print_format=json",
            "-f",
            "null",
            "-",
        ]

        analyze = subprocess.run(
            analyze_cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=180,
        )

        ffmpeg_output = (
            analyze.stdout
            + "\n"
            + analyze.stderr
        )

        data = _extract_loudnorm_json(
            ffmpeg_output
        )

        #
        # FALLBACK IF JSON IS NOT FOUND
        #

        if data is None:

            logger.warning(
                "Could not extract loudnorm JSON for %s. "
                "Falling back to one-pass normalization.",
                in_path,
            )

            normalize_cmd = [
                "ffmpeg",
                "-hide_banner",
                "-nostats",
                "-y",
                "-i",
                in_path,
                "-af",
                f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11",
                "-ar",
                "44100",
                "-ac",
                "1",
                out_path,
            ]

            result = subprocess.run(
                normalize_cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore",
                timeout=600,
            )

            if result.returncode != 0:

                logger.warning(
                    "Fallback normalization failed:\n%s",
                    result.stderr,
                )

                return False

            return _verify_output(out_path)

        #
        # PASS 2 - TRUE TWO PASS NORMALIZATION
        #

        normalize_filter = (
            "loudnorm="
            f"I={target_lufs}:"
            "TP=-1.5:"
            "LRA=11:"
            f"measured_I={data['input_i']}:"
            f"measured_LRA={data['input_lra']}:"
            f"measured_TP={data['input_tp']}:"
            f"measured_thresh={data['input_thresh']}:"
            f"offset={data['target_offset']}:"
            "linear=true:"
            "print_format=summary"
        )

        normalize_cmd = [
            "ffmpeg",
            "-hide_banner",
            "-nostats",
            "-y",
            "-i",
            in_path,
            "-af",
            normalize_filter,
            "-ar",
            "44100",
            "-ac",
            "1",
            out_path,
        ]

        result = subprocess.run(
            normalize_cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=600,
        )

        if result.returncode != 0:

            logger.warning(
                "Normalization failed:\n%s",
                result.stderr,
            )

            return False

        return _verify_output(out_path)

    except subprocess.TimeoutExpired:

        logger.warning(
            "FFmpeg timeout while processing %s",
            in_path,
        )

        return False

    except Exception as exc:

        logger.exception(
            "Normalization exception: %s",
            exc,
        )

        return False


def run(config, db_conn=None):

    logger.info(
        "Running audio normalization stage"
    )

    ds = Path(DATASET_DIR)

    db_path = ds / "sonicmorph.db"

    conn = sqlite3.connect(
        str(db_path)
    )

    cur = conn.cursor()

    target_lufs = (
        config.pipeline
        .get("normalization", {})
        .get("target_lufs", -14)
        if hasattr(config, "pipeline")
        else -14
    )

    cur.execute(
        """
        SELECT song_id, file_path
        FROM songs
        WHERE status_normalized='pending'
        AND is_duplicate=0
        """
    )

    rows = cur.fetchall()

    if not rows:

        logger.info(
            "No songs pending normalization"
        )

        conn.close()

        return True

    processing_workers = (
        config.pipeline
        .get("concurrency", {})
        .get("processing_workers", 2)
        if hasattr(config, "pipeline")
        else 2
    )

    def normalize_one(item):

        song_id, file_path = item

        from dataset_generation.jobs import (
            create_job,
            start_job,
            complete_job,
        )

        job_id = create_job(
            "s05_audio_normalization",
            song=song_id,
        )

        start_job(job_id)

        src = Path(file_path)

        out_path = (
            str(src.with_suffix(""))
            + "_normalized.wav"
        )

        ok = _ffmpeg_normalize(
            str(src),
            out_path,
            target_lufs,
        )

        complete_job(
            job_id,
            success=ok,
        )

        return (
            song_id,
            out_path,
            ok,
        )

    with ThreadPoolExecutor(
        max_workers=processing_workers
    ) as executor:

        futures = {
            executor.submit(
                normalize_one,
                row,
            ): row
            for row in rows
        }

        for future in as_completed(
            futures
        ):

            song_id, out_path, ok = (
                future.result()
            )

            if ok:

                cur.execute(
                    """
                    UPDATE songs
                    SET
                        file_path=?,
                        status_normalized='done'
                    WHERE song_id=?
                    """,
                    (
                        out_path,
                        song_id,
                    ),
                )

            else:

                cur.execute(
                    """
                    UPDATE songs
                    SET status_normalized='failed'
                    WHERE song_id=?
                    """,
                    (song_id,),
                )

                logger.warning(
                    "Normalization failed for %s",
                    song_id,
                )

            conn.commit()

    conn.close()

    return True