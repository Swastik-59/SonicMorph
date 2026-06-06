import logging
import sqlite3

from pathlib import Path
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)

from sonicmorph.collectors.youtube_collector import (
    YouTubeCollector,
)

from sonicmorph.utils import (
    slugify,
    generate_id,
    compute_file_hash,
    get_audio_info,
    ensure_dir,
)

from sonicmorph.config import DATASET_DIR

logger = logging.getLogger(__name__)


def _insert_song(
    conn: sqlite3.Connection,
    song: dict,
):
    cur = conn.cursor()

    cur.execute(
        """
        INSERT OR REPLACE INTO songs
        (
            song_id,
            artist_id,
            title,
            duration,
            sample_rate,
            channels,
            file_path,
            file_hash,
            source,
            dataset_version
        )
        VALUES
        (
            :song_id,
            :artist_id,
            :title,
            :duration,
            :sample_rate,
            :channels,
            :file_path,
            :file_hash,
            :source,
            :dataset_version
        )
        """,
        song,
    )

    conn.commit()


def run(config, db_conn=None):

    logger.info(
        "Running audio collection stage"
    )

    ds = Path(DATASET_DIR)

    raw_dir = ds / "raw"

    raw_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    artists = (
        config.artists.get("artists")
        if getattr(config, "artists", None)
        else []
    )

    if not artists:

        logger.info(
            "No artists configured for collection"
        )

        return True

    db_path = ds / "sonicmorph.db"

    conn = sqlite3.connect(
        str(db_path)
    )

    download_workers = (
        config.pipeline
        .get("concurrency", {})
        .get("download_workers", 3)
        if hasattr(config, "pipeline")
        else 3
    )

    def process_artist(artist_cfg):

        name = artist_cfg.get("name")

        if not name:
            return []

        if not artist_cfg.get(
            "enabled",
            True,
        ):
            logger.debug(
                "Artist %s disabled",
                name,
            )
            return []

        target_song_count = int(
            artist_cfg.get(
                "target_song_count",
                25,
            )
        )

        artist_slug = slugify(
            name
        )

        artist_dir = ensure_dir(
            raw_dir / artist_slug
        )

        collector = YouTubeCollector(
            config.sources
            if hasattr(config, "sources")
            else {}
        )

        logger.info(
            "Collecting up to %s songs for %s",
            target_song_count,
            name,
        )

        from sonicmorph.jobs import (
            create_job,
            start_job,
            complete_job,
        )

        results = []

        downloaded_hashes = set()

        for candidate in collector.discover(
            name
        ):

            if (
                len(results)
                >= target_song_count
            ):

                logger.info(
                    "Target reached for %s (%s songs)",
                    name,
                    target_song_count,
                )

                break

            job_id = create_job(
                "s02_audio_collection",
                artist=artist_slug,
            )

            start_job(job_id)

            try:

                out_path = collector.download(
                    candidate,
                    artist_dir,
                )

            except Exception as exc:

                logger.warning(
                    "Download failed for %s: %s",
                    name,
                    exc,
                )

                complete_job(
                    job_id,
                    success=False,
                )

                continue

            try:

                file_hash = (
                    compute_file_hash(
                        out_path
                    )
                )

                if (
                    file_hash
                    in downloaded_hashes
                ):

                    logger.info(
                        "Duplicate download skipped: %s",
                        out_path.name,
                    )

                    try:
                        out_path.unlink(
                            missing_ok=True
                        )
                    except Exception:
                        pass

                    complete_job(
                        job_id,
                        success=False,
                    )

                    continue

                downloaded_hashes.add(
                    file_hash
                )

                info = get_audio_info(
                    out_path
                )

                song = {
                    "song_id": generate_id(),
                    "artist_id": artist_slug,
                    "title": out_path.stem,
                    "duration": (
                        info.get("duration")
                        if info
                        else None
                    ),
                    "sample_rate": (
                        info.get("sample_rate")
                        if info
                        else None
                    ),
                    "channels": (
                        info.get("channels")
                        if info
                        else None
                    ),
                    "file_path": str(
                        out_path
                    ),
                    "file_hash": file_hash,
                    "source": "youtube",
                    "dataset_version": (
                        config.pipeline
                        .get("dataset", {})
                        .get(
                            "version",
                            "v1",
                        )
                        if hasattr(
                            config,
                            "pipeline",
                        )
                        else "v1"
                    ),
                }

                results.append(
                    song
                )

                logger.info(
                    "[%s/%s] Downloaded %s",
                    len(results),
                    target_song_count,
                    out_path.name,
                )

                complete_job(
                    job_id,
                    success=True,
                )

            except Exception as exc:

                logger.exception(
                    "Failed processing %s: %s",
                    out_path,
                    exc,
                )

                complete_job(
                    job_id,
                    success=False,
                )

        logger.info(
            "Finished artist %s with %s songs",
            name,
            len(results),
        )

        return results

    with ThreadPoolExecutor(
        max_workers=download_workers
    ) as executor:

        futures = {
            executor.submit(
                process_artist,
                artist_cfg,
            ): artist_cfg
            for artist_cfg in artists
        }

        for future in as_completed(
            futures
        ):

            try:

                songs = future.result()

                for song in songs:

                    _insert_song(
                        conn,
                        song,
                    )

                    logger.info(
                        "Imported %s -> %s",
                        song["file_path"],
                        song["song_id"],
                    )

            except Exception as exc:

                logger.exception(
                    "Artist processing failed: %s",
                    exc,
                )

    conn.close()

    return True