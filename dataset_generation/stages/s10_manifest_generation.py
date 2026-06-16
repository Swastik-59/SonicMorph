import logging
import sqlite3
import json
from pathlib import Path
from dataset_generation.config import DATASET_DIR
from dataset_generation.utils import ensure_dir

logger = logging.getLogger(__name__)


def run(config, db_conn=None):
    logger.info("Running manifest generation stage")
    ds = Path(DATASET_DIR)
    manifests_dir = ensure_dir(ds / "manifests")
    db_path = ds / "sonicmorph.db"
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    # Clips are the authoritative packaged output; manifests are a derived export.
    cur.execute(
        "SELECT c.clip_id, c.song_id, s.artist_id, s.title, c.file_path, c.start_time, c.end_time, c.duration, c.target "
        "FROM clips c JOIN songs s ON s.song_id = c.song_id "
        "WHERE s.status_packaged = 'done' "
        "ORDER BY s.artist_id, c.song_id, c.target, c.start_time"
    )
    rows = cur.fetchall()
    if not rows:
        logger.info("No packaged clips available for manifest generation")
        return True

    manifest = []
    for clip_id, song_id, artist_id, title, file_path, start_time, end_time, duration, target in rows:
        entry = {
            "clip_id": clip_id,
            "song_id": song_id,
            "artist_id": artist_id,
            "title": title,
            "file_path": file_path,
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,
            "target": target,
        }
        manifest.append(entry)

    manifest_path = manifests_dir / f"manifest_{config.pipeline.get('dataset', {}).get('version','v1')}.json"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    conn.close()
    logger.info("Wrote manifest: %s", manifest_path)
    return True
