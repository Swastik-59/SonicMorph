import logging
import sqlite3
from pathlib import Path
from dataset_generation.config import DATASET_DIR
from dataset_generation.utils import get_audio_info

logger = logging.getLogger(__name__)


def run(config, db_conn=None):
    logger.info("Running audio validation stage")
    ds = Path(DATASET_DIR)
    db_path = ds / "sonicmorph.db"
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    cur.execute("SELECT song_id, file_path FROM songs WHERE status_validated = 'pending'")
    rows = cur.fetchall()
    if not rows:
        logger.info("No songs pending validation")
        return True

    vcfg = config.pipeline.get("validation", {}) if hasattr(config, 'pipeline') else {}
    min_d = vcfg.get("min_duration", 30)
    max_d = vcfg.get("max_duration", 600)

    for song_id, file_path in rows:
        info = get_audio_info(Path(file_path))
        if not info:
            cur.execute("UPDATE songs SET status_validated = 'rejected', validation_errors = ? WHERE song_id = ?", ("unreadable", song_id))
            conn.commit()
            continue

        duration = info.get("duration")
        if duration is None or duration < min_d or duration > max_d:
            cur.execute("UPDATE songs SET status_validated = 'rejected', validation_errors = ? WHERE song_id = ?", (f"duration_out_of_range:{duration}", song_id))
        else:
            cur.execute("UPDATE songs SET status_validated = 'done' WHERE song_id = ?", (song_id,))
        conn.commit()

    conn.close()
    return True
