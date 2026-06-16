import logging
import sqlite3
from pathlib import Path
from dataset_generation.config import DATASET_DIR
from dataset_generation.utils import slugify

logger = logging.getLogger(__name__)


def run(config, db_conn=None):
    logger.info("Running artist discovery stage")
    ds = Path(DATASET_DIR)
    db_path = ds / "sonicmorph.db"
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    cfg_artists = config.artists.get("artists") if hasattr(config, 'artists') else []
    for a in cfg_artists:
        name = a.get("name")
        if not name:
            continue
        artist_id = slugify(name)
        enabled = 1 if a.get("enabled", True) else 0
        target = a.get("target_song_count", 100)
        cur.execute("INSERT OR REPLACE INTO artists (artist_id, name, enabled, target_song_count) VALUES (?, ?, ?, ?)",
                    (artist_id, name, enabled, target))
    conn.commit()
    conn.close()
    return True
