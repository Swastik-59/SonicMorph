import logging
import sqlite3
from pathlib import Path
from sonicmorph.config import DATASET_DIR
from sonicmorph.utils import slugify

try:
    from mutagen import File as MutagenFile
except Exception:  # pragma: no cover
    MutagenFile = None

logger = logging.getLogger(__name__)


def run(config, db_conn=None):
    logger.info("Running metadata extraction stage")
    ds = Path(DATASET_DIR)
    db_path = ds / "sonicmorph.db"
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    cur.execute("SELECT song_id, artist_id, file_path FROM songs WHERE status_metadata = 'pending'")
    rows = cur.fetchall()
    if not rows:
        logger.info("No songs pending metadata extraction")
        return True

    for song_id, artist_id, file_path in rows:
        title = None
        album = None
        year = None
        try:
            if MutagenFile:
                m = MutagenFile(file_path)
                if m:
                    title = m.get("TIT2") or m.tags.get("title") if getattr(m, 'tags', None) else None
                    album = m.get("TALB") or m.tags.get("album") if getattr(m, 'tags', None) else None
                    # year parsing
                    y = m.get("TDRC") or m.tags.get("date") if getattr(m, 'tags', None) else None
                    if y:
                        try:
                            year = int(str(y))
                        except Exception:
                            year = None
        except Exception:
            logger.debug("Mutagen failed for %s", file_path)

        if not title:
            # Fallback: filename
            title = Path(file_path).stem

        cur.execute(
            "UPDATE songs SET title = ?, album = ?, release_year = ?, status_metadata = 'done' WHERE song_id = ?",
            (str(title), str(album) if album else None, year, song_id),
        )
        conn.commit()

    conn.close()
    return True
