import logging
import sqlite3
from pathlib import Path
from difflib import SequenceMatcher
import subprocess
import json
from sonicmorph.config import DATASET_DIR
from sonicmorph.utils import compute_file_hash

logger = logging.getLogger(__name__)


def _compute_fp_with_fpcalc(path: str) -> str | None:
    try:
        res = subprocess.run(["fpcalc", "-json", path], capture_output=True, text=True, check=True)
        data = json.loads(res.stdout)
        return data.get("fingerprint")
    except Exception:
        return None


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def run(config, db_conn=None):
    logger.info("Running duplicate detection stage (two-stage)")
    ds = Path(DATASET_DIR)
    db_path = ds / "sonicmorph.db"
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    # Stage 1: fast checks (file_hash, metadata/duration)
    cur.execute("SELECT song_id, artist_id, file_path, file_hash, title, duration FROM songs WHERE status_deduplicated = 'pending'")
    rows = cur.fetchall()
    if not rows:
        logger.info("No songs pending deduplication")
        return True

    # Build quick lookup by file_hash and (title, artist, duration bucket)
    hash_map = {}
    for song_id, artist_id, file_path, file_hash, title, duration in rows:
        if file_hash:
            if file_hash in hash_map:
                # Mark duplicate
                cur.execute("UPDATE songs SET is_duplicate = 1, duplicate_of = ? , status_deduplicated = 'done' WHERE song_id = ?", (hash_map[file_hash], song_id))
                conn.commit()
                continue
            hash_map[file_hash] = song_id

    # For survivors, attempt fingerprint (expensive) if enabled
    fingerprint_enabled = config.pipeline.get("deduplication", {}).get("enable_fingerprint", True) if hasattr(config,'pipeline') else True
    threshold = config.pipeline.get("deduplication", {}).get("fingerprint_threshold", 0.85) if hasattr(config,'pipeline') else 0.85

    cur.execute("SELECT song_id, file_path, fingerprint FROM songs WHERE status_deduplicated = 'pending' AND is_duplicate = 0")
    survivors = cur.fetchall()
    for song_id, file_path, fingerprint in survivors:
        if fingerprint:
            # compare against others
            cur2 = conn.cursor()
            cur2.execute("SELECT song_id, fingerprint FROM songs WHERE fingerprint IS NOT NULL AND song_id != ?", (song_id,))
            for other_id, other_fp in cur2.fetchall():
                sim = _similarity(fingerprint, other_fp)
                if sim >= threshold:
                    cur.execute("UPDATE songs SET is_duplicate = 1, duplicate_of = ?, status_deduplicated='done' WHERE song_id = ?", (other_id, song_id))
                    conn.commit()
                    break
            continue

        if fingerprint_enabled:
            fp = _compute_fp_with_fpcalc(file_path)
            if fp:
                cur.execute("UPDATE songs SET fingerprint = ? WHERE song_id = ?", (fp, song_id))
                conn.commit()
                # compare now
                cur2 = conn.cursor()
                cur2.execute("SELECT song_id, fingerprint FROM songs WHERE fingerprint IS NOT NULL AND song_id != ?", (song_id,))
                for other_id, other_fp in cur2.fetchall():
                    sim = _similarity(fp, other_fp)
                    if sim >= threshold:
                        cur.execute("UPDATE songs SET is_duplicate = 1, duplicate_of = ?, status_deduplicated='done' WHERE song_id = ?", (other_id, song_id))
                        conn.commit()
                        break

        # If we reach here and not marked duplicate, mark as done
        cur.execute("UPDATE songs SET status_deduplicated = 'done' WHERE song_id = ?", (song_id,))
        conn.commit()

    conn.close()
    return True
