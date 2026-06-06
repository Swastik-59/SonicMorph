import logging
import sqlite3
import json
from statistics import mean
from pathlib import Path
from sonicmorph.config import DATASET_DIR
from sonicmorph.utils import ensure_dir
from sonicmorph.utils import slugify

logger = logging.getLogger(__name__)


def run(config, db_conn=None):
    logger.info("Running quality reporting stage")
    ds = Path(DATASET_DIR)
    reports_dir = ensure_dir(ds / "reports")
    manifest_path = ds / "manifests" / f"manifest_{config.pipeline.get('dataset', {}).get('version', 'v1')}.json"
    db_path = ds / "sonicmorph.db"
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    # Consume the manifest produced by s10 so reporting follows the packaging pipeline.
    manifest_entries = []
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as fh:
            manifest_entries = json.load(fh) or []

    if manifest_entries:
        artists = sorted({entry["artist_id"] for entry in manifest_entries if entry.get("artist_id")})
    else:
        cur.execute("SELECT DISTINCT artist_id FROM songs")
        artists = [r[0] for r in cur.fetchall()]
    pipeline_version = config.pipeline.get("dataset", {}).get("version", "v1") if hasattr(config,'pipeline') else "v1"

    for artist in artists:
        if manifest_entries:
            artist_song_ids = [entry["song_id"] for entry in manifest_entries if entry.get("artist_id") == artist and entry.get("song_id")]
            if artist_song_ids:
                placeholders = ",".join("?" for _ in artist_song_ids)
                cur.execute(
                    f"SELECT song_id, duration, is_duplicate, status_validated, status_separated FROM songs WHERE song_id IN ({placeholders})",
                    artist_song_ids,
                )
                rows = cur.fetchall()
            else:
                cur.execute("SELECT song_id, duration, is_duplicate, status_validated, status_separated FROM songs WHERE artist_id = ?", (artist,))
                rows = cur.fetchall()
        else:
            cur.execute("SELECT song_id, duration, is_duplicate, status_validated, status_separated FROM songs WHERE artist_id = ?", (artist,))
            rows = cur.fetchall()
        total = len(rows)
        if total == 0:
            continue
        durations = [r[1] for r in rows if r[1]]
        dup_count = sum(1 for r in rows if r[2])
        validated = sum(1 for r in rows if r[3] == 'done')
        separated = sum(1 for r in rows if r[4] == 'done')

        report = {
            "artist_id": artist,
            "pipeline_version": pipeline_version,
            "songs_collected": total,
            "valid_songs": validated,
            "rejected_songs": total - validated,
            "missing_stems": total - separated,
            "duplicate_percent": (dup_count / total) * 100,
            "average_duration": mean(durations) if durations else None,
            "coverage_score": None,  # depends on configured target per-artist
        }

        # Try to read target_song_count from config
        target = None
        cfg_artists = config.artists.get("artists") if hasattr(config,'artists') else []
        for a in cfg_artists:
            if a.get("name") and a.get("name").lower().replace(' ', '_') == artist:
                target = a.get("target_song_count")
                break
        if target:
            report["coverage_score"] = (validated / target) * 100

        path = reports_dir / f"report_{artist}.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        logger.info("Wrote quality report for %s -> %s", artist, path)

    # Aggregate CSV reports
    import csv
    # artist_coverage.csv
    cov_path = reports_dir / "artist_coverage.csv"
    with open(cov_path, "w", newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        writer.writerow(["artist_id", "songs_collected", "valid_songs", "coverage_score"])
        for artist in artists:
            cur.execute("SELECT COUNT(*), SUM(CASE WHEN status_validated='done' THEN 1 ELSE 0 END) FROM songs WHERE artist_id = ?", (artist,))
            total, valid = cur.fetchone()
            valid = valid or 0
            total = total or 0
            target = None
            cfg_artists = config.artists.get("artists") if hasattr(config,'artists') else []
            for a in cfg_artists:
                if a.get("name") and slugify(a.get("name")) == artist:
                    target = a.get("target_song_count")
                    break
            coverage = (valid / target * 100) if target else None
            writer.writerow([artist, total, valid, coverage])

    # dataset_health.csv
    health_path = reports_dir / "dataset_health.csv"
    with open(health_path, "w", newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        writer.writerow(["metric", "value"])
        cur.execute("SELECT COUNT(*) FROM songs")
        total_songs = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM songs WHERE is_duplicate = 1")
        dup_songs = cur.fetchone()[0]
        writer.writerow(["total_songs", total_songs])
        writer.writerow(["duplicate_songs", dup_songs])

    # missing_artists.csv (artists with zero collected)
    miss_path = reports_dir / "missing_artists.csv"
    with open(miss_path, "w", newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        writer.writerow(["artist_id", "target_song_count", "collected"])
        cfg_artists = config.artists.get("artists") if hasattr(config,'artists') else []
        for a in cfg_artists:
            aid = slugify(a.get("name")) if a.get("name") else None
            target = a.get("target_song_count")
            cur.execute("SELECT COUNT(*) FROM songs WHERE artist_id = ?", (aid,))
            collected = cur.fetchone()[0]
            if collected == 0:
                writer.writerow([aid, target, collected])

    # preprocessing_summary.csv (basic per-stage counts)
    prep_path = reports_dir / "preprocessing_summary.csv"
    with open(prep_path, "w", newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        writer.writerow(["stage", "completed", "pending"])
        stages = ["status_validated", "status_deduplicated", "status_normalized", "status_separated", "status_metadata", "status_features", "status_packaged"]
        for s in stages:
            cur.execute(f"SELECT SUM(CASE WHEN {s}='done' THEN 1 ELSE 0 END), SUM(CASE WHEN {s}='pending' THEN 1 ELSE 0 END) FROM songs")
            done, pend = cur.fetchone()
            writer.writerow([s, done or 0, pend or 0])

    logger.info("Wrote aggregate reports to %s", reports_dir)

    conn.close()
    return True
