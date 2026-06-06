import sqlite3
from pathlib import Path
from .config import DATASET_DIR, config

SCHEMA = """
CREATE TABLE IF NOT EXISTS artists (
    artist_id           TEXT PRIMARY KEY,
    name                TEXT NOT NULL UNIQUE,
    enabled             INTEGER NOT NULL DEFAULT 1,
    target_song_count   INTEGER NOT NULL DEFAULT 100,
    created_at          TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS songs (
    song_id             TEXT PRIMARY KEY,
    artist_id           TEXT NOT NULL REFERENCES artists(artist_id),
    title               TEXT,
    album               TEXT,
    release_year        INTEGER,
    duration            REAL,
    sample_rate         INTEGER,
    channels            INTEGER,
    bitrate             INTEGER,
    source              TEXT NOT NULL,
    source_url          TEXT,
    source_id           TEXT,
    file_path           TEXT,
    file_hash           TEXT,
    fingerprint         TEXT,
    collection_date     TEXT NOT NULL DEFAULT (datetime('now')),
    status_validated    TEXT NOT NULL DEFAULT 'pending',
    status_deduplicated TEXT NOT NULL DEFAULT 'pending',
    status_normalized   TEXT NOT NULL DEFAULT 'pending',
    status_separated    TEXT NOT NULL DEFAULT 'pending',
    status_metadata     TEXT NOT NULL DEFAULT 'pending',
    status_features     TEXT NOT NULL DEFAULT 'pending',
    status_packaged     TEXT NOT NULL DEFAULT 'pending',
    is_duplicate        INTEGER NOT NULL DEFAULT 0,
    duplicate_of        TEXT REFERENCES songs(song_id),
    validation_errors   TEXT,
    error_log           TEXT,
    dataset_version     TEXT NOT NULL DEFAULT 'v1',
    updated_at          TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS stems (
    stem_id     TEXT PRIMARY KEY,
    song_id     TEXT NOT NULL REFERENCES songs(song_id),
    stem_type   TEXT NOT NULL,
    file_path   TEXT NOT NULL,
    duration    REAL,
    sample_rate INTEGER,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS features (
    feature_id              TEXT PRIMARY KEY,
    song_id                 TEXT NOT NULL REFERENCES songs(song_id),
    tempo                   REAL,
    musical_key             TEXT,
    mfcc_mean               TEXT,
    mfcc_std                TEXT,
    chroma_mean             TEXT,
    rms_mean                REAL,
    rms_std                 REAL,
    spectral_centroid_mean  REAL,
    spectral_centroid_std   REAL,
    duration                REAL,
    created_at              TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS clips (
    clip_id     TEXT PRIMARY KEY,
    song_id     TEXT NOT NULL REFERENCES songs(song_id),
    target      TEXT NOT NULL,
    file_path   TEXT NOT NULL,
    start_time  REAL NOT NULL,
    end_time    REAL NOT NULL,
    duration    REAL NOT NULL,
    source_type TEXT,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id          TEXT PRIMARY KEY,
    stage           TEXT NOT NULL,
    dataset_version TEXT NOT NULL DEFAULT 'v1',
    started_at      TEXT NOT NULL DEFAULT (datetime('now')),
    completed_at    TEXT,
    status          TEXT NOT NULL DEFAULT 'running',
    songs_processed INTEGER DEFAULT 0,
    songs_failed    INTEGER DEFAULT 0,
    error_message   TEXT
);

CREATE TABLE IF NOT EXISTS downloads (
    download_id TEXT PRIMARY KEY,
    song_id     TEXT REFERENCES songs(song_id),
    source      TEXT,
    url         TEXT,
    file_path   TEXT,
    status      TEXT DEFAULT 'pending',
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS jobs (
    job_id      TEXT PRIMARY KEY,
    name        TEXT,
    params      TEXT,
    status      TEXT DEFAULT 'pending',
    started_at  TEXT,
    completed_at TEXT,
    attempts    INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_songs_artist ON songs(artist_id);
CREATE INDEX IF NOT EXISTS idx_songs_status ON songs(status_validated, status_separated);
CREATE INDEX IF NOT EXISTS idx_stems_song ON stems(song_id);
CREATE INDEX IF NOT EXISTS idx_features_song ON features(song_id);
CREATE INDEX IF NOT EXISTS idx_clips_target ON clips(target);
CREATE INDEX IF NOT EXISTS idx_clips_song ON clips(song_id);
CREATE INDEX IF NOT EXISTS idx_songs_packaged ON songs(status_packaged);
CREATE INDEX IF NOT EXISTS idx_songs_normalized ON songs(status_normalized);
CREATE INDEX IF NOT EXISTS idx_songs_duplicate ON songs(is_duplicate);
CREATE INDEX IF NOT EXISTS idx_downloads_song ON downloads(song_id);
"""


def get_db_path():
    dataset_dir = Path(DATASET_DIR)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    return dataset_dir / "sonicmorph.db"


def init_db():
    db_path = get_db_path()
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    # Enable foreign keys and set WAL journaling for better concurrency
    cur.execute('PRAGMA foreign_keys = ON')
    cur.execute('PRAGMA journal_mode = WAL')
    cur.execute('PRAGMA synchronous = NORMAL')
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()
    return db_path


if __name__ == "__main__":
    print("Initializing database at:", init_db())
