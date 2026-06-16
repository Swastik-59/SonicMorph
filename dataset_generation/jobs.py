from pathlib import Path
import sqlite3
import uuid
from datetime import datetime
from typing import Optional
from dataset_generation.config import DATASET_DIR


def _db_path():
    ds = Path(DATASET_DIR)
    return ds / "sonicmorph.db"


def create_job(stage: str, artist: Optional[str] = None, song: Optional[str] = None, conn: sqlite3.Connection | None = None) -> str:
    job_id = uuid.uuid4().hex
    close_conn = False
    if conn is None:
        conn = sqlite3.connect(str(_db_path()))
        close_conn = True
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO jobs (job_id, name, params, status, started_at, attempts) VALUES (?, ?, ?, ?, ?, ?)",
        (job_id, stage, f"artist={artist}\nsong={song}" if artist or song else None, "pending", None, 0),
    )
    conn.commit()
    if close_conn:
        conn.close()
    return job_id


def start_job(job_id: str, conn: sqlite3.Connection | None = None):
    close_conn = False
    if conn is None:
        conn = sqlite3.connect(str(_db_path()))
        close_conn = True
    cur = conn.cursor()
    cur.execute("UPDATE jobs SET status = 'running', started_at = ? WHERE job_id = ?", (datetime.utcnow().isoformat(), job_id))
    conn.commit()
    if close_conn:
        conn.close()


def complete_job(job_id: str, success: bool = True, conn: sqlite3.Connection | None = None):
    close_conn = False
    if conn is None:
        conn = sqlite3.connect(str(_db_path()))
        close_conn = True
    cur = conn.cursor()
    status = 'completed' if success else 'failed'
    cur.execute("UPDATE jobs SET status = ?, completed_at = ? WHERE job_id = ?", (status, datetime.utcnow().isoformat(), job_id))
    conn.commit()
    if close_conn:
        conn.close()


def get_job(stage: str, conn: sqlite3.Connection | None = None):
    close_conn = False
    if conn is None:
        conn = sqlite3.connect(str(_db_path()))
        close_conn = True
    cur = conn.cursor()
    cur.execute("SELECT job_id, status FROM jobs WHERE name = ? ORDER BY started_at DESC LIMIT 1", (stage,))
    row = cur.fetchone()
    if close_conn:
        conn.close()
    return row
