import logging
import sqlite3
import json
import math
import subprocess
from pathlib import Path
from sonicmorph.config import DATASET_DIR
from sonicmorph.utils import ensure_dir, slugify, generate_id

logger = logging.getLogger(__name__)


def _ffmpeg_extract(in_path: str, out_path: str, start: float, duration: float, sample_rate: int = 44100):
    cmd = [
        "ffmpeg", "-hide_banner", "-y", "-ss", str(start), "-i", in_path,
        "-t", str(duration), "-ar", str(sample_rate), "-ac", "1", out_path
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except Exception:
        return False


def _sliding_windows(length: float, clip_dur: float, overlap: float):
    step = clip_dur - overlap
    if step <= 0:
        return []
    starts = []
    pos = 0.0
    while pos + 0.5 < length:
        starts.append(pos)
        pos += step
    return starts


def run(config, db_conn=None):
    logger.info("Running dataset packaging stage")
    ds = Path(DATASET_DIR)
    processed = ensure_dir(ds / "processed")
    style_dir = ensure_dir(processed / "style_embedder")
    musicgen_dir = ensure_dir(processed / "musicgen")
    rvc_dir = ensure_dir(processed / "rvc")
    manifests_dir = ensure_dir(ds / "manifests")

    db_path = ds / "sonicmorph.db"
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    dataset_targets = config.pipeline.get("dataset_targets", {}) if hasattr(config, "pipeline") else {}
    style_cfg = dataset_targets.get("style_embedder", {})
    music_cfg = dataset_targets.get("musicgen", {})
    rvc_cfg = dataset_targets.get("rvc", {})
    style_clip_duration = float(style_cfg.get("clip_duration", 5.0))
    style_clip_overlap = float(style_cfg.get("clip_overlap", 2.5))
    music_clip_duration = float(music_cfg.get("clip_duration", 20.0))
    music_clip_overlap = float(music_cfg.get("clip_overlap", 5.0))
    rvc_min_duration = float(rvc_cfg.get("min_duration", 5.0))

    # Style embedder: 5s clips with 2.5s overlap
    cur.execute("SELECT song_id, artist_id, file_path, duration FROM songs WHERE status_packaged = 'pending' AND is_duplicate = 0")
    rows = cur.fetchall()
    style_manifest = []
    music_manifest = []
    rvc_manifest = []

    for song_id, artist_id, file_path, duration in rows:
        artist_slug = slugify(artist_id)
        artist_style_dir = ensure_dir(style_dir / artist_slug)
        artist_music_dir = ensure_dir(musicgen_dir / artist_slug)

        dur = duration or 0.0
        # Style
        for idx, start in enumerate(_sliding_windows(dur, style_clip_duration, style_clip_overlap)):
            clip_name = f"{song_id}_style_{idx:05d}.wav"
            out_path = artist_style_dir / clip_name
            ok = _ffmpeg_extract(file_path, str(out_path), start, style_clip_duration)
            if ok:
                style_manifest.append({"clip_path": str(out_path), "artist": artist_id, "duration": style_clip_duration, "source_song": song_id})
                cur.execute("INSERT OR REPLACE INTO clips (clip_id, song_id, target, file_path, start_time, end_time, duration, source_type) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                            (generate_id(), song_id, 'style_embedder', str(out_path), start, start+style_clip_duration, style_clip_duration, 'full_mix'))
        # Musicgen: 20s clips with 5s overlap
        for idx, start in enumerate(_sliding_windows(dur, music_clip_duration, music_clip_overlap)):
            clip_name = f"{song_id}_music_{idx:05d}.wav"
            out_path = artist_music_dir / clip_name
            ok = _ffmpeg_extract(file_path, str(out_path), start, music_clip_duration)
            if ok:
                # metadata JSON
                meta = {
                    "artist": artist_id,
                    "song": song_id,
                    "duration": music_clip_duration,
                    "source": file_path,
                    "sample_rate": config.pipeline.get("audio", {}).get("sample_rate", 44100) if hasattr(config,'pipeline') else 44100,
                }
                json_path = out_path.with_suffix('.json')
                with open(json_path, 'w', encoding='utf-8') as fh:
                    json.dump(meta, fh, indent=2)
                music_manifest.append({"clip": str(out_path), "meta": str(json_path)})
                cur.execute("INSERT OR REPLACE INTO clips (clip_id, song_id, target, file_path, start_time, end_time, duration, source_type) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                            (generate_id(), song_id, 'musicgen', str(out_path), start, start+music_clip_duration, music_clip_duration, 'full_mix'))

        # RVC: find vocal stem
        stem_dir = ds / "stems" / artist_id / song_id
        vocal_file = None
        if stem_dir.exists():
            for p in stem_dir.rglob("*"):
                if p.is_file() and ("voc" in p.stem.lower() or "vocal" in p.stem.lower()):
                    vocal_file = str(p)
                    break

        if vocal_file:
            artist_rvc_dir = ensure_dir(rvc_dir / artist_slug)
            # Use webrtcvad to extract voiced segments from vocal stem
            try:
                from sonicmorph.vad import extract_vocal_clips
                min_dur = config.pipeline.get("rvc", {}).get("min_clip_secs", 3.0) if hasattr(config,'pipeline') else 3.0
                max_dur = config.pipeline.get("rvc", {}).get("max_clip_secs", 15.0) if hasattr(config,'pipeline') else 15.0
                merge_gap = config.pipeline.get("rvc", {}).get("merge_gap_secs", 0.5) if hasattr(config,'pipeline') else 0.5
                clips = extract_vocal_clips(Path(vocal_file), artist_rvc_dir, min_dur=min_dur, max_dur=max_dur, merge_gap=merge_gap)
                for out_path, s, e in clips:
                    rvc_manifest.append({"clip": str(out_path), "artist": artist_id, "song": song_id, "start": s, "end": e})
                    cur.execute("INSERT OR REPLACE INTO clips (clip_id, song_id, target, file_path, start_time, end_time, duration, source_type) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                                (generate_id(), song_id, 'rvc', str(out_path), float(s), float(e), float(e - s), 'vocals'))
            except Exception:
                # fallback: export a short excerpt using ffmpeg
                clip_name = f"{song_id}_vocal_00001.wav"
                out_path = artist_rvc_dir / clip_name
                ok = _ffmpeg_extract(vocal_file, str(out_path), 0, min(60.0, dur or 60.0))
                if ok:
                    fallback_dur = float(min(max(rvc_min_duration, 60.0), dur or 60.0))
                    rvc_manifest.append({"clip": str(out_path), "artist": artist_id, "song": song_id})
                    cur.execute("INSERT OR REPLACE INTO clips (clip_id, song_id, target, file_path, start_time, end_time, duration, source_type) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                                (generate_id(), song_id, 'rvc', str(out_path), 0.0, fallback_dur, fallback_dur, 'vocals'))

        cur.execute("UPDATE songs SET status_packaged = 'done' WHERE song_id = ?", (song_id,))
        conn.commit()

    # Write manifests
    style_path = manifests_dir / "style_embedder_manifest.json"
    with open(style_path, 'w', encoding='utf-8') as fh:
        json.dump(style_manifest, fh, indent=2)

    music_path = manifests_dir / "musicgen_manifest.json"
    with open(music_path, 'w', encoding='utf-8') as fh:
        json.dump(music_manifest, fh, indent=2)

    rvc_path = manifests_dir / "rvc_manifest.json"
    with open(rvc_path, 'w', encoding='utf-8') as fh:
        json.dump(rvc_manifest, fh, indent=2)

    conn.close()
    logger.info("Packaging complete: style=%s music=%s rvc=%s", style_path, music_path, rvc_path)
    return True
