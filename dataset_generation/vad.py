from __future__ import annotations

import webrtcvad
import collections
import contextlib
import wave
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
from typing import List, Tuple


def _frame_generator(frame_duration_ms: int, audio: bytes, sample_rate: int):
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)  # 2 bytes per sample (16-bit)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / (2 * sample_rate))
    while offset + n <= len(audio):
        yield audio[offset:offset + n]
        timestamp += duration
        offset += n


def _bytes_from_float_array(arr: np.ndarray) -> bytes:
    # expects int16 array
    return arr.tobytes()


def extract_vocal_clips(vocal_path: Path, out_dir: Path, min_dur: float = 3.0, max_dur: float = 15.0, sample_rate: int = 16000, merge_gap: float = 0.5) -> List[Tuple[Path, float, float]]:
    """Extract voiced segments from a vocal stem using webrtcvad.

    Returns list of tuples (out_path, start_sec, end_sec).
    """
    vad = webrtcvad.Vad(2)
    # Load audio, resample to sample_rate, mono, 16-bit
    y, sr = sf.read(str(vocal_path), always_2d=False)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    if sr != sample_rate:
        y = librosa.resample(y.astype(float), orig_sr=sr, target_sr=sample_rate)
        sr = sample_rate
    # convert to int16
    y16 = (y * 32767.0).astype(np.int16)
    audio_bytes = y16.tobytes()

    frame_ms = 30
    frames = list(_frame_generator(frame_ms, audio_bytes, sr))
    voiced = [vad.is_speech(f, sr) for f in frames]

    # Convert frame indices to time ranges
    frame_duration = frame_ms / 1000.0
    segments = []
    start = None
    for i, is_voiced in enumerate(voiced):
        if is_voiced and start is None:
            start = i
        if (not is_voiced or i == len(voiced) - 1) and start is not None:
            end = i if not is_voiced else i
            s = start * frame_duration
            e = (end + 1) * frame_duration
            segments.append((s, e))
            start = None

    # Convert frame indices to time ranges
    frame_duration = frame_ms / 1000.0
    segments = []
    start = None
    for i, is_voiced in enumerate(voiced):
        if is_voiced and start is None:
            start = i
        if not is_voiced and start is not None:
            end = i
            s = start * frame_duration
            e = end * frame_duration
            segments.append((s, e))
            start = None
    
     # Merge close segments
    merged = []
    for s, e in segments:
         if not merged:
             merged.append([s, e])
         else:
             prev_s, prev_e = merged[-1]
             if s - prev_e <= merge_gap:
                 merged[-1][1] = e
             else:
                 merged.append([s, e])

     # Clip durations and export between min_dur and max_dur
    out_dir.mkdir(parents=True, exist_ok=True)
    results: List[Tuple[Path, float, float]] = []
    for s, e in merged:
         seg_dur = e - s
         if seg_dur < min_dur:
             continue
         # split long segments into multiple max_dur windows
         if seg_dur <= max_dur:
             start_t = s
             end_t = e
             # write segment
             segment = y16[int(start_t * sr):int(end_t * sr)]
             # convert back to 16k float for writing at 44100 later
             segment_f = segment.astype(np.float32) / 32767.0
             # resample back to 44100 for model usage
             segment_resampled = librosa.resample(segment_f, orig_sr=sr, target_sr=44100)
             out_name = f"clip_{int(start_t*1000)}_{int(end_t*1000)}.wav"
             out_path = out_dir / out_name
             sf.write(str(out_path), segment_resampled, 44100, subtype='PCM_16')
             results.append((out_path, start_t, end_t))
         else:
             # split into windows
             pos = s
             while pos + min_dur < e:
                 end_win = min(pos + max_dur, e)
                 if end_win - pos >= min_dur:
                     segment = y16[int(pos * sr):int(end_win * sr)]
                     segment_f = segment.astype(np.float32) / 32767.0
                     segment_resampled = librosa.resample(segment_f, orig_sr=sr, target_sr=44100)
                     out_name = f"clip_{int(pos*1000)}_{int(end_win*1000)}.wav"
                     out_path = out_dir / out_name
                     sf.write(str(out_path), segment_resampled, 44100, subtype='PCM_16')
                     results.append((out_path, pos, end_win))
                 pos += max_dur

    return results
