"""
Custom Dataset for SonicMorph MusicGen fine-tuning.

Expects flat paired files in dataset/processed/musicgen/<artist>/:
    044983a7867a4e549bf4499c78ec8571_music_00000.json
    044983a7867a4e549bf4499c78ec8571_music_00000.wav
    ...
"""

import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset


# ── ARTIST → TEXT DESCRIPTION MAP ────────────────────────────────────────────
# Used to generate the text conditioning prompt from the artist name in JSON.
INSTRUMENT_MAP: dict[str, str] = {
    "arctic_monkeys":
        "indie rock electric guitar, driving bass, "
        "tight punchy drums, Alex Turner vocals, "
        "energetic British rock",
    "daft_punk":
        "analog synthesizers, vocoder harmonies, "
        "electronic dance beats, funky filtered bass, "
        "French house music",
    "geese":
        "experimental post-punk guitars, "
        "dynamic shifting arrangements, raw energy",
    "kanye_west":
        "hip-hop trap drums, deep 808 bass, "
        "soul samples, modern rap production",
    "nirvana":
        "heavy distorted electric guitar, "
        "grunge drums, loud-quiet-loud dynamics, "
        "raw punk energy",
    "pink_floyd":
        "psychedelic electric guitar, Hammond organ, "
        "ambient soundscapes, epic progressive arrangements",
    "queen":
        "multi-layered vocal harmonies, "
        "Brian May guitar tone, classic rock production, "
        "anthemic arena rock",
    "radiohead":
        "ambient guitars, glitchy electronics, "
        "atmospheric synth pads, melancholic alternative rock",
    "tame_impala":
        "psychedelic synthesizers, layered guitars, "
        "reverb-drenched production, dreamy indie pop",
    "the_beatles":
        "jangly Rickenbacker electric guitar, "
        "Mellotron, lush vocal harmonies, "
        "1960s British Invasion pop rock",
    "the_strokes":
        "two-guitar garage rock, punchy minimalist bass, "
        "tight lo-fi drums, indie rock",
    "default":
        "live band, electric guitar, bass, drums, "
        "professional studio recording",
}

TEMPO_BUCKETS = [
    (60,  "very slow"),
    (80,  "slow"),
    (110, "mid-tempo"),
    (140, "uptempo"),
    (999, "fast"),
]


def tempo_word(bpm: float) -> str:
    for ceiling, label in TEMPO_BUCKETS:
        if bpm < ceiling:
            return label
    return "fast"


def build_prompt(artist_key: str, duration: float = 20.0) -> str:
    instruments = INSTRUMENT_MAP.get(artist_key, INSTRUMENT_MAP["default"])
    artist_display = artist_key.replace("_", " ").title()
    return (
        f"A {artist_display} style song. "
        f"{instruments}. "
        f"Professional studio recording, high quality mix and master."
    )


# ── DATASET ───────────────────────────────────────────────────────────────────
class MusicGenFineTuneDataset(Dataset):
    """
    Loads paired (WAV, JSON) clips for a single target artist.
    Returns (audio_tensor [1, T], prompt_string).
    """

    TARGET_SR    = 32_000   # MusicGen's native sample rate
    MAX_DURATION = 20.0     # your clips are exactly 20 s

    def __init__(
        self,
        data_dir:  str,
        artist:    str,
        augment:   bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.artist   = artist
        self.augment  = augment
        self.prompt   = build_prompt(artist)

        # collect all JSON files that have a matching WAV
        artist_dir = self.data_dir / artist
        if not artist_dir.exists():
            raise FileNotFoundError(
                f"Artist directory not found: {artist_dir}\n"
                f"Expected: {self.data_dir}/<artist>/"
            )

        self.pairs: list[tuple[Path, Path]] = []
        for json_path in sorted(artist_dir.glob("*.json")):
            wav_path = json_path.with_suffix(".wav")
            if wav_path.exists():
                self.pairs.append((wav_path, json_path))
            else:
                print(f"  WARN: no WAV for {json_path.name} — skipping")

        if not self.pairs:
            raise RuntimeError(
                f"No WAV+JSON pairs found in {artist_dir}"
            )

        self.max_samples = int(self.MAX_DURATION * self.TARGET_SR)
        print(
            f"Dataset [{artist}]: {len(self.pairs)} clips | "
            f"prompt: '{self.prompt[:60]}...'"
        )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        wav_path, _ = self.pairs[idx]
        audio = self._load(wav_path)
        if self.augment:
            audio = self._augment(audio)
        return audio, self.prompt

    # ── private ───────────────────────────────────────────
    def _load(self, path: Path) -> torch.Tensor:
        audio, sr = torchaudio.load(str(path))

        # stereo → mono
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        # resample to 32 kHz
        if sr != self.TARGET_SR:
            audio = T.Resample(sr, self.TARGET_SR)(audio)

        # exact length
        if audio.shape[1] >= self.max_samples:
            # random crop during training
            start = random.randint(0, audio.shape[1] - self.max_samples)
            audio = audio[:, start : start + self.max_samples]
        else:
            audio = F.pad(audio, (0, self.max_samples - audio.shape[1]))

        return audio  # [1, max_samples]

    def _augment(self, audio: torch.Tensor) -> torch.Tensor:
        # ± 3 dB volume jitter
        if random.random() > 0.5:
            gain = 10 ** (random.uniform(-3.0, 3.0) / 20.0)
            audio = (audio * gain).clamp(-1.0, 1.0)
        return audio