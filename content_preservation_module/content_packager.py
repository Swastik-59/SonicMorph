from dataclasses import dataclass
from pathlib import Path

import torch
import torchaudio

from content_preservation_module.song_features import SongFeatures


@dataclass
class ContentPackage:
    """
    Everything required for downstream MusicGen generation.
    """
    text_prompt:     str

    melody_wav:      torch.Tensor   # [1, T]  at 32 kHz
    melody_sr:       int            # always 32_000

    duration:        float          # seconds, capped at MAX_DURATION
    tempo_bpm:       float
    key_signature:   str            # e.g. "F major"

    style_embedding: torch.Tensor   # [256]

    # ── quick sanity check ────────────────────────────────
    def validate(self) -> None:
        """Raise if any field is obviously wrong."""
        assert self.melody_wav.dim() == 2,          "melody_wav must be [1, T]"
        assert self.melody_wav.shape[0] == 1,       "melody_wav must be mono"
        assert self.melody_sr == 32_000,            "melody_sr must be 32 000"
        assert 0 < self.duration <= 30,             "duration out of range"
        assert self.tempo_bpm > 0,                  "tempo_bpm must be positive"
        assert self.style_embedding.shape == (256,),"style_embedding must be [256]"
        assert self.text_prompt.strip(),            "text_prompt is empty"


class ContentPackager:
    """
    Assembles a SongFeatures + style embedding + melody stem
    into a ContentPackage ready for MusicGen.
    """

    TARGET_SR    = 32_000
    MAX_DURATION = 30           # MusicGen's per-pass generation limit (seconds)

    # ── artist instrument descriptions ────────────────────
    # Keys must match the directory names in your dataset exactly.
    INSTRUMENT_MAP: dict[str, str] = {

        "arctic_monkeys":
            "indie rock electric guitar, driving bass, "
            "tight punchy drums, Alex Turner vocals",

        "daft_punk":
            "analog synthesizers, vocoder harmonies, "
            "electronic dance beats, funky filtered bass",

        "geese":
            "experimental post-punk guitars, "
            "dynamic shifting arrangements, raw energy",

        "kanye_west":
            "hip-hop trap drums, deep 808 bass, "
            "soul samples, modern rap production",

        "nirvana":
            "heavy distorted electric guitar, "
            "grunge drums, loud-quiet-loud dynamics, "
            "Kurt Cobain raw vocals",

        "pink_floyd":
            "psychedelic electric guitar, Hammond organ, "
            "ambient soundscapes, epic arrangements",

        "queen":
            "multi-layered vocal harmonies, "
            "Brian May guitar, classic rock production, "
            "anthemic structure",

        "radiohead":
            "ambient guitars, glitchy electronics, "
            "atmospheric synth pads, Thom Yorke vocals",

        "tame_impala":
            "psychedelic synthesizers, layered guitars, "
            "reverb-drenched production, dreamy vocals",

        "the_beatles":
            "jangly Rickenbacker electric guitar, "
            "Mellotron, lush vocal harmonies, "
            "1960s British studio production",

        "the_strokes":
            "two-guitar garage rock, "
            "punchy minimalist bass, tight lo-fi drums, "
            "Julian Casablancas deadpan vocals",

        # ── fallback ──────────────────────────────────────
        "default":
            "live band, guitar, bass, drums, "
            "professional studio recording",
    }

    # ── prompt templates ─────────────────────────────────
    # Rotated per call to add variety across generations.
    _PROMPT_TEMPLATES = [
        (
            "A {tempo} song in {key}, {bpm:.0f} BPM, "
            "in the style of {artist}. "
            "Instrumentation: {instruments}. "
            "Professional studio recording, high quality mix."
        ),
        (
            "{artist} style cover. "
            "{tempo} {mode} song, key of {key}, {bpm:.0f} BPM. "
            "Instruments: {instruments}. "
            "Studio quality, full arrangement."
        ),
        (
            "Genre: {mode} rock/pop. Key: {key}. Tempo: {bpm:.0f} BPM. "
            "Artist style: {artist}. "
            "{instruments}. "
            "High fidelity studio production."
        ),
    ]
    _template_idx = 0

    # ── public API ────────────────────────────────────────
    def package(
        self,
        features:         SongFeatures,
        melody_stem_path: str | Path,
        style_embedding:  torch.Tensor,
        target_artist:    str,
        extra_tags:       list[str] | None = None,
        rotate_prompt:    bool = False,
    ) -> ContentPackage:
        """
        Build and return a fully populated ContentPackage.

        Args:
            features:          SongFeatures loaded from the database.
            melody_stem_path:  Path to the  other.wav  Demucs stem
                               (NOT vocals.wav — use vocals for RVC only).
            style_embedding:   256-dim tensor from the style embedder.
            target_artist:     Display name, e.g. "The Beatles".
            extra_tags:        Optional list of extra descriptors appended
                               to the prompt, e.g. ["reverb", "warm"].
            rotate_prompt:     If True, cycles through prompt templates so
                               consecutive calls produce varied prompts.
        """
        melody_wav, melody_sr = self._load_melody(melody_stem_path)
        text_prompt           = self._build_prompt(
            features, target_artist, extra_tags, rotate_prompt
        )

        duration = min(features.duration, self.MAX_DURATION)

        pkg = ContentPackage(
            text_prompt     = text_prompt,
            melody_wav      = melody_wav,
            melody_sr       = melody_sr,
            duration        = duration,
            tempo_bpm       = features.tempo,
            key_signature   = features.key_signature,
            style_embedding = style_embedding,
        )

        pkg.validate()   # catch obvious problems early

        print("\nGenerated Prompt:")
        print(f"  {text_prompt}")
        print(f"\nContent Package:")
        print(f"  Duration      : {pkg.duration:.1f} s")
        print(f"  Tempo         : {pkg.tempo_bpm:.1f} BPM")
        print(f"  Key           : {pkg.key_signature}")
        print(f"  Melody shape  : {pkg.melody_wav.shape}")
        print(f"  Style emb     : {pkg.style_embedding.shape}")

        return pkg

    def build_prompt(
        self,
        features:      SongFeatures,
        target_artist: str,
        extra_tags:    list[str] | None = None,
    ) -> str:
        """Public single-template prompt builder (no rotation)."""
        return self._build_prompt(features, target_artist, extra_tags,
                                  rotate=False)

    # ── private helpers ───────────────────────────────────
    def _build_prompt(
        self,
        features:      SongFeatures,
        target_artist: str,
        extra_tags:    list[str] | None,
        rotate:        bool,
    ) -> str:
        artist_key    = target_artist.lower().replace(" ", "_")
        instruments   = self.INSTRUMENT_MAP.get(
            artist_key, self.INSTRUMENT_MAP["default"]
        )

        if rotate:
            template = self._PROMPT_TEMPLATES[
                self._template_idx % len(self._PROMPT_TEMPLATES)
            ]
            ContentPackager._template_idx += 1
        else:
            template = self._PROMPT_TEMPLATES[0]

        # key_signature already contains mode, e.g. "F major"
        # We extract just the note for templates that need it separately.
        prompt = template.format(
            tempo       = features.tempo_description,   # "fast", "mid-tempo", etc.
            mode        = features.mode,                # "major" / "minor"
            key         = features.key_signature,       # "F major"
            bpm         = features.tempo,
            artist      = target_artist,
            instruments = instruments,
        )

        if extra_tags:
            prompt = prompt.rstrip(".") + ". " + ", ".join(extra_tags) + "."

        return prompt

    def _load_melody(
        self,
        melody_path: str | Path,
    ) -> tuple[torch.Tensor, int]:
        """
        Load the melody stem, convert to mono 32 kHz,
        and truncate to MAX_DURATION seconds.
        """
        path = Path(melody_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Melody stem not found: {path}\n"
                "Make sure Demucs has run and 'other.wav' exists."
            )

        wav, sr = torchaudio.load(str(path))

        # ── mono ──────────────────────────────────────────
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        # ── resample to TARGET_SR ─────────────────────────
        if sr != self.TARGET_SR:
            wav = torchaudio.transforms.Resample(sr, self.TARGET_SR)(wav)

        # ── truncate ──────────────────────────────────────
        max_samples = self.MAX_DURATION * self.TARGET_SR
        wav = wav[:, :max_samples]

        # ── guard against silent / empty stems ───────────
        if wav.shape[1] == 0:
            raise ValueError(
                f"Melody stem at {path} is empty after loading. "
                "Check the Demucs output."
            )

        rms = wav.pow(2).mean().sqrt().item()
        if rms < 1e-5:
            print(
                f"  WARNING: melody stem '{path.name}' appears nearly silent "
                f"(RMS={rms:.2e}). Is this the correct stem?"
            )

        return wav, self.TARGET_SR