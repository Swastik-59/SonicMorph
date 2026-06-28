from dataclasses import dataclass
from pathlib import Path

import torch
import torchaudio

from content_preservation_module.song_features import (
    SongFeatures,
)


@dataclass
class ContentPackage:
    """
    Everything required for downstream generation.
    """

    text_prompt: str

    melody_wav: torch.Tensor
    melody_sr: int

    duration: float
    tempo_bpm: float
    key_signature: str

    style_embedding: torch.Tensor


class ContentPackager:

    TARGET_SR = 32000
    MAX_DURATION = 30

    INSTRUMENT_MAP = {

        "the_beatles":
            "jangly electric guitar, "
            "Rickenbacker guitars, "
            "lush harmonies, Mellotron",

        "nirvana":
            "distorted electric guitars, "
            "grunge drums, raw energy",

        "radiohead":
            "ambient guitars, synthesizers, "
            "atmospheric textures",

        "queen":
            "multi-layer vocal harmonies, "
            "classic rock instrumentation",

        "kanye_west":
            "hip-hop drums, deep bass, "
            "modern production",

        "arctic_monkeys":
            "indie rock guitars, driving bass, "
            "tight drums",

        "geese":
            "experimental rock instrumentation, "
            "dynamic arrangements",

        "default":
            "live band, guitar, bass, drums"
    }

    def build_prompt(
        self,
        features: SongFeatures,
        target_artist: str,
        extra_tags: list[str] | None = None,
    ) -> str:

        if features.tempo < 80:
            tempo_desc = "slow"

        elif features.tempo < 130:
            tempo_desc = "mid-tempo"

        else:
            tempo_desc = "fast"

        artist_key = (
            target_artist.lower()
            .replace(" ", "_")
        )

        instrumentation = self.INSTRUMENT_MAP.get(
            artist_key,
            self.INSTRUMENT_MAP["default"]
        )

        prompt = (
            f"A {tempo_desc} {features.mode} song "
            f"in {features.key_signature}, "
            f"{features.tempo:.0f} BPM, "
            f"performed in the style of "
            f"{target_artist}. "
            f"Instrumentation: {instrumentation}. "
            f"Professional studio recording. "
            f"High quality mix and master."
        )

        if extra_tags:
            prompt += " " + ", ".join(extra_tags)

        return prompt

    def load_melody(
        self,
        melody_path: str | Path
    ):

        wav, sr = torchaudio.load(str(melody_path))

        if wav.shape[0] > 1:
            wav = wav.mean(
                dim=0,
                keepdim=True
            )

        if sr != self.TARGET_SR:

            resampler = torchaudio.transforms.Resample(
                sr,
                self.TARGET_SR
            )

            wav = resampler(wav)

        max_samples = (
            self.MAX_DURATION
            * self.TARGET_SR
        )

        wav = wav[:, :max_samples]

        return wav, self.TARGET_SR

    def package(
        self,
        features: SongFeatures,
        melody_stem_path: str,
        style_embedding: torch.Tensor,
        target_artist: str,
        extra_tags=None,
    ) -> ContentPackage:

        melody_wav, melody_sr = self.load_melody(
            melody_stem_path
        )

        prompt = self.build_prompt(
            features=features,
            target_artist=target_artist,
            extra_tags=extra_tags,
        )

        print("\nGenerated Prompt:")
        print(prompt)

        return ContentPackage(
            text_prompt=prompt,

            melody_wav=melody_wav,
            melody_sr=melody_sr,

            duration=min(
                features.duration,
                self.MAX_DURATION
            ),

            tempo_bpm=features.tempo,

            key_signature=(
                features.key_signature
            ),

            style_embedding=style_embedding,
        )