from dataclasses import dataclass


@dataclass
class SongFeatures:
    """
    Lightweight representation of song-level content features.
    Loaded from the SonicMorph features table.
    """

    duration: float
    tempo: float
    key: str
    mode: str

    rms_mean: float = 0.0
    spectral_centroid_mean: float = 0.0

    @property
    def key_signature(self) -> str:
        return f"{self.key} {self.mode}"