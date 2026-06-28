from dataclasses import dataclass, field


@dataclass
class SongFeatures:
    """
    Lightweight representation of song-level content features.
    Loaded from the SonicMorph features table.
    """

    duration:               float
    tempo:                  float
    key:                    str
    mode:                   str

    rms_mean:               float = 0.0
    spectral_centroid_mean: float = 0.0

    # ── derived ───────────────────────────────────────────
    @property
    def key_signature(self) -> str:
        """e.g.  'F major'  or  'A minor'"""
        return f"{self.key} {self.mode}"

    @property
    def tempo_description(self) -> str:
        """Human-readable tempo bucket for prompt building."""
        if self.tempo < 60:
            return "very slow"
        if self.tempo < 80:
            return "slow"
        if self.tempo < 110:
            return "mid-tempo"
        if self.tempo < 140:
            return "uptempo"
        return "fast"

    def __post_init__(self):
        # Sanitise: tempo must be positive
        if self.tempo <= 0:
            raise ValueError(f"Invalid tempo: {self.tempo}")
        # Normalise mode casing
        self.mode = self.mode.lower().strip()
        self.key  = self.key.strip()