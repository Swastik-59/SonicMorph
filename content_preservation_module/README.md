# Content Preservation Module

Builds a MusicGen-ready content package from three inputs:

1. Song-level musical features from the SQLite database
2. Melody stem audio (Demucs other.wav)
3. Artist style embedding vector (256-dim)

The output is a validated `ContentPackage` that can be passed into downstream generation.

## What This Module Does

- Loads song features from `dataset/sonicmorph.db`
- Parses and normalizes musical metadata (tempo, key, mode)
- Loads and standardizes melody audio to mono 32 kHz
- Builds style-aware text prompts using artist instrumentation templates
- Assembles everything into one strongly validated object

## Files

- `song_features.py`:
  - `SongFeatures` dataclass
  - Derived properties: `key_signature`, `tempo_description`

- `feature_loader.py`:
  - `FeatureLoader(db_path)`
  - `load(song_id)` returns `SongFeatures`
  - `exists(song_id)` checks DB presence

- `content_packager.py`:
  - `ContentPackage` dataclass (final payload)
  - `ContentPackager` orchestration class
  - Prompt building, melody loading, validation

- `test_content_packager.py`:
  - End-to-end smoke test for loading, packaging, and prompt rotation

- `__init__.py`:
  - Public exports: `SongFeatures`, `FeatureLoader`, `ContentPackage`, `ContentPackager`

## Data Contracts

### SongFeatures

Core fields:

- `duration` (float)
- `tempo` (float, must be positive)
- `key` (str)
- `mode` (str, normalized lowercase)
- `rms_mean` (float)
- `spectral_centroid_mean` (float)

Derived fields:

- `key_signature` (example: F major)
- `tempo_description` buckets:
  - very slow: tempo < 60
  - slow: tempo < 80
  - mid-tempo: tempo < 110
  - uptempo: tempo < 140
  - fast: tempo >= 140

### ContentPackage

Required shape and constraints:

- `text_prompt`: non-empty string
- `melody_wav`: torch tensor with shape [1, T]
- `melody_sr`: exactly 32000
- `duration`: 0 < duration <= 30
- `tempo_bpm`: positive
- `key_signature`: string
- `style_embedding`: torch tensor with shape [256]

Validation runs automatically during packaging.

## Audio Handling Rules

The packager expects the melody stem from Demucs:

- Use other.wav for melody/instrument guidance
- Do not use vocals.wav for this module

Audio processing behavior:

- Multi-channel inputs are mixed down to mono
- Audio is resampled to 32000 Hz if needed
- Audio is truncated to max 30 seconds

## Prompt Generation

Prompts are built from:

- Tempo descriptor from song tempo
- Key signature and mode
- Target artist name
- Artist-specific instrumentation description map

You can enable prompt variation with template rotation via the `rotate_prompt` flag.

## Quick Usage

Run from project root in your virtual environment.

    python -m content_preservation_module.test_content_packager

Programmatic usage example:

    from pathlib import Path
    import torch
    from content_preservation_module import FeatureLoader, ContentPackager

    loader = FeatureLoader(Path("dataset/sonicmorph.db"))
    features = loader.load("YOUR_SONG_ID")

    style_embedding = torch.load("models/style_embedder/geese.pt", weights_only=True)

    packager = ContentPackager()
    package = packager.package(
        features=features,
        melody_stem_path="path/to/other.wav",
        style_embedding=style_embedding,
        target_artist="geese",
        extra_tags=["warm", "punchy"],
        rotate_prompt=False,
    )

## Database Expectations

`FeatureLoader` expects a `features` table with at least these columns:

- `song_id`
- `duration`
- `tempo`
- `musical_key`
- `rms_mean`
- `spectral_centroid_mean`

If `musical_key` is missing or malformed, safe fallbacks are applied:

- key defaults to C
- mode defaults to major

## Common Errors

- Database not found:
  - Verify `dataset/sonicmorph.db` exists

- Song ID missing:
  - Confirm the `song_id` exists in the `features` table

- Invalid tempo:
  - Tempo must be greater than 0 in the DB

- Melody stem not found:
  - Ensure Demucs output path exists and points to other.wav

- Embedding shape mismatch:
  - Style embedding must be a 256-dim tensor

## Notes

- This module is deterministic for fixed inputs unless prompt rotation is enabled.
- Artist instrumentation defaults to a generic live-band profile when artist key is unknown.