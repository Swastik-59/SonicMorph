"""
Quick smoke-test for the content preservation module.
Run from the project root:
    python -m content_preservation_module.test_content_packager
"""

from pathlib import Path

import torch

from content_preservation_module import (
    FeatureLoader,
    ContentPackager,
)

# ── config ────────────────────────────────────────────────
DB_PATH    = Path("dataset/sonicmorph.db")
SONG_ID    = "5e74c5df0189453e9b57d2311151b100"

# NOTE: use other.wav (the melody/instrument stem), NOT vocals.wav.
# vocals.wav is reserved for the RVC voice conversion component.
MELODY_STEM = Path(
    r"dataset\stems\geese"
    r"\4d54490bb7fb430cb9642513efbd5efa"
    r"\htdemucs_ft"
    r"\Geese-Geese - Gravity Blues (Official Audio)-HY3CPF4FkYU_normalized"
    r"\other.wav"
)

STYLE_EMBEDDING_PATH = Path("models/style_embedder/geese.pt")
TARGET_ARTIST        = "geese"

# ── load ──────────────────────────────────────────────────
print("Loading features from database...")
loader   = FeatureLoader(DB_PATH)
features = loader.load(SONG_ID)

print(f"  Duration : {features.duration:.1f} s")
print(f"  Tempo    : {features.tempo:.1f} BPM  ({features.tempo_description})")
print(f"  Key      : {features.key_signature}")
print(f"  RMS      : {features.rms_mean:.4f}")

print("\nLoading style embedding...")
style_embedding = torch.load(
    str(STYLE_EMBEDDING_PATH),
    weights_only=True,
)
print(f"  Shape    : {style_embedding.shape}")

# ── package ───────────────────────────────────────────────
print("\nBuilding content package...")
packager = ContentPackager()
package  = packager.package(
    features         = features,
    melody_stem_path = MELODY_STEM,
    style_embedding  = style_embedding,
    target_artist    = TARGET_ARTIST,
)

# ── results ───────────────────────────────────────────────
print("\n" + "=" * 55)
print("  CONTENT PACKAGE — FULL SUMMARY")
print("=" * 55)
print(f"  Prompt        : {package.text_prompt}")
print(f"  Duration      : {package.duration} s")
print(f"  Tempo         : {package.tempo_bpm} BPM")
print(f"  Key           : {package.key_signature}")
print(f"  Melody shape  : {package.melody_wav.shape}")
print(f"  Style emb     : {package.style_embedding.shape}")
print("=" * 55)

# ── prompt variation test ─────────────────────────────────
print("\nPrompt rotation test (3 templates):")
for i in range(3):
    p = packager.package(
        features         = features,
        melody_stem_path = MELODY_STEM,
        style_embedding  = style_embedding,
        target_artist    = TARGET_ARTIST,
        rotate_prompt    = True,
    )
    print(f"\n  [{i+1}] {p.text_prompt}")

print("\nAll checks passed.")