from pathlib import Path

import torch

from content_preservation_module import (
    FeatureLoader,
    ContentPackager,
)

DB_PATH = (
    Path("dataset/sonicmorph.db")
)

SONG_ID = "5e74c5df0189453e9b57d2311151b100"

MELODY_STEM = Path(
    r"dataset\stems\geese\4d54490bb7fb430cb9642513efbd5efa\htdemucs_ft\Geese-Geese - Gravity Blues (Official Audio)-HY3CPF4FkYU_normalized\vocals.wav"
)

STYLE_EMBEDDING = (
    "models/style_embedder/geese.pt"
)

TARGET_ARTIST = "geese"


loader = FeatureLoader(DB_PATH)

features = loader.load(SONG_ID)

style_embedding = torch.load(
    STYLE_EMBEDDING,
    weights_only=False
)

packager = ContentPackager()

package = packager.package(
    features=features,
    melody_stem_path=MELODY_STEM,
    style_embedding=style_embedding,
    target_artist=TARGET_ARTIST,
)

print("\n===== CONTENT PACKAGE =====")

print(package.text_prompt)
print(package.duration)
print(package.tempo_bpm)
print(package.key_signature)
print(package.melody_wav.shape)
print(package.style_embedding.shape)