from artist_style_embedder.style_embedder import (
    ArtistAudioDataset,
    StyleEncoder
)

import torch

dataset = ArtistAudioDataset(
    "dataset/processed/style_embedder",
    augment=True
)

mel1, mel2, _ = dataset[0]

print("Input shape:", mel1.shape)

model = StyleEncoder()

with torch.no_grad():
    z = model(mel1.unsqueeze(0))

print("Embedding shape:", z.shape)