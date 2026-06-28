# validate_embeddings.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from pathlib import Path

# ── resolve path relative to THIS file, not the working directory ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EMBED_DIR    = PROJECT_ROOT / "models" / "style_embedder"

print(f"Looking in: {EMBED_DIR}")

# ── load only artist embedding files (skip model/epoch checkpoints) ──
SKIP = {"best_model"}

artists    = []
embeddings = []

for pt_file in sorted(EMBED_DIR.glob("*.pt")):
    if pt_file.stem in SKIP or "epoch" in pt_file.stem or "paused" in pt_file.stem:
        continue
    emb = torch.load(str(pt_file), weights_only=True)
    if emb.dim() != 1 or emb.shape[0] != 256:
        print(f"  Skipping {pt_file.name} — unexpected shape {emb.shape}")
        continue
    artists.append(pt_file.stem)
    embeddings.append(emb.numpy())
    print(f"  Loaded: {pt_file.name}  {emb.shape}")

if not embeddings:
    raise RuntimeError(
        f"No artist embeddings found in {EMBED_DIR}\n"
        "Expected files like arctic_monkeys.pt, nirvana.pt etc."
    )

X = np.stack(embeddings)   # [N, 256]
print(f"\nLoaded {len(artists)} artist embeddings: {artists}\n")

# ── pairwise cosine similarity matrix ──────────────────────────────
norms = np.linalg.norm(X, axis=1, keepdims=True)
sim   = (X / norms) @ (X / norms).T

print("Cosine similarity matrix (diagonal = 1.0 = self):")
header = "".join(f"{a[:8]:>10}" for a in artists)
print(f"{'':20}{header}")
for i, row in enumerate(sim):
    vals = "".join(f"{v:10.3f}" for v in row)
    print(f"{artists[i]:<20}{vals}")

# flag suspiciously high off-diagonal similarities
print("\nOff-diagonal pairs above 0.7 (potential confusion):")
found_any = False
for i in range(len(artists)):
    for j in range(i + 1, len(artists)):
        if sim[i, j] > 0.7:
            print(f"  ⚠  {artists[i]}  ↔  {artists[j]}  :  {sim[i,j]:.3f}")
            found_any = True
if not found_any:
    print("  None — all artists are well separated!")

# ── t-SNE visualisation ────────────────────────────────────────────
# perplexity must be strictly less than n_samples
perplexity = min(5, len(artists) - 1)
tsne  = TSNE(n_components=2, perplexity=perplexity,
             random_state=42)
X_2d  = tsne.fit_transform(X)

fig, ax = plt.subplots(figsize=(10, 8))
colors  = plt.cm.Set1(np.linspace(0, 1, len(artists)))

for i, (name, point) in enumerate(zip(artists, X_2d)):
    ax.scatter(*point, s=300, color=colors[i], zorder=3, edgecolors="white",
               linewidths=1.5)
    ax.annotate(
        name.replace("_", "\n").title(),
        xy       = point,
        fontsize = 10,
        ha       = "center",
        va       = "bottom",
        xytext   = (0, 14),
        textcoords = "offset points",
        color    = colors[i],
        fontweight = "bold",
    )

ax.set_title("SonicMorph — Artist Style Embeddings (t-SNE)", fontsize=14, pad=16)
ax.axis("off")
fig.tight_layout()

out_path = EMBED_DIR / "embedding_tsne.png"
plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
plt.show()
print(f"\nPlot saved: {out_path}")