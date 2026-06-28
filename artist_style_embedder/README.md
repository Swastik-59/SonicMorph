# Artist Style Embedder

Trainable audio embedding model for SonicMorph. The script in this folder converts audio clips into mel spectrograms, trains a contrastive style encoder, and exports one 256-dimensional embedding per artist.

## What it does

The pipeline is built around three steps:

1. Precompute mel spectrograms from WAV files and store them as `.pt` tensors.
2. Train a CNN + Transformer style encoder with SimCLR-style NT-Xent loss.
3. Save one normalized embedding per artist for downstream similarity and visualization tasks.

The current training run finished with a best loss of `0.0031` and produced artist embeddings for:

- `arctic_monkeys`
- `geese`
- `kanye_west`
- `nirvana`
- `queen`
- `radiohead`
- `the_beatles`
- `the_strokes`

## Folder Layout

This folder contains:

- `style_embedder.py` - training, precompute, and embedding export script
- `validate_embeddings.py` - cosine similarity and t-SNE validation script
- `embedding_tsne.png` - generated visualization output

The script reads and writes data in these locations relative to the project root:

- Input WAVs: `dataset/processed/style_embedder/`
- Precomputed mels: `dataset/processed/style_embedder_mels/`
- Saved models and artist embeddings: `models/style_embedder/`

## Quick Start

### 1. Precompute mel tensors

Run this once before training. It converts each WAV file into a cached mel tensor.

```bash
python artist_style_embedder/style_embedder.py --precompute
```

### 2. Train the embedder

Run training after precomputing mels. The script resumes automatically from checkpoints unless you pass `--fresh`.

```bash
python artist_style_embedder/style_embedder.py
```

Useful flags:

- `--fresh` - ignore existing checkpoints and start from scratch
- `--epochs N` - default `80`
- `--batch_size N` - default `128`
- `--lr F` - default `3e-4`
- `--temperature F` - default `0.07`

### 3. Export artist embeddings only

After training, regenerate artist centroids from the best checkpoint.

```bash
python artist_style_embedder/style_embedder.py --embed-only
```

## Validation

Use the validation script to inspect the saved artist embeddings:

```bash
python artist_style_embedder/validate_embeddings.py
```

What it reports:

- The loaded embedding files and their shapes
- A cosine similarity matrix
- Any off-diagonal pairs above `0.7`
- A 2D t-SNE plot saved to `models/style_embedder/embedding_tsne.png`

In the current run, all off-diagonal cosine similarities stayed below `0.7`, which indicates the artist embeddings were well separated.

## Training Notes

- Input windows are sampled from precomputed mel tensors using random crops.
- Augmentation includes frequency masking, time masking, amplitude jitter, and a small pitch shift via mel-bin roll.
- Training uses mixed precision on CUDA when available.
- Checkpoints are saved automatically:
  - `best_model.pt` for the lowest loss seen so far
  - `epoch_XXX.pt` every 10 epochs
  - `paused_checkpoint.pt` when interrupted with Ctrl+C

## Output Files

After a successful run, expect these files in `models/style_embedder/`:

- `best_model.pt` - best model checkpoint
- `epoch_080.pt` - final epoch checkpoint, if training reaches epoch 80
- `<artist>.pt` - one normalized embedding per artist

## Implementation Summary

The model combines:

- A 4-layer CNN front end for local spectrogram features
- A Transformer encoder for temporal aggregation
- A projection head that outputs a 256-dim L2-normalized embedding

The loss is NT-Xent contrastive loss with two augmented views per sample.

## Reproducibility

The scripts are designed to be restart-friendly:

- Re-running `--precompute` skips existing mel tensors
- Training resumes from the latest checkpoint by default
- `--fresh` disables checkpoint resume behavior

If you want to extend the pipeline, the two main entry points are `style_embedder.py` for training and `validate_embeddings.py` for inspection.