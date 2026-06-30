"""
SonicMorph — MusicGen Inference
Connects: ContentPackager + StyleEmbedder + Fine-tuned MusicGen

Usage
-----
python -m musicgen_finetuner.inference \
    --source   "dataset/stems/geese/.../other.wav" \
    --artist   geese \
    --song-id  5e74c5df0189453e9b57d2311151b100

python -m musicgen_finetuner.inference \
    --source   "dataset/stems/nirvana/.../other.wav" \
    --artist   the_beatles \
    --song-id  <song_id_in_db>
"""

import argparse
from pathlib import Path

import torch
import torchaudio

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── default paths ─────────────────────────────────────────────────────────────
DB_PATH        = PROJECT_ROOT / "dataset" / "sonicmorph.db"
EMBEDDER_DIR   = PROJECT_ROOT / "models" / "style_embedder"
MUSICGEN_DIR   = PROJECT_ROOT / "models" / "musicgen"
OUTPUT_DIR     = PROJECT_ROOT / "output" / "musicgen"
BASE_MODEL     = "facebook/musicgen-melody"   # melody-conditioned variant


# ── helpers ───────────────────────────────────────────────────────────────────
def load_artist_embedding(artist: str, device: str) -> torch.Tensor:
    path = EMBEDDER_DIR / f"{artist}.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"No style embedding for '{artist}' at {path}.\n"
            "Run the style embedder first."
        )
    emb = torch.load(str(path), weights_only=True).to(device)
    print(f"  Style embedding loaded: {emb.shape}  ({artist})")
    return emb


def load_finetuned_musicgen(artist: str, device: str):
    """
    Load MusicGen-Melody and overlay fine-tuned LM weights
    for the target artist if they exist.
    Falls back to base model if no fine-tuned checkpoint is found.
    """
    from audiocraft.models import MusicGen

    print(f"  Loading base model: {BASE_MODEL}...")
    model = MusicGen.get_pretrained(BASE_MODEL)
    model.lm                = model.lm.to(device)
    model.compression_model = model.compression_model.to(device)

    # look for fine-tuned weights
    ckpt_dir = MUSICGEN_DIR / artist
    ckpt     = None
    for name in ("best.pt", "paused.pt"):
        p = ckpt_dir / name
        if p.exists():
            ckpt = p
            break
    if ckpt is None:
        candidates = sorted(ckpt_dir.glob("epoch_*.pt"))
        if candidates:
            ckpt = candidates[-1]

    if ckpt:
        print(f"  Loading fine-tuned LM weights: {ckpt.name}...")
        state = torch.load(str(ckpt), map_location=device, weights_only=False)
        model.lm.load_state_dict(state["model_state"])
        print(f"  Fine-tuned weights applied.")
    else:
        print(
            f"  WARNING: No fine-tuned checkpoint found for '{artist}'.\n"
            f"  Using base MusicGen — output will not match artist style.\n"
            f"  Run:  python -m musicgen_finetuner.finetune --artist {artist}"
        )

    model.lm.eval()
    model.compression_model.eval()
    return model


# ── main inference function ───────────────────────────────────────────────────
def generate_cover(
    source_stem_path: str,     # other.wav from Demucs (melody stem)
    target_artist:    str,     # e.g. "the_beatles"
    song_id:          str,     # song ID in sonicmorph.db
    duration:         float  = 30.0,
    temperature:      float  = 1.0,
    cfg_coef:         float  = 3.0,
    top_k:            int    = 250,
    output_path:      str  = None,
) -> Path:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*60}")
    print(f"  SonicMorph Inference")
    print(f"  Source : {Path(source_stem_path).name}")
    print(f"  Artist : {target_artist}")
    print(f"{'='*60}\n")

    # ── 1. Load content package (uses content_preservation_module) ────────────
    print("[1/4] Building content package...")
    from content_preservation_module import FeatureLoader, ContentPackager

    features = FeatureLoader(DB_PATH).load(song_id)
    packager = ContentPackager()
    package  = packager.package(
        features         = features,
        melody_stem_path = source_stem_path,
        style_embedding  = torch.zeros(256),   # placeholder — real one loaded below
        target_artist    = target_artist.replace("_", " ").title(),
    )

    # ── 2. Load artist style embedding (uses style_embedder) ─────────────────
    print("\n[2/4] Loading style embedding...")
    style_emb = load_artist_embedding(target_artist, device)

    # attach real embedding to package
    package.style_embedding = style_emb

    # ── 3. Load fine-tuned MusicGen ──────────────────────────────────────────
    print("\n[3/4] Loading fine-tuned MusicGen...")
    model = load_finetuned_musicgen(target_artist, device)

    # ── 4. Generate ──────────────────────────────────────────────────────────
    print(f"\n[4/4] Generating {duration:.0f}s cover...")
    print(f"  Prompt   : {package.text_prompt[:80]}...")
    print(f"  Melody   : {package.melody_wav.shape}  @{package.melody_sr} Hz")
    print(f"  Style    : {package.style_embedding.shape}")

    model.set_generation_params(
        duration    = min(duration, package.duration),
        top_k       = top_k,
        temperature = temperature,
        cfg_coef    = cfg_coef,
    )

    melody = package.melody_wav.unsqueeze(0).to(device)  # [1, 1, T]

    with torch.no_grad():
        output = model.generate_with_chroma(
            descriptions       = [package.text_prompt],
            melody_wavs        = melody,
            melody_sample_rate = package.melody_sr,
            progress           = True,
        )

    # ── save output ───────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if output_path is None:
        src_name   = Path(source_stem_path).parent.parent.name[:12]
        output_path = str(
            OUTPUT_DIR / f"{src_name}_as_{target_artist}.wav"
        )

    torchaudio.save(
        output_path,
        output[0].cpu(),
        sample_rate = model.sample_rate,
    )

    print(f"\n  Done!  Saved: {output_path}")
    return Path(output_path)


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="SonicMorph — Generate a cover with fine-tuned MusicGen"
    )
    parser.add_argument(
        "--source", required=True,
        help="Path to other.wav (Demucs melody stem of the source song)"
    )
    parser.add_argument(
        "--artist", required=True,
        help="Target artist key, e.g. the_beatles, nirvana"
    )
    parser.add_argument(
        "--song-id", required=True,
        help="song_id in sonicmorph.db for the source song"
    )
    parser.add_argument("--duration",    type=float, default=30.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--cfg-coef",    type=float, default=3.0)
    parser.add_argument("--top-k",       type=int,   default=250)
    parser.add_argument("--output",      type=str,   default=None)
    args = parser.parse_args()

    generate_cover(
        source_stem_path = args.source,
        target_artist    = args.artist,
        song_id          = args.song_id,
        duration         = args.duration,
        temperature      = args.temperature,
        cfg_coef         = args.cfg_coef,
        top_k            = args.top_k,
        output_path      = args.output,
    )


if __name__ == "__main__":
    main()