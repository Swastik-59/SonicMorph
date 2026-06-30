"""
SonicMorph — MusicGen Fine-Tuner
Optimised for RTX 4060 8 GB / Windows

Usage
-----
# Fine-tune on one artist (always start here):
python -m musicgen_finetuner.finetune --artist the_beatles

# Fine-tune on ALL artists sequentially:
python -m musicgen_finetuner.finetune --all

# Resume an interrupted run:
python -m musicgen_finetuner.finetune --artist the_beatles   # auto-resumes

# Start fresh (ignore checkpoints):
python -m musicgen_finetuner.finetune --artist the_beatles --fresh

# Test inference after training:
python -m musicgen_finetuner.finetune --artist the_beatles --test-only
"""

import argparse
import os
import signal
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
DATA_DIR      = PROJECT_ROOT / "dataset" / "processed" / "musicgen"
MODEL_DIR     = PROJECT_ROOT / "models" / "musicgen"
SAMPLE_DIR    = PROJECT_ROOT / "models" / "musicgen" / "samples"

# ── training defaults (tuned for 8 GB VRAM) ──────────────────────────────────
DEFAULT_BASE_MODEL  = "facebook/musicgen-small"   # 300 M params — fits in 8 GB
DEFAULT_EPOCHS      = 15      # 15-20 is plenty for a ~700-clip per-artist set;
                               # diminishing returns past this for style fine-tuning
DEFAULT_BATCH_SIZE  = 2       # per-step batch; effective = batch × grad_accum
DEFAULT_GRAD_ACCUM  = 8       # effective batch size = 16
DEFAULT_LR          = 5e-5
DEFAULT_SAVE_EVERY  = 5       # save checkpoint every N epochs

ALL_ARTISTS = [
    "arctic_monkeys", "geese", "kanye_west", "nirvana",
    "queen", "radiohead", "the_beatles", "the_strokes",
]

# ── graceful interrupt ────────────────────────────────────────────────────────
_STOP = False

def _handle_sigint(sig, frame):
    global _STOP
    print("\n\n  Ctrl+C — finishing this batch then saving...\n")
    _STOP = True

signal.signal(signal.SIGINT, _handle_sigint)

# ── GPU optimisations ─────────────────────────────────────────────────────────
torch.backends.cudnn.benchmark        = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True


# ── checkpoint helpers ────────────────────────────────────────────────────────
def ckpt_dir(artist: str) -> Path:
    d = MODEL_DIR / artist
    d.mkdir(parents=True, exist_ok=True)
    return d


def find_checkpoint(artist: str) -> Path | None:
    """Priority: paused → best → latest epoch."""
    d = ckpt_dir(artist)
    for name in ("paused.pt", "best.pt"):
        p = d / name
        if p.exists():
            return p
    candidates = sorted(d.glob("epoch_*.pt"))
    return candidates[-1] if candidates else None


def save_checkpoint(
    path:       Path,
    model:      nn.Module,
    opt:        torch.optim.Optimizer,
    scaler:     torch.cuda.amp.GradScaler,
    epoch:      int,
    loss:       float,
    artist:     str,
):
    torch.save({
        "epoch"      : epoch,
        "model_state": model.state_dict(),
        "optimizer"  : opt.state_dict(),
        "scaler"     : scaler.state_dict(),
        "loss"       : loss,
        "artist"     : artist,
    }, str(path))
    print(f"  Saved: {path.name}")


# ── conditioning helper ───────────────────────────────────────────────────────
def get_condition_tensors(model, descriptions: list[str], device: str):
    """
    Pass text descriptions through MusicGen's conditioning provider
    to get the cross-attention tensors the LM expects.
    """
    from audiocraft.modules.conditioners import ConditioningAttributes

    attributes = [
        ConditioningAttributes(text={"description": d})
        for d in descriptions
    ]
    tokenized         = model.lm.condition_provider.tokenize(attributes)
    condition_tensors = model.lm.condition_provider(tokenized)
    return condition_tensors


# ── training loop for one artist ─────────────────────────────────────────────
def train_one_artist(
    artist:       str,
    base_model:   str  = DEFAULT_BASE_MODEL,
    epochs:       int  = DEFAULT_EPOCHS,
    batch_size:   int  = DEFAULT_BATCH_SIZE,
    grad_accum:   int  = DEFAULT_GRAD_ACCUM,
    lr:           float = DEFAULT_LR,
    save_every:   int  = DEFAULT_SAVE_EVERY,
    fresh:        bool = False,
    test_only:    bool = False,
):
    global _STOP
    _STOP = False

    device  = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"

    print(f"\n{'='*60}")
    print(f"  MusicGen Fine-Tuner  |  Artist: {artist}")
    print(f"{'='*60}")
    print(f"  Base model  : {base_model}")
    print(f"  Device      : {device}")
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU         : {props.name}  ({props.total_memory/1e9:.1f} GB)")
    print(f"  Batch size  : {batch_size}  ×  grad_accum {grad_accum} = effective {batch_size*grad_accum}")
    print(f"  Epochs      : {epochs}")
    print(f"{'='*60}\n")

    # ── load base model ───────────────────────────────────
    print(f"Loading {base_model}...")
    from audiocraft.models import MusicGen
    model = MusicGen.get_pretrained(base_model)
    # Force float32 master weights. autocast() only casts *ops* during the
    # forward pass — it assumes the underlying parameters/gradients are
    # float32. If the pretrained checkpoint loads as float16 (as it can
    # on some audiocraft versions when device='cuda'), GradScaler will
    # fail with "Attempting to unscale FP16 gradients." and the loss can
    # overflow to NaN in fp16 cross-entropy. Forcing fp32 here fixes both.
    model.lm = model.lm.to(device=device, dtype=torch.float32)
    model.compression_model = model.compression_model.to(device=device, dtype=torch.float32)

    # freeze EnCodec — only fine-tune the language model
    for p in model.compression_model.parameters():
        p.requires_grad_(False)
    model.compression_model.eval()

    # unfreeze LM
    model.lm.train()
    trainable = sum(p.numel() for p in model.lm.parameters() if p.requires_grad)
    print(f"  Trainable params (LM only): {trainable/1e6:.1f} M")

    if test_only:
        _run_test_inference(model, artist, device)
        return

    # ── dataset ───────────────────────────────────────────
    from musicgen_finetuner.dataset import MusicGenFineTuneDataset

    dataset = MusicGenFineTuneDataset(
        data_dir = str(DATA_DIR),
        artist   = artist,
        augment  = True,
    )
    loader = DataLoader(
        dataset,
        batch_size         = batch_size,
        shuffle            = True,
        num_workers        = 2,           # overlap audio load/resample/augment with GPU work
        persistent_workers = True,
        prefetch_factor    = 2,
        pin_memory         = use_amp,
        drop_last          = True,
    )
    print(f"  Batches per epoch: {len(loader)}")

    # ── PERF FIX: cache text conditioning ──────────────────
    # The text prompt is identical for every clip of this artist
    # (build_prompt(artist) never changes mid-training), but the
    # original loop called the T5 text encoder fresh on every single
    # step — 342 redundant forward passes per epoch for the exact
    # same input. Compute it once here and reuse the cached tensor
    # for every batch. Wrapped in no_grad() because the text encoder
    # is frozen; the LM's own cross-attention K/V weights (which DO
    # need gradients) still receive gradients normally, since they
    # operate on this cached tensor downstream inside compute_predictions.
    with torch.no_grad():
        cached_condition_tensors = get_condition_tensors(
            model, [dataset.prompt] * batch_size, device
        )
    print(f"  Cached text conditioning for prompt (saves a T5 pass every step)\n")

    # ── optimiser ─────────────────────────────────────────
    opt    = torch.optim.AdamW(model.lm.parameters(), lr=lr, weight_decay=0.01)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    start_epoch = 1
    best_loss   = float("inf")
    ckpt_path   = None if fresh else find_checkpoint(artist)

    # ── resume ────────────────────────────────────────────
    if ckpt_path:
        print(f"  Resuming from: {ckpt_path.name}")
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        model.lm.load_state_dict(ckpt["model_state"])
        opt.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_loss   = ckpt.get("loss", float("inf"))
        for _ in range(ckpt["epoch"]):
            sched.step()
        print(f"  Resumed — next epoch: {start_epoch} | best loss: {best_loss:.4f}\n")
    else:
        print("  No checkpoint found — starting fresh.\n")

    if start_epoch > epochs:
        print(f"  Already completed {epochs} epochs.")
        _run_test_inference(model, artist, device)
        return

    print(f"  Ctrl+C at any time → saves and exits cleanly.\n")

    # ── training ──────────────────────────────────────────
    for epoch in range(start_epoch, epochs + 1):
        model.lm.train()
        total_loss  = 0.0
        n_batches   = 0
        t0          = time.time()

        opt.zero_grad(set_to_none=True)

        pbar = tqdm(
            loader,
            desc         = f"Epoch {epoch:3d}/{epochs}",
            dynamic_ncols = True,
            leave         = True,
        )

        for step, (audio, descriptions) in enumerate(pbar):
            audio = audio.to(device, non_blocking=True)   # [B, 1, T]

            if torch.isnan(audio).any():
                print("NaNs detected in input audio")
                continue

            if torch.isinf(audio).any():
                print("Inf detected in input audio")
                continue

            try:
                # tokenise audio with frozen EnCodec
                with torch.no_grad():
                    codes, _ = model.compression_model.encode(audio)  # [B, K, T_codes]

                # ── FIX: build conditioning AND run the forward pass
                # inside the SAME autocast context. Previously the LM
                # forward call was dedented outside the `with` block,
                # so it ran in float32 while condition_tensors were
                # float16 — causing "expected Float but found Half".
                with torch.cuda.amp.autocast(enabled=use_amp):
                    # Use the cached conditioning computed once before
                    # the training loop, instead of re-running T5 here.
                    condition_tensors = cached_condition_tensors

                    # forward pass through LM
                    lm_out = model.lm.compute_predictions(
                        codes=codes,
                        conditions=[],
                        condition_tensors=condition_tensors,
                    )

                    # logits: [B, K, T, Card]
                    logits = lm_out.logits.float()

                    # ── FIX: MusicGen's delay pattern fills early
                    # positions in some codebooks with a special
                    # placeholder token equal to `card` (one index past
                    # the valid vocabulary, e.g. 2048 for a 2048-size
                    # codebook). compute_predictions() returns `mask`
                    # to tell you which (B, K, T) positions are real
                    # audio codes vs. placeholder. Previously we ran
                    # cross_entropy over EVERY position, including the
                    # placeholder ones — which means the target index
                    # (2048) was out of range for logits that only have
                    # `card` (2048) classes. That out-of-bounds index is
                    # undefined behaviour on GPU and silently produced
                    # NaN on every single step. Selecting only the
                    # valid (masked) positions before computing the
                    # loss fixes this at the source.
                    mask = lm_out.mask  # [B, K, T] bool — True = valid

                    targets = codes  # [B, K, T]

                    logits_valid  = logits[mask]   # [N_valid, Card]
                    targets_valid = targets[mask]  # [N_valid]

                    loss = torch.nn.functional.cross_entropy(
                        logits_valid,
                        targets_valid,
                    )

                    loss = loss / grad_accum

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"\n  Skipping step {step}: loss is {loss.item()}")
                    opt.zero_grad(set_to_none=True)
                    continue

                scaler.scale(loss).backward()

                if (step + 1) % grad_accum == 0 or (step + 1) == len(loader):
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.lm.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)

                total_loss += loss.item() * grad_accum
                n_batches  += 1
                pbar.set_postfix(loss=f"{loss.item()*grad_accum:.4f}")

            except torch.cuda.OutOfMemoryError:
                print(
                    "\n  OOM — clearing cache and skipping this batch.\n"
                    "  If this happens repeatedly, reduce --batch_size to 1."
                )
                torch.cuda.empty_cache()
                opt.zero_grad(set_to_none=True)
                continue

            if _STOP:
                break

        # ── epoch summary ─────────────────────────────────
        avg_loss   = total_loss / max(n_batches, 1)
        epoch_time = (time.time() - t0) / 60
        sched.step()
        is_best    = avg_loss < best_loss

        print(
            f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | "
            f"LR: {sched.get_last_lr()[0]:.2e} | "
            f"Time: {epoch_time:.1f} min"
        )

        d = ckpt_dir(artist)

        if is_best:
            best_loss = avg_loss
            save_checkpoint(
                d / "best.pt", model.lm, opt, scaler, epoch, avg_loss, artist
            )

        if epoch % save_every == 0:
            save_checkpoint(
                d / f"epoch_{epoch:03d}.pt",
                model.lm, opt, scaler, epoch, avg_loss, artist
            )

        # ── Ctrl+C handler ────────────────────────────────
        if _STOP:
            save_checkpoint(
                d / "paused.pt", model.lm, opt, scaler, epoch, avg_loss, artist
            )
            print(f"\n  Paused at epoch {epoch}.")
            print(f"  Re-run the same command to resume from epoch {epoch+1}.\n")
            return

    # ── generate a test sample on completion ──────────────
    print(f"\n  Training complete. Best loss: {best_loss:.4f}")
    _run_test_inference(model, artist, device)


# ── test inference ────────────────────────────────────────────────────────────
def _run_test_inference(model, artist: str, device: str):
    """Generate a short sample to verify the fine-tuned model works."""
    from musicgen_finetuner.dataset import build_prompt

    print(f"\n  Generating test sample for '{artist}'...")
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    model.lm.eval()
    model.compression_model.eval()

    prompt = build_prompt(artist)
    print(f"  Prompt: {prompt[:80]}...")

    model.set_generation_params(
        duration    = 10,       # 10-second test clip
        top_k       = 250,
        temperature = 1.0,
        cfg_coef    = 3.0,
    )

    with torch.no_grad():
        output = model.generate(
            descriptions = [prompt],
            progress     = True,
        )

    out_path = SAMPLE_DIR / f"{artist}_test_sample.wav"
    torchaudio.save(
        str(out_path),
        output[0].cpu(),
        sample_rate = model.sample_rate,
    )
    print(f"  Sample saved: {out_path}\n")


# ── entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="SonicMorph MusicGen Fine-Tuner"
    )
    parser.add_argument("--artist",      type=str,   default=None,
                        help="Single artist key, e.g. the_beatles")
    parser.add_argument("--all",         action="store_true",
                        help="Fine-tune all artists sequentially")
    parser.add_argument("--fresh",       action="store_true",
                        help="Ignore existing checkpoints, start from scratch")
    parser.add_argument("--test-only",   action="store_true",
                        help="Skip training, just generate a test sample")
    parser.add_argument("--base-model",  type=str,   default=DEFAULT_BASE_MODEL)
    parser.add_argument("--epochs",      type=int,   default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size",  type=int,   default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--grad-accum",  type=int,   default=DEFAULT_GRAD_ACCUM)
    parser.add_argument("--lr",          type=float, default=DEFAULT_LR)
    parser.add_argument("--save-every",  type=int,   default=DEFAULT_SAVE_EVERY)
    args = parser.parse_args()

    if not args.artist and not args.all:
        parser.error("Provide --artist <name>  or  --all")

    artists = ALL_ARTISTS if args.all else [args.artist]

    for artist in artists:
        train_one_artist(
            artist      = artist,
            base_model  = args.base_model,
            epochs      = args.epochs,
            batch_size  = args.batch_size,
            grad_accum  = args.grad_accum,
            lr          = args.lr,
            save_every  = args.save_every,
            fresh       = args.fresh,
            test_only   = args.test_only,
        )
        if _STOP:
            print("  Stopped by user.")
            break


if __name__ == "__main__":
    main()