#!/usr/bin/env python3
"""
SonicMorph — Artist Style Embedder
Optimised for Windows + NVIDIA GPU

Workflow:
  1. python style_embedder.py --precompute   # convert WAVs to .pt mels (run ONCE)
  2. python style_embedder.py                # train (auto-resumes from checkpoint)
  3. python style_embedder.py --embed-only   # cache artist embeddings after training

Other flags:
  --fresh          ignore all checkpoints, start from scratch
  --epochs N       default 80
  --batch_size N   default 128
  --lr F           default 3e-4
  --temperature F  default 0.07
"""

import sys, os, random, signal, argparse, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ── PATHS ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT    = Path(__file__).resolve().parent.parent
DEFAULT_WAV_DIR = PROJECT_ROOT / "dataset" / "processed" / "style_embedder"
DEFAULT_MEL_DIR = PROJECT_ROOT / "dataset" / "processed" / "style_embedder_mels"
DEFAULT_MDL_DIR = PROJECT_ROOT / "models"  / "style_embedder"

# ── WINDOWS WORKER COUNT ──────────────────────────────────────────────────────
# Pre-computed .pt dataset has NO nn.Module → safe for workers on Windows
NUM_WORKERS = 4 if sys.platform == "win32" else 6

# ── GPU OPTIMISATIONS ─────────────────────────────────────────────────────────
torch.backends.cudnn.benchmark        = True   # auto-tune conv kernels
torch.backends.cuda.matmul.allow_tf32 = True   # faster matmul on Ampere
torch.backends.cudnn.allow_tf32       = True

# ── GRACEFUL INTERRUPT ────────────────────────────────────────────────────────
_INTERRUPTED = False

def _handle_sigint(sig, frame):
    global _INTERRUPTED
    print("\n\n  Ctrl+C caught — saving checkpoint after this batch...\n")
    _INTERRUPTED = True

signal.signal(signal.SIGINT, _handle_sigint)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — PRE-COMPUTE MEL SPECTROGRAMS  (run once before training)
# ══════════════════════════════════════════════════════════════════════════════
def precompute_mels(
    wav_dir: str = str(DEFAULT_WAV_DIR),
    mel_dir: str = str(DEFAULT_MEL_DIR),
    sr:      int = 22050,
    n_mels:  int = 128,
    n_fft:   int = 2048,
    hop:     int = 512,
):
    """
    Reads every WAV clip, computes a log-mel spectrogram, saves as float16 .pt.
    Skips files that already exist — safe to re-run after interruptions.
    Disk cost: ~55 KB per clip  x  37,611 clips  ≈  2 GB total.
    """
    mel_fn = nn.Sequential(
        T.MelSpectrogram(sample_rate=sr, n_fft=n_fft, hop_length=hop,
                         n_mels=n_mels, f_min=20, f_max=8000),
        T.AmplitudeToDB(top_db=80),
    )

    wav_path = Path(wav_dir)
    mel_path = Path(mel_dir)

    artist_dirs = sorted(d for d in wav_path.iterdir() if d.is_dir())
    if not artist_dirs:
        raise RuntimeError(f"No artist directories found in {wav_dir}")

    converted = skipped = errors = 0

    for artist_dir in artist_dirs:
        out_dir = mel_path / artist_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        wavs = sorted(artist_dir.glob("*.wav"))
        print(f"  {artist_dir.name}: {len(wavs)} clips")

        for wav_file in tqdm(wavs, desc=f"  {artist_dir.name}", leave=False):
            out_file = out_dir / (wav_file.stem + ".pt")
            if out_file.exists():
                skipped += 1
                continue
            try:
                waveform, file_sr = torchaudio.load(str(wav_file))
                if file_sr != sr:
                    waveform = T.Resample(file_sr, sr)(waveform)
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                with torch.no_grad():
                    mel = mel_fn(waveform)           # [1, n_mels, T_frames]
                torch.save(mel.half(), str(out_file))  # float16 saves ~50% disk
                converted += 1
            except Exception as e:
                print(f"    WARN: {wav_file.name}: {e}")
                errors += 1

    print(f"\n  Done — {converted} converted | {skipped} skipped | {errors} errors")
    print(f"  Output: {mel_dir}\n")


# ══════════════════════════════════════════════════════════════════════════════
# MODEL DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════
class MelSpectrogramExtractor(nn.Module):
    """Used only for inference / embedding computation (not in training dataset)."""
    def __init__(self, sr=22050, n_mels=128, n_fft=2048, hop=512):
        super().__init__()
        self.mel   = T.MelSpectrogram(sample_rate=sr, n_fft=n_fft,
                                       hop_length=hop, n_mels=n_mels,
                                       f_min=20, f_max=8000)
        self.to_db = T.AmplitudeToDB(top_db=80)
    def forward(self, x): return self.to_db(self.mel(x))


class StyleEncoder(nn.Module):
    """CNN + Transformer → 256-dim L2-normalised style embedding."""

    def __init__(self, n_mels=128, embed_dim=256, n_heads=8, n_layers=4):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1,   32,  3, stride=2, padding=1), nn.BatchNorm2d(32),  nn.GELU(),
            nn.Conv2d(32,  64,  3, stride=2, padding=1), nn.BatchNorm2d(64),  nn.GELU(),
            nn.Conv2d(64,  128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.GELU(),
        )
        self.pos_embed  = nn.Parameter(torch.randn(1, 512, 256) * 0.02)
        enc_layer       = nn.TransformerEncoderLayer(
            d_model=256, nhead=n_heads, dim_feedforward=1024,
            dropout=0.1, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.projection  = nn.Sequential(
            nn.Linear(256, 512), nn.GELU(), nn.Linear(512, embed_dim),
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = self.cnn(mel)                       # [B, 256, H', T']
        x = x.mean(dim=2).permute(0, 2, 1)     # [B, T', 256]
        S = x.shape[1]
        if S > self.pos_embed.shape[1]:
            raise RuntimeError(f"Seq len {S} > pos_embed max {self.pos_embed.shape[1]}")
        x = x + self.pos_embed[:, :S]
        x = self.transformer(x).mean(dim=1)     # [B, 256]
        return F.normalize(self.projection(x), dim=-1)


class NTXentLoss(nn.Module):
    """SimCLR NT-Xent contrastive loss."""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.T = temperature

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        B   = z_i.shape[0]
        z   = torch.cat([z_i, z_j], dim=0)
        sim = torch.mm(z, z.T) / self.T
        lbl = torch.cat([torch.arange(B) + B, torch.arange(B)]).to(z.device)
        sim.masked_fill_(torch.eye(2 * B, dtype=torch.bool, device=z.device),
                         float("-inf"))
        return F.cross_entropy(sim, lbl)


# ══════════════════════════════════════════════════════════════════════════════
# DATASET  — loads .pt mels, NO nn.Module → safe for Windows multiprocessing
# ══════════════════════════════════════════════════════════════════════════════
class PrecomputedMelDataset(Dataset):
    """
    Loads float16 mel .pt files written by precompute_mels().
    __getitem__ is pure tensor ops — no audio I/O, no nn.Module.
    This is what allows num_workers > 0 on Windows.
    """
    TARGET_FRAMES = 216     # 5 s × 22050 Hz / hop 512 ≈ 216 frames

    def __init__(self, mel_dir: str, augment: bool = True):
        self.mel_dir = Path(mel_dir)
        self.augment = augment
        self.clips:  list[Path] = []
        self.labels: list[int]  = []

        artist_dirs = sorted(d for d in self.mel_dir.iterdir() if d.is_dir())
        if not artist_dirs:
            raise RuntimeError(
                f"No artist directories found in {mel_dir}.\n"
                "Run:  python style_embedder.py --precompute"
            )

        self.artists   = artist_dirs
        self.n_classes = len(artist_dirs)

        for label, adir in enumerate(artist_dirs):
            pts = sorted(adir.glob("*.pt"))
            if not pts:
                print(f"  WARNING: no .pt files in {adir.name} — run --precompute first")
                continue
            for p in pts:
                self.clips.append(p)
                self.labels.append(label)

        if not self.clips:
            raise RuntimeError(
                "No .pt files found anywhere.\n"
                "Run:  python style_embedder.py --precompute"
            )

        print(f"Dataset ready: {len(self.clips):,} clips | {self.n_classes} artists")
        for i, d in enumerate(self.artists):
            cnt = self.labels.count(i)
            print(f"  [{i}] {d.name:<25} {cnt:>6} clips")

    def __len__(self) -> int:
        return len(self.clips)

    def _random_window(self, mel: torch.Tensor) -> torch.Tensor:
        T = mel.shape[2]
        if T >= self.TARGET_FRAMES:
            s = random.randint(0, T - self.TARGET_FRAMES)
            return mel[:, :, s : s + self.TARGET_FRAMES].clone()
        return F.pad(mel, (0, self.TARGET_FRAMES - T))

    def _augment_mel(self, mel: torch.Tensor) -> torch.Tensor:
        """SpecAugment-style augmentation — pure tensor ops, runs in worker process."""
        mel = mel.float()

        # Frequency masking (wipe random band of mel bins)
        if random.random() > 0.4:
            f_w = random.randint(4, 20)
            f_s = random.randint(0, 128 - f_w)
            mel[:, f_s : f_s + f_w, :] = 0.0

        # Time masking (wipe random time band)
        if random.random() > 0.4:
            t_w = random.randint(5, 30)
            t_s = random.randint(0, self.TARGET_FRAMES - t_w)
            mel[:, :, t_s : t_s + t_w] = 0.0

        # Amplitude jitter ±3 dB
        if random.random() > 0.5:
            mel = mel + random.uniform(-3.0, 3.0)

        # Vertical pitch-shift by ±4 mel bins (cheap roll)
        if random.random() > 0.5:
            mel = torch.roll(mel, random.randint(-4, 4), dims=1)

        return mel.half()

    def __getitem__(self, idx: int):
        mel = torch.load(self.clips[idx], weights_only=True)  # float16 [1,128,T]

        m1 = self._random_window(mel)
        m2 = self._random_window(mel)   # independent random window = second view

        if self.augment:
            m1 = self._augment_mel(m1)
            m2 = self._augment_mel(m2)

        return m1.float(), m2.float(), self.labels[idx]


# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def find_resume_checkpoint(save_dir: str) -> str | None:
    """
    Priority: paused_checkpoint.pt → best_model.pt → latest epoch_XXX.pt
    paused_checkpoint.pt is written on Ctrl+C and is always the most recent epoch.
    """
    sd = Path(save_dir)
    for name in ("paused_checkpoint.pt", "best_model.pt"):
        p = sd / name
        if p.exists():
            return str(p)
    candidates = sorted(sd.glob("epoch_*.pt"))
    return str(candidates[-1]) if candidates else None


def _save(path: str, model, opt, scaler, epoch: int,
          loss: float, dataset: PrecomputedMelDataset):
    torch.save({
        "epoch"       : epoch,
        "model_state" : model.state_dict(),
        "optimizer"   : opt.state_dict(),
        "scaler"      : scaler.state_dict(),
        "loss"        : loss,
        "n_artists"   : dataset.n_classes,
        "artists"     : [d.name for d in dataset.artists],
    }, path)


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════════
def train_style_embedder(
    mel_dir:     str   = str(DEFAULT_MEL_DIR),
    save_dir:    str   = str(DEFAULT_MDL_DIR),
    epochs:      int   = 80,
    batch_size:  int   = 128,
    lr:          float = 3e-4,
    temperature: float = 0.07,
    fresh:       bool  = False,
):
    global _INTERRUPTED
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"

    print(f"\n{'='*62}")
    print(f"  SonicMorph — Style Embedder")
    print(f"{'='*62}")
    print(f"  Device      : {device}")
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU         : {props.name}")
        print(f"  VRAM        : {props.total_memory / 1e9:.1f} GB")
    print(f"  AMP (fp16)  : {use_amp}")
    print(f"  Workers     : {NUM_WORKERS}")
    print(f"  Batch size  : {batch_size}  (effective negatives per sample: {2*batch_size-2})")
    print(f"  Mel dir     : {mel_dir}")
    print(f"  Save dir    : {save_dir}")
    print(f"{'='*62}\n")

    # ── dataset & loader ──────────────────────────────────────────────────────
    dataset = PrecomputedMelDataset(mel_dir, augment=True)
    loader  = DataLoader(
        dataset,
        batch_size         = batch_size,
        shuffle            = True,
        num_workers        = NUM_WORKERS,
        pin_memory         = use_amp,
        drop_last          = True,              # uniform batch size for NT-Xent
        persistent_workers = NUM_WORKERS > 0,  # keep workers alive between epochs
        prefetch_factor    = 2 if NUM_WORKERS > 0 else None,
    )

    # ── model & optimiser ─────────────────────────────────────────────────────
    model   = StyleEncoder(embed_dim=256).to(device)
    loss_fn = NTXentLoss(temperature=temperature)
    opt     = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)

    os.makedirs(save_dir, exist_ok=True)
    start_epoch = 1
    best_loss   = float("inf")

    # ── auto-resume ───────────────────────────────────────────────────────────
    if not fresh:
        ckpt_path = find_resume_checkpoint(save_dir)
        if ckpt_path:
            print(f"  Resuming from: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state"])
            opt.load_state_dict(ckpt["optimizer"])
            if "scaler" in ckpt:
                scaler.load_state_dict(ckpt["scaler"])
            start_epoch = ckpt["epoch"] + 1
            best_loss   = ckpt.get("loss", float("inf"))
            # Restore scheduler to correct LR position
            for _ in range(ckpt["epoch"]):
                sched.step()
            print(f"  Resumed — next epoch: {start_epoch} | best loss so far: {best_loss:.4f}\n")
        else:
            print("  No checkpoint found — starting fresh.\n")
    else:
        print("  --fresh: ignoring existing checkpoints.\n")

    if start_epoch > epochs:
        print(f"  Already finished {epochs} epochs. Nothing to do.")
        print(f"  Run --embed-only to compute artist embeddings.\n")
        return model

    n_batches_per_epoch = len(loader)
    print(f"  Epochs {start_epoch} → {epochs}  |  {n_batches_per_epoch} batches/epoch")
    print(f"  Ctrl+C at any time → saves checkpoint and exits cleanly.\n")

    # ── training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches  = 0
        t0         = time.time()

        pbar = tqdm(loader,
                    desc        = f"Epoch {epoch:3d}/{epochs}",
                    leave       = True,
                    dynamic_ncols= True)

        for mel1, mel2, _ in pbar:
            mel1 = mel1.to(device, non_blocking=True)
            mel2 = mel2.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = loss_fn(model(mel1), model(mel2))

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            total_loss += loss.item()
            n_batches  += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            if _INTERRUPTED:
                break

        # ── end-of-epoch stats ────────────────────────────────────────────────
        avg_loss   = total_loss / max(n_batches, 1)
        epoch_secs = time.time() - t0
        sched.step()
        is_best    = avg_loss < best_loss

        print(
            f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | "
            f"LR: {sched.get_last_lr()[0]:.2e} | "
            f"Time: {epoch_secs/60:.1f} min"
        )

        # ── save best ─────────────────────────────────────────────────────────
        if is_best:
            best_loss = avg_loss
            _save(os.path.join(save_dir, "best_model.pt"),
                  model, opt, scaler, epoch, avg_loss, dataset)
            print(f"  Saved: best_model.pt  ← new best (loss {best_loss:.4f})")

        # ── save every 10 epochs ──────────────────────────────────────────────
        if epoch % 10 == 0:
            name = f"epoch_{epoch:03d}.pt"
            _save(os.path.join(save_dir, name),
                  model, opt, scaler, epoch, avg_loss, dataset)
            print(f"  Saved: {name}")

        # ── handle Ctrl+C: save pause checkpoint and exit cleanly ─────────────
        if _INTERRUPTED:
            pause_path = os.path.join(save_dir, "paused_checkpoint.pt")
            _save(pause_path, model, opt, scaler, epoch, avg_loss, dataset)
            print(f"\n  Saved paused checkpoint → {pause_path}")
            print(f"  Next run will auto-resume from epoch {epoch + 1}.")
            print(f"  Just run:  python style_embedder.py\n")
            sys.exit(0)

    print(f"\n  Training complete.  Best loss: {best_loss:.4f}")
    print(f"  Best model: {os.path.join(save_dir, 'best_model.pt')}\n")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# ARTIST EMBEDDING COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════
def compute_artist_embedding(
    model:          StyleEncoder,
    artist_mel_dir: str,
    n_samples:      int = 200,
    device:         str = "cuda",
) -> torch.Tensor:
    TARGET   = 216
    pt_files = sorted(Path(artist_mel_dir).glob("*.pt"))
    if not pt_files:
        raise RuntimeError(f"No .pt files in {artist_mel_dir}")

    random.shuffle(pt_files)
    pt_files   = pt_files[:n_samples]
    model.eval()
    embeddings = []

    with torch.no_grad():
        for f in tqdm(pt_files, desc=f"  {Path(artist_mel_dir).name}", leave=False):
            try:
                mel = torch.load(str(f), weights_only=True).float()
                T   = mel.shape[2]
                for _ in range(3):       # 3 random windows per file
                    s   = random.randint(0, max(0, T - TARGET))
                    win = mel[:, :, s : s + TARGET]
                    if win.shape[2] < TARGET:
                        win = F.pad(win, (0, TARGET - win.shape[2]))
                    z = model(win.unsqueeze(0).to(device))
                    embeddings.append(z.squeeze(0).cpu())
            except Exception as e:
                print(f"    Skipping {f.name}: {e}")

    if not embeddings:
        raise RuntimeError(f"No embeddings computed for {artist_mel_dir}")

    centroid = F.normalize(torch.stack(embeddings).mean(0, keepdim=True), dim=-1).squeeze(0)
    print(f"    {len(embeddings)} windows → {centroid.shape}")
    return centroid


def compute_all_artist_embeddings(
    model_ckpt: str,
    mel_dir:    str = str(DEFAULT_MEL_DIR),
    save_dir:   str = str(DEFAULT_MDL_DIR),
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model from: {model_ckpt}")
    ckpt  = torch.load(model_ckpt, map_location=device, weights_only=False)
    model = StyleEncoder(embed_dim=256).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    for adir in sorted(Path(mel_dir).iterdir()):
        if not adir.is_dir():
            continue
        print(f"\n  {adir.name}")
        emb  = compute_artist_embedding(model, str(adir), device=device)
        path = os.path.join(save_dir, f"{adir.name}.pt")
        torch.save(emb, path)
        print(f"  Saved: {path}")

    print(f"\n  All embeddings saved to {save_dir}/")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SonicMorph Style Embedder")
    parser.add_argument("--precompute",  action="store_true",
                        help="Convert WAV clips to mel .pt files (run once)")
    parser.add_argument("--fresh",       action="store_true",
                        help="Start training from scratch, ignore checkpoints")
    parser.add_argument("--embed-only",  action="store_true",
                        help="Skip training, just compute & save artist embeddings")
    parser.add_argument("--epochs",      type=int,   default=80)
    parser.add_argument("--batch_size",  type=int,   default=128)
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--wav_dir",     type=str,   default=str(DEFAULT_WAV_DIR))
    parser.add_argument("--mel_dir",     type=str,   default=str(DEFAULT_MEL_DIR))
    parser.add_argument("--save_dir",    type=str,   default=str(DEFAULT_MDL_DIR))
    args = parser.parse_args()

    if args.precompute:
        precompute_mels(wav_dir=args.wav_dir, mel_dir=args.mel_dir)

    elif args.embed_only:
        best = os.path.join(args.save_dir, "best_model.pt")
        if not os.path.isfile(best):
            print(f"ERROR: no model found at {best}")
            sys.exit(1)
        compute_all_artist_embeddings(best, args.mel_dir, args.save_dir)

    else:
        trained = train_style_embedder(
            mel_dir     = args.mel_dir,
            save_dir    = args.save_dir,
            epochs      = args.epochs,
            batch_size  = args.batch_size,
            lr          = args.lr,
            temperature = args.temperature,
            fresh       = args.fresh,
        )
        # Auto-compute embeddings when training finishes naturally
        best = os.path.join(args.save_dir, "best_model.pt")
        if os.path.isfile(best) and not _INTERRUPTED:
            print("Computing artist embeddings from best model...")
            compute_all_artist_embeddings(best, args.mel_dir, args.save_dir)