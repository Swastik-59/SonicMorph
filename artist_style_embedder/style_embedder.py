import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_DATA_DIR = PROJECT_ROOT / "dataset" / "processed" / "style_embedder"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "style_embedder"

#  AUDIO PREPROCESSING 
class MelSpectrogramExtractor(nn.Module):
    """Converts raw waveform to log-mel spectrogram."""
    def __init__(self, sr=22050, n_mels=128, n_fft=2048, hop=512):
        super().__init__()
        self.mel = T.MelSpectrogram(
            sample_rate=sr, n_fft=n_fft, hop_length=hop,
            n_mels=n_mels, f_min=20, f_max=8000
        )
        self.to_db = T.AmplitudeToDB(top_db=80)
    def forward(self, waveform):
        mel = self.mel(waveform)
        return self.to_db(mel)
    
    
# ENCODER BACKBONE 
class StyleEncoder(nn.Module):
    """CNN + Transformer encoder → 256-dim style embedding."""
    def __init__(self, n_mels=128, embed_dim=256, n_heads=8, n_layers=4):
        super().__init__()
        # CNN Frontend: extract local spectral patterns
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3,3), stride=(2,2), padding=1),
            nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=(3,3), stride=(2,2), padding=1),
            nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=(3,3), stride=(2,2), padding=1),
            nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, 256, kernel_size=(3,3), stride=(2,2), padding=1),
            nn.BatchNorm2d(256), nn.GELU(),
        )
        # Compute sequence length after CNN: input (1, 128, T) → CNN → (256, 8, T')
        self.cnn_out_channels = 256
        # Positional encoding for Transformer
        self.pos_embed = nn.Parameter(torch.randn(1, 512, self.cnn_out_channels))
        # Transformer Encoder: capture long-range temporal dependencies
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.cnn_out_channels, nhead=n_heads,
            dim_feedforward=1024, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        # Projection head (for contrastive training)
        self.projection = nn.Sequential(
            nn.Linear(self.cnn_out_channels, 512),
            nn.GELU(),
            nn.Linear(512, embed_dim)
        )
    def forward(self, mel):
        # mel: [B, 1, n_mels, T]
        x = self.cnn(mel)              # [B, 256, H', T']
        B, C, H, T = x.shape
        x = x.mean(dim=2)              # [B, 256, T']  – average over freq dim
        x = x.permute(0, 2, 1)        # [B, T', 256]  – seq-first for Transformer
        # Add positional embedding (truncate/pad to actual seq length)
        seq_len = x.shape[1]
        if seq_len > self.pos_embed.shape[1]:
            raise RuntimeError(
                f"Sequence length {seq_len} exceeds "
                f"maximum positional embedding size "
                f"{self.pos_embed.shape[1]}"
            )
        x = x + self.pos_embed[:, :seq_len, :]
        x = self.transformer(x)        # [B, T', 256]
        x = x.mean(dim=1)              # [B, 256]  – temporal average pooling
        z = self.projection(x)         # [B, embed_dim]
        return F.normalize(z, dim=-1)  # L2 normalise → unit hypersphere embedding
    
# NT-Xent CONTRASTIVE LOSS 
class NTXentLoss(nn.Module):
    """SimCLR-style contrastive loss."""
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.T = temperature
    def forward(self, z_i, z_j):
        # z_i, z_j: [B, D] – two augmented views of same samples
        B = z_i.shape[0]
        z = torch.cat([z_i, z_j], dim=0)           # [2B, D]
        sim = torch.mm(z, z.T) / self.T             # [2B, 2B] cosine similarity
        # Positive pairs: (i, i+B) and (i+B, i)
        labels = torch.arange(B, device=z.device)
        labels = torch.cat([labels + B, labels])    # [2B]
        # Mask out self-similarity diagonal
        mask = torch.eye(2*B, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask, float('-inf'))
        loss = F.cross_entropy(sim, labels)
        return loss

# DATASET 
class ArtistAudioDataset(Dataset):
    """
    Loads 5-second audio clips from artist directories.
    data/processed/
      the_beatles/  clip_001.wav  clip_002.wav ...
      nirvana/      clip_001.wav  ...
    """
    CLIP_DURATION = 5.0   # seconds
    SR            = 22050
    def __init__(self, data_dir: str, augment: bool = True):
        self.data_dir = Path(data_dir)
        self.augment  = augment
        self.mel_ext  = MelSpectrogramExtractor(sr=self.SR)
        self.clips    = []
        self.labels   = []
        self.artists  = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        self.n_classes = len(self.artists)
        for idx, artist_dir in enumerate(self.artists):
            wav_files = list(artist_dir.glob('*.wav'))
            for f in wav_files:
                self.clips.append(f)
                self.labels.append(idx)
        print(f"Dataset: {len(self.clips)} clips | {self.n_classes} artists")
        if len(self.clips) == 0:
            raise RuntimeError(
            f"No wav files found in {self.data_dir}. "
            "Expected structure:\n"
            "style_embedder/\n"
            "  artist_name/\n"
            "    clip_001.wav"
        )
            
    def __len__(self): return len(self.clips)
    def load_clip(self, path: Path) -> torch.Tensor:
        """Load random 5-sec window from a file."""
        waveform, sr = torchaudio.load(str(path))
        if sr != self.SR:
            waveform = torchaudio.transforms.Resample(sr, self.SR)(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        target_len = int(self.CLIP_DURATION * self.SR)
        if waveform.shape[1] > target_len:
            start = torch.randint(0, waveform.shape[1] - target_len, (1,)).item()
            waveform = waveform[:, start:start + target_len]
        else:
            waveform = F.pad(waveform, (0, target_len - waveform.shape[1]))
        return waveform  # [1, target_len]
    def augment_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        """Random augmentations to improve contrastive learning."""
        # Pitch shift (slight)
        if torch.rand(1) > 0.5:
            shift = torch.randint(-2, 3, (1,)).item()
            waveform = torchaudio.transforms.PitchShift(
                self.SR, n_steps=shift)(waveform)
        # Gaussian noise
        if torch.rand(1) > 0.5:
            waveform = waveform + 0.005 * torch.randn_like(waveform)
        # Time masking (random silence patches)
        if torch.rand(1) > 0.5:
            mask_len = int(0.1 * waveform.shape[1])
            start    = torch.randint(0, waveform.shape[1] - mask_len, (1,)).item()
            waveform[:, start:start + mask_len] = 0
        return waveform
    
    def __getitem__(self, idx):
        wav1 = self.load_clip(self.clips[idx])
        wav2 = self.load_clip(self.clips[idx])
        if self.augment:
            wav1 = self.augment_waveform(wav1)
            wav2 = self.augment_waveform(wav2)

        mel1 = self.mel_ext(wav1)
        mel2 = self.mel_ext(wav2)

        return mel1, mel2, self.labels[idx]
    
# TRAINING LOOP 
def train_style_embedder(
    data_dir: str = str(DEFAULT_DATA_DIR),
    save_dir: str = str(DEFAULT_MODEL_DIR),
    epochs      : int   = 80,
    batch_size  : int   = 64,
    lr          : float = 3e-4,
    temperature : float = 0.07
):
    device  = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = ArtistAudioDataset(data_dir, augment=True)
    loader  = DataLoader(dataset, batch_size=batch_size,
                         shuffle=True, num_workers=2, pin_memory=True)
    model   = StyleEncoder(embed_dim=256).to(device)
    loss_fn = NTXentLoss(temperature=temperature)
    opt     = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    os.makedirs(save_dir, exist_ok=True)
    best_loss = float('inf')
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for mel1, mel2, _ in tqdm(loader, desc=f'Epoch {epoch}/{epochs}'):
            mel1, mel2 = mel1.to(device), mel2.to(device)
            z1 = model(mel1)
            z2 = model(mel2)
            loss = loss_fn(z1, z2)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        sched.step()
        print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | LR: {sched.get_last_lr()[0]:.6f}")
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0 or avg_loss < best_loss:
            if avg_loss < best_loss:
                best_loss = avg_loss
                ckpt_name = 'best_model.pt'
            else:
                ckpt_name = f'epoch_{epoch:03d}.pt'
            torch.save({
                'epoch'     : epoch,
                'model_state': model.state_dict(),
                'loss'      : avg_loss,
                'optimizer' : opt.state_dict(),
            }, os.path.join(save_dir, ckpt_name))
    return model

# COMPUTE ARTIST EMBEDDING 
def compute_artist_embedding(
    model: StyleEncoder,
    artist_audio_dir: str,
    n_samples: int = 200,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Average embedding across n_samples random clips from an artist's directory.
    Returns a single 256-dim style vector for the artist.
    """
    from pathlib import Path
    import random
    model.eval()
    mel_ext = MelSpectrogramExtractor()
    wav_files = list(Path(artist_audio_dir).glob('*.wav'))
    random.shuffle(wav_files)
    wav_files = wav_files[:n_samples]
    embeddings = []
    SR = 22050
    CLIP_LEN = int(5.0 * SR)
    with torch.no_grad():
        for f in tqdm(wav_files, desc='Computing artist embedding'):
            try:
                waveform, sr = torchaudio.load(str(f))
                if sr != SR:
                    waveform = torchaudio.transforms.Resample(sr, SR)(waveform)
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                if waveform.shape[1] < CLIP_LEN:
                    continue
                # Take 3 random windows per file for better coverage
                for _ in range(3):
                    start = torch.randint(0, waveform.shape[1] - CLIP_LEN, (1,)).item()
                    clip  = waveform[:, start:start + CLIP_LEN]
                    mel   = mel_ext(clip).unsqueeze(0).to(device)
                    z     = model(mel)
                    embeddings.append(z.cpu())
            except Exception as e:
                print(f"  Skipping {f.name}: {e}")
    all_embeddings = torch.cat(embeddings, dim=0)   # [N, 256]
    artist_embedding = all_embeddings.mean(dim=0)   # [256]
    artist_embedding = F.normalize(artist_embedding.unsqueeze(0), dim=-1).squeeze(0)
    print(f"Artist embedding computed from {len(embeddings)} clips. Shape: {artist_embedding.shape}")
    return artist_embedding


if __name__ == "__main__":
    train_style_embedder(
        data_dir=str(DEFAULT_DATA_DIR),
        save_dir=str(DEFAULT_MODEL_DIR),
        epochs=80,
        batch_size=64,
        lr=3e-4,
    )