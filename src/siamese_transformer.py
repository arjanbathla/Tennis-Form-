"""
Siamese Transformer Network for Tennis Stroke Comparison
========================================================
Custom encoder layer guarantees attention weight extraction
on PyTorch 2.8+ where the default fast path skips hooks.
"""

import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.model_selection import train_test_split
import math
import warnings
warnings.filterwarnings('ignore')

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Using Apple M4 GPU (MPS)")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Using NVIDIA GPU (CUDA)")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU")

PROJECT_ROOT = Path(__file__).parent.parent
PREPROCESSED_DIR = PROJECT_ROOT / "data" / "preprocessed"
RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_DIR = PROJECT_ROOT / "models"

JOINT_NAMES = [
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

MAX_SEQ_LENGTH = 200
INPUT_DIM = 24
EMBED_DIM = 128
NUM_HEADS = 8
NUM_LAYERS = 3
DROPOUT = 0.1
EMBEDDING_SIZE = 64
MARGIN = 1.0
LEARNING_RATE = 0.001
BATCH_SIZE = 16
NUM_EPOCHS = 100
PAIRS_PER_EPOCH = 2000


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class CustomEncoderLayer(nn.Module):
    # need_weights=True bypasses PyTorch's fast path which skips attention weight computation
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.last_attn_weights = None

    def forward(self, src, src_key_padding_mask=None):
        attn_output, attn_weights = self.self_attn(
            src, src, src,
            key_padding_mask=src_key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        self.last_attn_weights = attn_weights.detach().cpu()

        src = src + self.dropout1(attn_output)
        src = self.norm1(src)
        ff_output = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(ff_output)
        src = self.norm2(src)
        return src


class StrokeTransformerEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_projection = nn.Linear(INPUT_DIM, EMBED_DIM)
        self.pos_encoder = PositionalEncoding(EMBED_DIM, max_len=MAX_SEQ_LENGTH + 10)

        self.layers = nn.ModuleList([
            CustomEncoderLayer(EMBED_DIM, NUM_HEADS, EMBED_DIM * 4, DROPOUT)
            for _ in range(NUM_LAYERS)
        ])

        self.output_projection = nn.Sequential(
            nn.Linear(EMBED_DIM, EMBEDDING_SIZE), nn.ReLU(),
            nn.Dropout(DROPOUT), nn.Linear(EMBEDDING_SIZE, EMBEDDING_SIZE),
        )

    def forward(self, x, mask=None):
        x = self.input_projection(x)
        x = self.pos_encoder(x)

        for layer in self.layers:
            x = layer(x, src_key_padding_mask=mask)

        if mask is not None:
            m = (~mask).unsqueeze(-1).float()
            x = (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)

        return self.output_projection(x)

    def get_attention_weights(self):
        return self.layers[-1].last_attn_weights


class SiameseTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = StrokeTransformerEncoder()

    def forward(self, x1, x2, mask1=None, mask2=None):
        e1, e2 = self.encoder(x1, mask1), self.encoder(x2, mask2)
        return torch.sqrt(torch.sum((e1 - e2) ** 2, dim=1) + 1e-8), e1, e2

    def get_embedding(self, x, mask=None):
        return self.encoder(x, mask)


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=MARGIN):
        super().__init__()
        self.margin = margin

    def forward(self, distance, label):
        similar = label * distance ** 2
        dissimilar = (1 - label) * torch.clamp(self.margin - distance, min=0.0) ** 2
        return torch.mean(similar + dissimilar)


class StrokePairDataset(Dataset):
    def __init__(self, expert_sequences, beginner_sequences, pairs_per_epoch=PAIRS_PER_EPOCH):
        self.expert_sequences = expert_sequences
        self.beginner_sequences = beginner_sequences
        self.pairs_per_epoch = pairs_per_epoch
        self._generate_pairs()

    def _generate_pairs(self):
        self.pairs = []
        n_e, n_b = len(self.expert_sequences), len(self.beginner_sequences)
        half = self.pairs_per_epoch // 2
        for _ in range(half):
            if np.random.random() > 0.5:
                i, j = np.random.choice(n_e, 2, replace=False)
                self.pairs.append((self.expert_sequences[i], self.expert_sequences[j], 1.0))
            else:
                i, j = np.random.choice(n_b, 2, replace=False)
                self.pairs.append((self.beginner_sequences[i], self.beginner_sequences[j], 1.0))
        for _ in range(half):
            self.pairs.append((self.expert_sequences[np.random.randint(n_e)],
                             self.beginner_sequences[np.random.randint(n_b)], 0.0))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        seq_a, seq_b, label = self.pairs[idx]
        a_pad, a_mask = self._pad(seq_a)
        b_pad, b_mask = self._pad(seq_b)
        return (torch.FloatTensor(a_pad), torch.FloatTensor(b_pad),
                torch.BoolTensor(a_mask), torch.BoolTensor(b_mask), torch.FloatTensor([label]))

    def _pad(self, seq):
        flat = seq.reshape(len(seq), -1)
        sl = len(flat)
        if sl >= MAX_SEQ_LENGTH:
            return flat[:MAX_SEQ_LENGTH], [False] * MAX_SEQ_LENGTH
        pad = np.zeros((MAX_SEQ_LENGTH - sl, flat.shape[1]))
        return np.vstack([flat, pad]), [False] * sl + [True] * (MAX_SEQ_LENGTH - sl)


def load_sequences(folder_path):
    sequences, names = [], []
    folder = Path(folder_path)
    if not folder.exists(): return sequences, names
    for f in sorted(folder.iterdir()):
        if f.name.endswith("_preprocessed.npy"):
            sequences.append(np.load(str(f)))
            names.append(f.stem)
    return sequences, names


def pad_single_sequence(seq):
    flat = seq.reshape(len(seq), -1)
    sl = len(flat)
    if sl >= MAX_SEQ_LENGTH:
        padded, mask = flat[:MAX_SEQ_LENGTH], [False] * MAX_SEQ_LENGTH
    else:
        pad = np.zeros((MAX_SEQ_LENGTH - sl, flat.shape[1]))
        padded, mask = np.vstack([flat, pad]), [False] * sl + [True] * (MAX_SEQ_LENGTH - sl)
    return (torch.FloatTensor(padded).unsqueeze(0).to(DEVICE),
            torch.BoolTensor(mask).unsqueeze(0).to(DEVICE))


def extract_attention_weights(model, sequence):
    model.eval()
    padded, mask = pad_single_sequence(sequence)
    with torch.no_grad():
        _ = model.get_embedding(padded, mask)

    attn = model.encoder.get_attention_weights()
    if attn is None:
        return None
    attn_np = attn[0].numpy()
    seq_len = int((~mask[0].cpu()).sum())
    frame_importance = attn_np[:seq_len, :seq_len].sum(axis=0)
    return frame_importance / frame_importance.max()


def train_model(expert_seqs, beginner_seqs):
    print(f"\n{'=' * 50}")
    print("TRAINING SIAMESE TRANSFORMER")
    print(f"{'=' * 50}")
    print(f"  Device: {DEVICE}")
    print(f"  Architecture: {NUM_LAYERS} layers, {NUM_HEADS} heads, {EMBED_DIM} dim")
    print(f"  Epochs: {NUM_EPOCHS}, Pairs/epoch: {PAIRS_PER_EPOCH}")

    expert_train, expert_val = train_test_split(expert_seqs, test_size=0.2, random_state=42)
    beginner_train, beginner_val = train_test_split(beginner_seqs, test_size=0.2, random_state=42)
    print(f"  Train: {len(expert_train)} expert, {len(beginner_train)} beginner")
    print(f"  Val: {len(expert_val)} expert, {len(beginner_val)} beginner")

    train_ds = StrokePairDataset(expert_train, beginner_train, pairs_per_epoch=PAIRS_PER_EPOCH)
    val_ds = StrokePairDataset(expert_val, beginner_val, pairs_per_epoch=400)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = SiameseTransformer().to(DEVICE)
    criterion = ContrastiveLoss(margin=MARGIN).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"\n  {'Epoch':<8} {'Train Loss':<14} {'Val Loss':<14} {'Val Acc':<10}")
    print(f"  {'-' * 44}")

    best_val_loss = float('inf')
    history = []

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_losses = []
        for sa, sb, ma, mb, label in train_loader:
            sa, sb = sa.to(DEVICE), sb.to(DEVICE)
            ma, mb = ma.to(DEVICE), mb.to(DEVICE)
            label = label.to(DEVICE)
            optimizer.zero_grad()
            dist, _, _ = model(sa, sb, ma, mb)
            loss = criterion(dist, label.squeeze())
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses, val_correct, val_total = [], 0, 0
        with torch.no_grad():
            for sa, sb, ma, mb, label in val_loader:
                sa, sb = sa.to(DEVICE), sb.to(DEVICE)
                ma, mb = ma.to(DEVICE), mb.to(DEVICE)
                label = label.to(DEVICE)
                dist, _, _ = model(sa, sb, ma, mb)
                loss = criterion(dist, label.squeeze())
                val_losses.append(loss.item())
                val_correct += ((dist < MARGIN/2).float() == label.squeeze()).sum().item()
                val_total += len(label)

        scheduler.step()
        at, av = np.mean(train_losses), np.mean(val_losses)
        va = val_correct / val_total if val_total > 0 else 0
        history.append({"epoch": epoch+1, "train_loss": round(float(at), 4),
                       "val_loss": round(float(av), 4), "val_accuracy": round(float(va), 4)})

        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f"  {epoch+1:<8} {at:<14.4f} {av:<14.4f} {va:<10.3f}")

        if av < best_val_loss:
            best_val_loss = av
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), str(MODEL_DIR / "siamese_transformer_best.pth"))

        train_ds._generate_pairs()
        val_ds._generate_pairs()

    print(f"\n  Best validation loss: {best_val_loss:.4f}")
    model.load_state_dict(torch.load(str(MODEL_DIR / "siamese_transformer_best.pth"), map_location=DEVICE, weights_only=True))

    test_attn = extract_attention_weights(model, expert_seqs[0])
    if test_attn is not None:
        print(f"  Attention extraction: WORKING ({len(test_attn)} frames)")
    else:
        print(f"  Attention extraction: NOT WORKING")

    return model, history


def evaluate_personal(model, expert_seqs, personal_seqs, personal_names, personal_sources):
    print(f"\n{'=' * 50}")
    print("PERSONAL STROKE EVALUATION")
    print(f"{'=' * 50}")
    model.eval()
    results = []
    ref_idx = np.random.choice(len(expert_seqs), min(10, len(expert_seqs)), replace=False)

    for i, (seq, name, source) in enumerate(zip(personal_seqs, personal_names, personal_sources)):
        distances = []
        for ri in ref_idx:
            p, pm = pad_single_sequence(seq)
            e, em = pad_single_sequence(expert_seqs[ri])
            with torch.no_grad():
                d, _, _ = model(p, e, pm, em)
                distances.append(d.item())

        avg_d = np.mean(distances)
        sim = round(100 * np.exp(-avg_d), 1)
        attn = extract_attention_weights(model, seq)

        result = {"video": name, "folder": source,
                  "siamese_distance_avg": round(float(avg_d), 4),
                  "siamese_distance_min": round(float(np.min(distances)), 4),
                  "siamese_similarity": sim, "num_frames": len(seq)}

        if attn is not None:
            top = np.argsort(attn)[-5:][::-1]
            result["top_attention_frames"] = top.tolist()
            result["attention_weights"] = attn.tolist()

        results.append(result)
        if (i+1) % 10 == 0 or i == 0:
            status = "with attention" if attn is not None else "NO attention"
            print(f"  [{i+1}/{len(personal_seqs)}] {name} — Sim: {sim}% ({status})")

    return results


def main():
    print("=" * 60)
    print("SIAMESE TRANSFORMER NETWORK")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(42); torch.manual_seed(42)

    print("\nLoading data...")
    expert_seqs, expert_names = [], []
    for s in ["forehand", "backhand"]:
        sq, nm = load_sequences(PREPROCESSED_DIR / "thetis" / s)
        expert_seqs.extend(sq); expert_names.extend(nm)

    beginner_seqs = []
    for s in ["forehand", "backhand"]:
        sq, _ = load_sequences(PREPROCESSED_DIR / "thetis_beginners" / s)
        beginner_seqs.extend(sq)

    personal_seqs, personal_names, personal_sources = [], [], []
    for fn in ["forehand_tennis_with_ball", "forehand_tennis_without_ball",
               "backhand_tennis_with_ball", "backhand_tennis_without_ball"]:
        sq, nm = load_sequences(PREPROCESSED_DIR / fn)
        personal_seqs.extend(sq); personal_names.extend(nm)
        personal_sources.extend([fn] * len(sq))

    print(f"  Expert: {len(expert_seqs)}, Beginner: {len(beginner_seqs)}, Personal: {len(personal_seqs)}")
    if not expert_seqs or not beginner_seqs:
        print("ERROR: Need both expert and beginner data."); return

    model, history = train_model(expert_seqs, beginner_seqs)
    results = evaluate_personal(model, expert_seqs, personal_seqs, personal_names, personal_sources)

    print(f"\n{'=' * 50}\nRESULTS SUMMARY\n{'=' * 50}")
    for folder in sorted(set(r["folder"] for r in results)):
        fr = [r for r in results if r["folder"] == folder]
        ac = sum(1 for r in fr if "attention_weights" in r)
        print(f"  {folder}: {np.mean([r['siamese_similarity'] for r in fr]):.1f}% avg, {ac}/{len(fr)} with attention")

    with open(str(RESULTS_DIR / "siamese_transformer_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(str(RESULTS_DIR / "siamese_training_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    with open(str(RESULTS_DIR / "siamese_model_info.json"), "w") as f:
        json.dump({"architecture": "Siamese Transformer", "device": str(DEVICE),
                   "embed_dim": EMBED_DIM, "num_heads": NUM_HEADS, "num_layers": NUM_LAYERS,
                   "embedding_size": EMBEDDING_SIZE, "max_seq_length": MAX_SEQ_LENGTH,
                   "input_dim": INPUT_DIM, "margin": MARGIN, "lr": LEARNING_RATE,
                   "batch_size": BATCH_SIZE, "epochs": NUM_EPOCHS, "pairs_per_epoch": PAIRS_PER_EPOCH,
                   "total_parameters": sum(p.numel() for p in model.parameters())}, f, indent=2)

    print(f"\n{'=' * 60}\nCOMPLETE\n{'=' * 60}")

if __name__ == "__main__":
    main()
