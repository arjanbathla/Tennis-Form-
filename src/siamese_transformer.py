"""
Siamese Transformer Network for Tennis Stroke Comparison
========================================================
Novel contribution: combines Siamese architecture (pairwise similarity
learning) with Transformer encoder branches (self-attention for temporal
sequences). This combination has not been previously applied to tennis
stroke quality assessment.

Architecture:
  Input A (player stroke) ──► Transformer Encoder ──► Embedding A ──┐
                                                                     ├─► Distance ──► Similarity Score
  Input B (expert stroke) ──► Transformer Encoder ──► Embedding B ──┘
                               (shared weights)

Training:
  - Positive pairs: expert-expert (should produce low distance)
  - Negative pairs: expert-beginner (should produce high distance)
  - Loss: Contrastive loss

Inference:
  - Compare player stroke against expert reference
  - Produces learned similarity score
  - Attention weights show which frames matter most

Optimised for Apple M4 with MPS GPU acceleration.
"""

import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.model_selection import train_test_split
import math
import warnings
warnings.filterwarnings('ignore')

# ================================================================
# DEVICE SELECTION — Apple M4 GPU acceleration
# ================================================================
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Using Apple M4 GPU (MPS)")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Using NVIDIA GPU (CUDA)")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU")

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
PREPROCESSED_DIR = PROJECT_ROOT / "data" / "preprocessed"
RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_DIR = PROJECT_ROOT / "models"

# Joint names
JOINT_NAMES = [
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]

# Model hyperparameters — upgraded for M4
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


class StrokeTransformerEncoder(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, embed_dim=EMBED_DIM,
                 num_heads=NUM_HEADS, num_layers=NUM_LAYERS,
                 dropout=DROPOUT, embedding_size=EMBEDDING_SIZE):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=MAX_SEQ_LENGTH + 10)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4, dropout=dropout, batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_projection = nn.Sequential(
            nn.Linear(embed_dim, embedding_size), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(embedding_size, embedding_size),
        )
        self.attention_weights = None
        self._register_hooks()

    def _register_hooks(self):
        self._hooks = []
        for layer in self.transformer_encoder.layers:
            layer.self_attn.need_weights = True
            layer.self_attn.average_attn_weights = True
            hook = layer.self_attn.register_forward_hook(self._attention_hook)
            self._hooks.append(hook)

    def _attention_hook(self, module, input, output):
        if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
            self.attention_weights = output[1].detach().cpu()

    def forward(self, x, mask=None):
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        if mask is not None:
            x = self.transformer_encoder(x, src_key_padding_mask=mask)
        else:
            x = self.transformer_encoder(x)

        if mask is not None:
            mask_expanded = (~mask).unsqueeze(-1).float()
            x = x * mask_expanded
            lengths = mask_expanded.sum(dim=1)
            x = x.sum(dim=1) / lengths.clamp(min=1)
        else:
            x = x.mean(dim=1)

        return self.output_projection(x)


class SiameseTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = StrokeTransformerEncoder()

    def forward(self, x1, x2, mask1=None, mask2=None):
        e1 = self.encoder(x1, mask1)
        e2 = self.encoder(x2, mask2)
        distance = torch.sqrt(torch.sum((e1 - e2) ** 2, dim=1) + 1e-8)
        return distance, e1, e2

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
        n_expert = len(self.expert_sequences)
        n_beginner = len(self.beginner_sequences)
        half = self.pairs_per_epoch // 2

        for _ in range(half):
            if np.random.random() > 0.5:
                i, j = np.random.choice(n_expert, 2, replace=False)
                self.pairs.append((self.expert_sequences[i], self.expert_sequences[j], 1.0))
            else:
                i, j = np.random.choice(n_beginner, 2, replace=False)
                self.pairs.append((self.beginner_sequences[i], self.beginner_sequences[j], 1.0))

        for _ in range(half):
            i = np.random.randint(n_expert)
            j = np.random.randint(n_beginner)
            self.pairs.append((self.expert_sequences[i], self.beginner_sequences[j], 0.0))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        seq_a, seq_b, label = self.pairs[idx]
        seq_a_padded, mask_a = self._pad_sequence(seq_a)
        seq_b_padded, mask_b = self._pad_sequence(seq_b)
        return (
            torch.FloatTensor(seq_a_padded),
            torch.FloatTensor(seq_b_padded),
            torch.BoolTensor(mask_a),
            torch.BoolTensor(mask_b),
            torch.FloatTensor([label]),
        )

    def _pad_sequence(self, seq):
        flat = seq.reshape(len(seq), -1)
        seq_len = len(flat)
        if seq_len >= MAX_SEQ_LENGTH:
            return flat[:MAX_SEQ_LENGTH], [False] * MAX_SEQ_LENGTH
        padding = np.zeros((MAX_SEQ_LENGTH - seq_len, flat.shape[1]))
        padded = np.vstack([flat, padding])
        mask = [False] * seq_len + [True] * (MAX_SEQ_LENGTH - seq_len)
        return padded, mask


def load_sequences(folder_path):
    sequences, names = [], []
    folder = Path(folder_path)
    if not folder.exists():
        return sequences, names
    for f in sorted(folder.iterdir()):
        if f.name.endswith("_preprocessed.npy"):
            sequences.append(np.load(str(f)))
            names.append(f.stem)
    return sequences, names


def pad_single_sequence(seq):
    flat = seq.reshape(len(seq), -1)
    seq_len = len(flat)
    if seq_len >= MAX_SEQ_LENGTH:
        padded = flat[:MAX_SEQ_LENGTH]
        mask = [False] * MAX_SEQ_LENGTH
    else:
        padding = np.zeros((MAX_SEQ_LENGTH - seq_len, flat.shape[1]))
        padded = np.vstack([flat, padding])
        mask = [False] * seq_len + [True] * (MAX_SEQ_LENGTH - seq_len)
    return (
        torch.FloatTensor(padded).unsqueeze(0).to(DEVICE),
        torch.BoolTensor(mask).unsqueeze(0).to(DEVICE),
    )


def extract_attention_weights(model, sequence):
    model.eval()
    padded, mask = pad_single_sequence(sequence)
    with torch.no_grad():
        _ = model.get_embedding(padded, mask)
    attn = model.encoder.attention_weights
    if attn is not None:
        attn_avg = attn[0].mean(dim=0).numpy()
        seq_len = int((~mask[0].cpu()).sum())
        frame_importance = attn_avg[:seq_len, :seq_len].sum(axis=0)
        frame_importance = frame_importance / frame_importance.max()
        return frame_importance
    return None


def train_model(expert_seqs, beginner_seqs):
    print(f"\n{'=' * 50}")
    print("TRAINING SIAMESE TRANSFORMER")
    print(f"{'=' * 50}")
    print(f"  Device: {DEVICE}")
    print(f"  Expert sequences: {len(expert_seqs)}")
    print(f"  Beginner sequences: {len(beginner_seqs)}")
    print(f"  Architecture: {NUM_LAYERS} layers, {NUM_HEADS} heads, {EMBED_DIM} dim")
    print(f"  Embedding size: {EMBEDDING_SIZE}")
    print(f"  Epochs: {NUM_EPOCHS}, Batch size: {BATCH_SIZE}")
    print(f"  Pairs per epoch: {PAIRS_PER_EPOCH}")

    expert_train, expert_val = train_test_split(expert_seqs, test_size=0.2, random_state=42)
    beginner_train, beginner_val = train_test_split(beginner_seqs, test_size=0.2, random_state=42)

    print(f"\n  Training: {len(expert_train)} expert, {len(beginner_train)} beginner")
    print(f"  Validation: {len(expert_val)} expert, {len(beginner_val)} beginner")

    train_dataset = StrokePairDataset(expert_train, beginner_train, pairs_per_epoch=PAIRS_PER_EPOCH)
    val_dataset = StrokePairDataset(expert_val, beginner_val, pairs_per_epoch=400)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = SiameseTransformer().to(DEVICE)
    criterion = ContrastiveLoss(margin=MARGIN).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Total parameters: {total_params:,}")

    print(f"\n  {'Epoch':<8} {'Train Loss':<14} {'Val Loss':<14} {'Val Acc':<10}")
    print(f"  {'-' * 44}")

    best_val_loss = float('inf')
    training_history = []

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_losses = []
        for seq_a, seq_b, mask_a, mask_b, label in train_loader:
            seq_a, seq_b = seq_a.to(DEVICE), seq_b.to(DEVICE)
            mask_a, mask_b = mask_a.to(DEVICE), mask_b.to(DEVICE)
            label = label.to(DEVICE)

            optimizer.zero_grad()
            distance, _, _ = model(seq_a, seq_b, mask_a, mask_b)
            loss = criterion(distance, label.squeeze())
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses, val_correct, val_total = [], 0, 0
        with torch.no_grad():
            for seq_a, seq_b, mask_a, mask_b, label in val_loader:
                seq_a, seq_b = seq_a.to(DEVICE), seq_b.to(DEVICE)
                mask_a, mask_b = mask_a.to(DEVICE), mask_b.to(DEVICE)
                label = label.to(DEVICE)

                distance, _, _ = model(seq_a, seq_b, mask_a, mask_b)
                loss = criterion(distance, label.squeeze())
                val_losses.append(loss.item())
                predicted_similar = (distance < MARGIN / 2).float()
                val_correct += (predicted_similar == label.squeeze()).sum().item()
                val_total += len(label)

        scheduler.step()
        avg_train = np.mean(train_losses)
        avg_val = np.mean(val_losses)
        val_acc = val_correct / val_total if val_total > 0 else 0

        training_history.append({
            "epoch": epoch + 1,
            "train_loss": round(float(avg_train), 4),
            "val_loss": round(float(avg_val), 4),
            "val_accuracy": round(float(val_acc), 4),
        })

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  {epoch+1:<8} {avg_train:<14.4f} {avg_val:<14.4f} {val_acc:<10.3f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), str(MODEL_DIR / "siamese_transformer_best.pth"))

        train_dataset._generate_pairs()
        val_dataset._generate_pairs()

    print(f"\n  Best validation loss: {best_val_loss:.4f}")
    model.load_state_dict(
        torch.load(str(MODEL_DIR / "siamese_transformer_best.pth"), map_location=DEVICE, weights_only=True)
    )
    return model, training_history


def evaluate_personal_recordings(model, expert_seqs, expert_names, personal_seqs, personal_names, personal_sources):
    print(f"\n{'=' * 50}")
    print("SIAMESE TRANSFORMER: PERSONAL STROKE EVALUATION")
    print(f"{'=' * 50}")

    model.eval()
    results = []
    num_refs = min(10, len(expert_seqs))
    ref_indices = np.random.choice(len(expert_seqs), num_refs, replace=False)

    for i, (personal_seq, name, source) in enumerate(zip(personal_seqs, personal_names, personal_sources)):
        distances = []
        for ref_idx in ref_indices:
            p_padded, p_mask = pad_single_sequence(personal_seq)
            e_padded, e_mask = pad_single_sequence(expert_seqs[ref_idx])
            with torch.no_grad():
                dist, _, _ = model(p_padded, e_padded, p_mask, e_mask)
                distances.append(dist.item())

        avg_distance = np.mean(distances)
        min_distance = np.min(distances)
        similarity = round(100 * np.exp(-avg_distance), 1)
        attention = extract_attention_weights(model, personal_seq)

        result = {
            "video": name, "folder": source,
            "siamese_distance_avg": round(float(avg_distance), 4),
            "siamese_distance_min": round(float(min_distance), 4),
            "siamese_similarity": similarity,
            "num_frames": len(personal_seq),
        }
        if attention is not None:
            top_frames = np.argsort(attention)[-5:][::-1]
            result["top_attention_frames"] = top_frames.tolist()
            result["attention_weights"] = attention.tolist()

        results.append(result)
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(personal_seqs)}] {name}")
            print(f"    Distance: {avg_distance:.4f}  |  Similarity: {similarity}%")

    return results


def compare_with_dtw_results(siamese_results):
    dtw_path = RESULTS_DIR / "dtw_comparison_results.json"
    if not dtw_path.exists():
        print("\n  DTW results not found, skipping comparison")
        return

    with open(str(dtw_path)) as f:
        dtw_results = json.load(f)

    print(f"\n{'=' * 50}")
    print("COMPARISON: SIAMESE TRANSFORMER vs DTW")
    print(f"{'=' * 50}")

    dtw_lookup = {r["video"]: r for r in dtw_results}
    folders = set(r["folder"] for r in siamese_results)

    print(f"\n  {'Folder':<38} {'DTW Sim':>10} {'Siamese Sim':>12}")
    print(f"  {'-' * 62}")

    for folder in sorted(folders):
        siamese_folder = [r for r in siamese_results if r["folder"] == folder]
        dtw_sims = [v["similarity_score"] for v in dtw_lookup.values() if v["folder"] == folder]
        siamese_sims = [r["siamese_similarity"] for r in siamese_folder]

        if dtw_sims and siamese_sims:
            print(f"  {folder:<38} {np.mean(dtw_sims):>9.1f}% {np.mean(siamese_sims):>11.1f}%")


def main():
    print("=" * 60)
    print("SIAMESE TRANSFORMER NETWORK")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)
    torch.manual_seed(42)

    print("\nLoading preprocessed data...")
    expert_seqs, expert_names = [], []
    for stroke in ["forehand", "backhand"]:
        seqs, names = load_sequences(PREPROCESSED_DIR / "thetis" / stroke)
        expert_seqs.extend(seqs)
        expert_names.extend(names)

    beginner_seqs, beginner_names = [], []
    for stroke in ["forehand", "backhand"]:
        seqs, names = load_sequences(PREPROCESSED_DIR / "thetis_beginners" / stroke)
        beginner_seqs.extend(seqs)
        beginner_names.extend(names)

    personal_seqs, personal_names, personal_sources = [], [], []
    for folder_name in ["forehand_tennis_with_ball", "forehand_tennis_without_ball",
                         "backhand_tennis_with_ball", "backhand_tennis_without_ball"]:
        seqs, names = load_sequences(PREPROCESSED_DIR / folder_name)
        personal_seqs.extend(seqs)
        personal_names.extend(names)
        personal_sources.extend([folder_name] * len(seqs))

    print(f"  Expert: {len(expert_seqs)}, Beginner: {len(beginner_seqs)}, Personal: {len(personal_seqs)}")

    if not expert_seqs or not beginner_seqs:
        print("\nERROR: Need both expert and beginner data.")
        return

    model, history = train_model(expert_seqs, beginner_seqs)

    siamese_results = evaluate_personal_recordings(
        model, expert_seqs, expert_names, personal_seqs, personal_names, personal_sources,
    )

    print(f"\n{'=' * 50}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 50}")
    folders = set(r["folder"] for r in siamese_results)
    print(f"\n  {'Folder':<38} {'Count':>6} {'Avg Sim':>10}")
    print(f"  {'-' * 56}")
    for folder in sorted(folders):
        fr = [r for r in siamese_results if r["folder"] == folder]
        print(f"  {folder:<38} {len(fr):>6} {np.mean([r['siamese_similarity'] for r in fr]):>9.1f}%")

    for stroke in ["forehand", "backhand"]:
        sr = [r for r in siamese_results if stroke in r["folder"]]
        if sr:
            print(f"  {stroke:<38} {np.mean([r['siamese_similarity'] for r in sr]):>9.1f}%")

    compare_with_dtw_results(siamese_results)

    with open(str(RESULTS_DIR / "siamese_transformer_results.json"), "w") as f:
        json.dump(siamese_results, f, indent=2)
    with open(str(RESULTS_DIR / "siamese_training_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    with open(str(RESULTS_DIR / "siamese_model_info.json"), "w") as f:
        json.dump({
            "architecture": "Siamese Transformer", "device": str(DEVICE),
            "embed_dim": EMBED_DIM, "num_heads": NUM_HEADS, "num_layers": NUM_LAYERS,
            "embedding_size": EMBEDDING_SIZE, "max_seq_length": MAX_SEQ_LENGTH,
            "input_dim": INPUT_DIM, "margin": MARGIN, "lr": LEARNING_RATE,
            "batch_size": BATCH_SIZE, "epochs": NUM_EPOCHS,
            "pairs_per_epoch": PAIRS_PER_EPOCH,
            "total_parameters": sum(p.numel() for p in model.parameters()),
        }, f, indent=2)

    print(f"\n{'=' * 60}")
    print("SIAMESE TRANSFORMER COMPLETE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
