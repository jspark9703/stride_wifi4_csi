"""
DWT-seq RNN Classifier
======================
DWT 노이즈 제거 시퀀스 (가변 길이)를 RNN으로 분류합니다.

시퀀스 → zero-pad to max_len → pack_padded_sequence → RNN → last hidden → MLP

실행 예시
---------
  python dwt_seq_rnn.py --feat_dir ../../data/feature_extraction/dwt-seq/260406
"""

import os
import sys
import glob
import argparse
import pickle
import json
from datetime import datetime
from typing import Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import sys
import os
try:
    from loss import FocalLoss
except ImportError:
    _base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    sys.path.append(os.path.join(_base, "src", "learning"))
    from loss import FocalLoss

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ─── 경로 ────────────────────────────────────────────────────
HERE      = os.path.dirname(os.path.abspath(__file__))
BASE_DIR  = os.path.abspath(os.path.join(HERE, "..", ".."))

DEFAULT_FEAT_DIR   = os.path.join(BASE_DIR, "data", "feature_extraction", "dwt-seq", "260406")
DEFAULT_MODEL_DIR  = os.path.join(BASE_DIR, "results", "baseline", "learning", "models")
DEFAULT_RESULT_DIR = os.path.join(BASE_DIR, "results", "baseline", "learning")

# ─── 스타일 ──────────────────────────────────────────────────
BG        = "#0D1117"
PANEL     = "#161B22"
TEXT      = "#E6EDF3"
GRID      = "#30363D"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT,
    "xtick.color": TEXT, "ytick.color": TEXT,
    "text.color": TEXT, "grid.color": GRID,
    "font.family": "DejaVu Sans", "font.size": 10,
})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ══════════════════════════════════════════════════════════════
# A. 데이터 로드
# ══════════════════════════════════════════════════════════════
def _parse_label(filename):
    stem = os.path.splitext(os.path.basename(filename))[0].lower()
    parts = stem.split("_")
    lraw = parts[-1]
    if "big"   in lraw: return "big"
    if "small" in lraw or "smal" in lraw: return "small"
    return None


def load_from_feature_npz(
    feat_dir: str,
    max_seq_len: int = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray]:
    """
    DWT-seq feature NPZ 로드 후 패딩.

    Returns
    -------
    X        : (N, max_seq_len, n_pca) float32 (zero-padded)
    y        : (N,) str
    subjects : (N,) str
    max_seq_len : int
    seq_lens : (N,) int64 (실제 시퀀스 길이)
    """
    files = sorted(glob.glob(os.path.join(feat_dir, "*.npz")))
    raw_seqs, y, subjects, seq_lens = [], [], [], []

    for fp in files:
        d = np.load(fp, allow_pickle=True)
        label = str(d["label"])
        if label not in ("big", "small"):
            continue

        seq = d["features"].astype(np.float32)  # (T_i, K)
        raw_seqs.append(seq)
        y.append(label)
        subjects.append(str(d["subject"]))
        seq_lens.append(seq.shape[0])

    if len(raw_seqs) == 0:
        return np.array([]), np.array([]), np.array([]), 0, np.array([])

    if max_seq_len is None:
        max_seq_len = max(seq_lens)

    K = raw_seqs[0].shape[1]
    N = len(raw_seqs)
    X = np.zeros((N, max_seq_len, K), dtype=np.float32)

    for i, (seq, T_i) in enumerate(zip(raw_seqs, seq_lens)):
        T_i = min(T_i, max_seq_len)
        X[i, :T_i, :] = seq[:T_i, :]

    seq_lens = np.array(seq_lens, dtype=np.int64)
    return X, np.array(y), np.array(subjects), max_seq_len, seq_lens


# ══════════════════════════════════════════════════════════════
# B. RNN 모델 (pack_padded_sequence 지원)
# ══════════════════════════════════════════════════════════════
class DWTSeqRNNClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 2,
    ):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            nonlinearity="tanh",
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor, seq_lens: torch.Tensor = None) -> torch.Tensor:
        # x: (batch, max_seq_len, input_size)
        if seq_lens is not None:
            # Pack padded sequence 사용
            sorted_lens, sort_idx = seq_lens.sort(descending=True)
            x_sorted = x[sort_idx]

            packed = nn.utils.rnn.pack_padded_sequence(
                x_sorted, sorted_lens.cpu(), batch_first=True, enforce_sorted=True
            )
            out_packed, h_n = self.rnn(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)

            # Last valid timestep per sample
            batch_size = out.size(0)
            last_hidden = out[torch.arange(batch_size), sorted_lens - 1, :]

            # Unsort
            _, unsort_idx = sort_idx.sort()
            last_hidden = last_hidden[unsort_idx]
        else:
            out, h_n = self.rnn(x)
            last_hidden = out[:, -1, :]

        return self.classifier(last_hidden)


# ══════════════════════════════════════════════════════════════
# C. 학습 & 평가
# ══════════════════════════════════════════════════════════════
def train_and_evaluate(X, y, subjects, args):
    """DWT-seq RNN 모델 학습 및 평가"""
    os.makedirs(args.model_dir,  exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    classes = le.classes_
    print(f"\n[Train] Classes: {classes}  |  Total samples: {len(X)}")
    for cls in classes:
        print(f"  {cls}: {(y == cls).sum()} samples")

    # ── Hold-out 80/20
    max_seq_len = X.shape[1]
    seq_lens = getattr(args, "seq_lens", np.full(len(X), max_seq_len, dtype=np.int64))

    X_tr, X_te, y_tr, y_te, seq_lens_tr, seq_lens_te = train_test_split(
        X, y_enc, seq_lens, test_size=0.2, random_state=42, stratify=y_enc
    )
    print(f"\n[Train] Hold-out split → Train={len(X_tr)}, Test={len(X_te)}")

    # ── StandardScaler for 3D data
    N_tr, T, F = X_tr.shape
    scaler = StandardScaler()
    scaler.fit(X_tr.reshape(N_tr, T * F))
    X_tr_scaled = scaler.transform(X_tr.reshape(N_tr, T*F)).reshape(N_tr, T, F)
    X_te_scaled = scaler.transform(X_te.reshape(X_te.shape[0], T*F)).reshape(-1, T, F)

    # ── Model setup
    input_size = X_tr_scaled.shape[2]
    hidden_size = getattr(args, "rnn_hidden", 128)
    num_layers = getattr(args, "rnn_layers", 2)
    dropout = getattr(args, "rnn_dropout", 0.3)

    model = DWTSeqRNNClassifier(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        num_classes=len(classes),
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )
    use_focal = getattr(args, "use_focal_loss", False)
    if use_focal:
        gamma = getattr(args, "focal_gamma", 2.0)
        alpha_val = getattr(args, "focal_alpha", [2.5, 1.0])
        alpha_tensor = torch.tensor(alpha_val, dtype=torch.float32).to(device) if alpha_val else None
        criterion = FocalLoss(gamma=gamma, alpha=alpha_tensor)
    else:
        criterion = nn.CrossEntropyLoss()

    # ── 학습
    print(f"\n[Train] Training DWT-seq RNN (hidden={hidden_size}, layers={num_layers}, max_seq_len={max_seq_len})")
    batch_size = 64
    epochs = args.epochs
    best_loss = float("inf")
    patience = 20
    no_improve_count = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for i in range(0, len(X_tr_scaled), batch_size):
            end_idx = min(i + batch_size, len(X_tr_scaled))
            X_batch = torch.tensor(X_tr_scaled[i:end_idx], dtype=torch.float32).to(device)
            y_batch = torch.tensor(y_tr[i:end_idx], dtype=torch.long).to(device)
            seq_lens_batch = torch.tensor(seq_lens_tr[i:end_idx], dtype=torch.long).to(device)

            optimizer.zero_grad()
            logits = model(X_batch, seq_lens_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * (end_idx - i)

        train_loss /= len(X_tr_scaled)

        model.eval()
        with torch.no_grad():
            val_logits = model(
                torch.tensor(X_tr_scaled, dtype=torch.float32).to(device),
                torch.tensor(seq_lens_tr, dtype=torch.long).to(device)
            )
            val_loss = criterion(val_logits, torch.tensor(y_tr, dtype=torch.long).to(device)).item()

        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            no_improve_count = 0
        else:
            no_improve_count += 1

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if no_improve_count >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    # ── Test evaluation
    model.eval()
    with torch.no_grad():
        te_logits = model(
            torch.tensor(X_te_scaled, dtype=torch.float32).to(device),
            torch.tensor(seq_lens_te, dtype=torch.long).to(device)
        )
        y_pred_enc = te_logits.argmax(dim=1).cpu().numpy()

    y_pred = le.inverse_transform(y_pred_enc)
    y_te_labels = le.inverse_transform(y_te)
    acc = accuracy_score(y_te_labels, y_pred)
    report = classification_report(y_te_labels, y_pred, target_names=classes)
    cm = confusion_matrix(y_te_labels, y_pred)

    print(f"\n[Result] Test Accuracy: {acc:.4f}")
    print(report)

    # ── 결과 저장
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    overwrite = getattr(args, "overwrite", True)
    if overwrite:
        report_fname = "dwt_seq_rnn_report.txt"
        confusion_tag = ""
    else:
        report_fname  = f"dwt_seq_rnn_report_{ts}.txt"
        confusion_tag = ts

    report_path = os.path.join(args.result_dir, report_fname)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"DWT-seq RNN Classification Report\n")
        f.write(f"Timestamp : {ts}\n")
        f.write(f"Feature   : DWT-seq (level={args.level}, n_pca={args.n_pca})\n")
        f.write(f"RNN       : hidden={hidden_size}, layers={num_layers}, dropout={dropout}\n")
        f.write(f"Max seq   : {max_seq_len}\n")
        f.write(f"Epochs    : {args.epochs}  LR={args.lr}\n")
        f.write(f"Train/Test: {len(X_tr)} / {len(X_te)}\n\n")
        f.write(f"Test Accuracy: {acc:.4f}\n\n")
        f.write(report)
        f.write(f"\nConfusion Matrix:\n{cm}\n")
    print(f"[Result] Report saved → {report_path}")

    _plot_confusion(cm, classes, confusion_tag, args.result_dir)

    # ── 5-Fold CV
    print("\n[CV] 5-Fold Cross Validation...")
    cv_accs = []
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y_enc)):
        X_tr_fold = X[tr_idx]
        X_val_fold = X[val_idx]
        y_tr_fold = y_enc[tr_idx]
        y_val_fold = y_enc[val_idx]
        seq_lens_tr_fold = seq_lens[tr_idx]
        seq_lens_val_fold = seq_lens[val_idx]

        N_fold, T, F = X_tr_fold.shape
        scaler_fold = StandardScaler()
        scaler_fold.fit(X_tr_fold.reshape(N_fold, T*F))
        X_tr_fold_scaled = scaler_fold.transform(X_tr_fold.reshape(N_fold, T*F)).reshape(N_fold, T, F)
        X_val_fold_scaled = scaler_fold.transform(X_val_fold.reshape(len(X_val_fold), T*F)).reshape(-1, T, F)

        model_fold = DWTSeqRNNClassifier(
            input_size=F, hidden_size=hidden_size, num_layers=num_layers,
            dropout=dropout, num_classes=len(classes),
        ).to(device)
        opt_fold = optim.Adam(model_fold.parameters(), lr=args.lr, weight_decay=1e-5)

        for epoch in range(args.epochs):
            model_fold.train()
            for i in range(0, len(X_tr_fold_scaled), batch_size):
                end_idx = min(i + batch_size, len(X_tr_fold_scaled))
                X_b = torch.tensor(X_tr_fold_scaled[i:end_idx], dtype=torch.float32).to(device)
                y_b = torch.tensor(y_tr_fold[i:end_idx], dtype=torch.long).to(device)
                seq_lens_b = torch.tensor(seq_lens_tr_fold[i:end_idx], dtype=torch.long).to(device)

                opt_fold.zero_grad()
                logits = model_fold(X_b, seq_lens_b)
                loss = criterion(logits, y_b)
                loss.backward()
                opt_fold.step()

        model_fold.eval()
        with torch.no_grad():
            val_logits = model_fold(
                torch.tensor(X_val_fold_scaled, dtype=torch.float32).to(device),
                torch.tensor(seq_lens_val_fold, dtype=torch.long).to(device)
            )
            y_val_pred = val_logits.argmax(dim=1).cpu().numpy()

        fold_acc = accuracy_score(y_val_fold, y_val_pred)
        cv_accs.append(fold_acc)
        print(f"  Fold {fold+1}: {fold_acc:.4f}")

    cv_mean = np.mean(cv_accs)
    cv_std = np.std(cv_accs)
    print(f"  CV Mean±Std: {cv_mean:.4f} ± {cv_std:.4f}")

    # ── 모델 저장
    model_path   = os.path.join(args.model_dir, "dwt_seq_rnn.pth")
    scaler_path  = os.path.join(args.model_dir, "dwt_seq_rnn_scaler.pkl")
    le_path      = os.path.join(args.model_dir, "dwt_seq_rnn_label_encoder.pkl")

    torch.save(model.state_dict(), model_path)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    with open(le_path, "wb") as f:
        pickle.dump(le, f)

    meta = {
        "level": args.level,
        "n_pca": args.n_pca,
        "rnn_hidden": hidden_size,
        "rnn_layers": num_layers,
        "rnn_dropout": dropout,
        "use_focal_loss": getattr(args, "use_focal_loss", False),
        "focal_gamma": getattr(args, "focal_gamma", 2.0),
        "focal_alpha": getattr(args, "focal_alpha", [2.5, 1.0]),
        "classes": list(classes),
        "input_size": int(input_size),
        "max_seq_len": int(max_seq_len),
        "test_accuracy": float(acc),
        "cv_mean": float(cv_mean),
        "cv_std": float(cv_std),
        "timestamp": ts,
    }
    meta_path = os.path.join(args.model_dir, "dwt_seq_rnn_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4, ensure_ascii=False)

    print(f"[Model] Saved → {model_path}")
    print(f"[Model] Meta  → {meta_path}")


def _plot_confusion(cm, classes, ts, result_dir):
    fig, ax = plt.subplots(figsize=(5, 4), facecolor=BG)
    ax.set_facecolor(PANEL)
    ax.spines[:].set_color(GRID)
    ax.tick_params(colors=TEXT)

    im = ax.imshow(cm, cmap="Blues", vmin=0)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes, rotation=90, va="center")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("DWT-seq RNN Confusion Matrix", fontsize=12, pad=8)

    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-10)
    for i in range(len(classes)):
        for j in range(len(classes)):
            pct = f"{cm_norm[i,j]*100:.1f}%"
            cnt = f"({cm[i,j]})"
            tc = "white" if cm_norm[i, j] > 0.5 else TEXT
            ax.text(j, i - 0.15, pct, ha="center", va="center", fontsize=12, color=tc, fontweight="bold")
            ax.text(j, i + 0.2,  cnt, ha="center", va="center", fontsize=9,  color=tc)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.ax.tick_params(colors=TEXT, labelsize=8)

    plt.tight_layout()
    fname = "dwt_seq_rnn_confusion.png" if not ts else f"dwt_seq_rnn_confusion_{ts}.png"
    out = os.path.join(result_dir, fname)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"[Result] Confusion matrix → {out}")


# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════
def _parse_args():
    p = argparse.ArgumentParser(description="DWT-seq RNN Classifier for WiFi CSI stride classification")
    p.add_argument("--feat_dir",   default=DEFAULT_FEAT_DIR,   help="Pre-extracted dwt-seq feature NPZ dir")
    p.add_argument("--level",      type=int, default=10, help="DWT level (default: 10)")
    p.add_argument("--n_pca",      type=int, default=6,  help="# PCA components (default: 6)")
    p.add_argument("--epochs",     type=int, default=300, help="Max epochs (default: 300)")
    p.add_argument("--lr",         type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    p.add_argument("--rnn_hidden", type=int, default=128, help="RNN hidden size (default: 128)")
    p.add_argument("--rnn_layers", type=int, default=2,   help="RNN num layers (default: 2)")
    p.add_argument("--rnn_dropout", type=float, default=0.3, help="RNN dropout (default: 0.3)")
    p.add_argument("--model_dir",  default=DEFAULT_MODEL_DIR,  help="Model save directory")
    p.add_argument("--result_dir", default=DEFAULT_RESULT_DIR, help="Result save directory")
    return p.parse_args()


def main():
    args = _parse_args()
    X, y, subjects, max_seq_len, seq_lens = load_from_feature_npz(args.feat_dir)
    if len(X) == 0:
        print("[ERROR] No valid samples loaded.")
        sys.exit(1)

    args.max_seq_len = max_seq_len
    args.seq_lens = seq_lens

    print(f"[Data] Loaded: X={X.shape}, seq_len_range=[{seq_lens.min()}, {seq_lens.max()}], unique labels={set(y)}")
    train_and_evaluate(X, y, subjects, args)


if __name__ == "__main__":
    main()
