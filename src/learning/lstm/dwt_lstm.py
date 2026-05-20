"""
DWT-LSTM Classifier (Segment-as-Timestep)
==========================================
각 인스턴스(csi_{date}_{time}_{user})의 슬라이딩 윈도우 세그먼트를
LSTM의 시간 스텝으로 사용합니다.

입력 구조
---------
  파일명: csi_{date}_{time}_{user}_{wNN}_{label}.npz
  인스턴스 = {csi}_{date}_{time}_{user} 기준 그룹
  feature shape per segment: (feature_dim,)
  LSTM 입력: (batch, max_segs, feature_dim)  — zero padding
  lengths: 각 인스턴스의 실제 세그먼트 수 (pack_padded_sequence 용)
"""

import os, sys, glob, json, pickle
from datetime import datetime
from collections import defaultdict
from typing import Tuple, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

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
HERE     = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

DEFAULT_FEAT_DIR   = os.path.join(BASE_DIR, "data", "ablation", "baseline", "feature_extraction", "dwt")
DEFAULT_MODEL_DIR  = os.path.join(BASE_DIR, "results", "baseline", "learning", "models")
DEFAULT_RESULT_DIR = os.path.join(BASE_DIR, "results", "baseline", "learning")

BG, PANEL, TEXT, GRID = "#0D1117", "#161B22", "#E6EDF3", "#30363D"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT,
    "xtick.color": TEXT, "ytick.color": TEXT,
    "text.color": TEXT, "grid.color": GRID,
    "font.family": "DejaVu Sans", "font.size": 10,
})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ══════════════════════════════════════════════════════════════
# A. 데이터 로드 — 세그먼트를 timestep으로 그룹핑
# ══════════════════════════════════════════════════════════════

def _instance_key(filename: str) -> str:
    """csi_{date}_{time}_{user}_{wNN}_{label}.npz → csi_{date}_{time}_{user}"""
    stem = os.path.splitext(os.path.basename(filename))[0]
    parts = stem.split("_")
    # parts: [csi, date, time, user, wNN, label]
    return "_".join(parts[:4])


def _seg_index(filename: str) -> int:
    """wNN 부분에서 세그먼트 인덱스 추출"""
    stem = os.path.splitext(os.path.basename(filename))[0]
    parts = stem.split("_")
    w_part = parts[4] if len(parts) > 4 else "w00"
    return int(w_part.replace("w", ""))


def _parse_label(filename: str) -> str | None:
    stem = os.path.splitext(os.path.basename(filename))[0].lower()
    lraw = stem.split("_")[-1]
    if "big"   in lraw: return "big"
    if "small" in lraw or "smal" in lraw: return "small"
    return None


def load_from_feature_npz(
    feat_dir: str,
    level: int = 5,    # 하위 호환성 유지용 (실제로는 feature shape에서 자동 감지)
    n_pca: int = 6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    DWT feature NPZ를 인스턴스 단위로 그룹핑하여 LSTM 입력 생성.

    Returns
    -------
    X        : (N, max_segs, feature_dim) float32 — zero-padded
    y        : (N,) str
    subjects : (N,) str
    lengths  : (N,) int — 각 인스턴스의 실제 세그먼트 수
    """
    files = sorted(glob.glob(os.path.join(feat_dir, "*.npz")))
    if not files:
        return np.array([]), np.array([]), np.array([]), np.array([])

    # 인스턴스별 그룹핑
    groups: dict[str, list] = defaultdict(list)
    for f in files:
        key = _instance_key(f)
        groups[key].append(f)

    X_list, y_list, subj_list, len_list = [], [], [], []
    feature_dim = None

    for key, seg_files in groups.items():
        # 세그먼트 인덱스 순서로 정렬
        seg_files = sorted(seg_files, key=_seg_index)

        label = _parse_label(seg_files[0])
        if label is None:
            continue

        feats = []
        for fp in seg_files:
            d = np.load(fp, allow_pickle=True)
            feat = d["features"].astype(np.float32)
            if feature_dim is None:
                feature_dim = len(feat)
            feats.append(feat)

        subject = key.split("_")[-1]  # user 부분

        X_list.append(np.stack(feats, axis=0))   # (num_segs, feature_dim)
        y_list.append(label)
        subj_list.append(subject)
        len_list.append(len(feats))

    if not X_list:
        return np.array([]), np.array([]), np.array([]), np.array([])

    # Zero padding
    max_segs = max(len_list)
    N = len(X_list)
    X_padded = np.zeros((N, max_segs, feature_dim), dtype=np.float32)
    for i, x in enumerate(X_list):
        X_padded[i, :len(x), :] = x

    print(f"[Data] Instances: {N}  |  max_segs={max_segs}  |  feature_dim={feature_dim}")
    print(f"[Data] Segments per instance: min={min(len_list)}, max={max_segs}, mean={np.mean(len_list):.1f}")

    return (
        X_padded,
        np.array(y_list),
        np.array(subj_list),
        np.array(len_list, dtype=np.int64),
    )


# ══════════════════════════════════════════════════════════════
# B. LSTM 모델
# ══════════════════════════════════════════════════════════════

class DWTSegLSTM(nn.Module):
    """
    Segment-as-Timestep LSTM.
    입력: (batch, max_segs, feature_dim) — packed sequence 처리
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # lengths를 CPU로 보내야 pack_padded_sequence가 동작
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out_packed, (h_n, _) = self.lstm(packed)
        # 마지막 레이어의 마지막 hidden state 사용
        last_hidden = h_n[-1]  # (batch, hidden_size)
        return self.classifier(last_hidden)


# ══════════════════════════════════════════════════════════════
# C. 학습 & 평가
# ══════════════════════════════════════════════════════════════

def train_and_evaluate(X, y, subjects, args):
    """
    X: (N, max_segs, feature_dim) — padded
    args.seq_lengths: (N,) — 실제 세그먼트 수
    """
    os.makedirs(args.model_dir,  exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)

    lengths = np.array(getattr(args, "seq_lengths", np.ones(len(X), dtype=np.int64)))

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    classes = le.classes_

    print(f"\n[Train] Classes: {classes}  |  Instances: {len(X)}")
    for cls in classes:
        print(f"  {cls}: {(y == cls).sum()} instances")

    # Hold-out 80/20
    idx_all = np.arange(len(X))
    idx_tr, idx_te = train_test_split(
        idx_all, test_size=0.2, random_state=42, stratify=y_enc
    )

    X_tr, X_te = X[idx_tr], X[idx_te]
    y_tr, y_te = y_enc[idx_tr], y_enc[idx_te]
    len_tr, len_te = lengths[idx_tr], lengths[idx_te]

    # StandardScaler on flat view
    N_tr, T, F = X_tr.shape
    scaler = StandardScaler()
    scaler.fit(X_tr.reshape(N_tr, T * F))

    def _scale(Xb, Lb):
        N, T_, F_ = Xb.shape
        Xs = scaler.transform(Xb.reshape(N, T_ * F_)).reshape(N, T_, F_)
        # 패딩 영역은 0으로 복원
        for i, l in enumerate(Lb):
            Xs[i, l:, :] = 0.0
        return Xs

    X_tr_s = _scale(X_tr, len_tr)
    X_te_s = _scale(X_te, len_te)

    feature_dim = X_tr_s.shape[2]
    hidden_size = getattr(args, "lstm_hidden", 128)
    num_layers  = getattr(args, "lstm_layers", 2)
    dropout     = getattr(args, "lstm_dropout", 0.3)

    model = DWTSegLSTM(
        input_size=feature_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        num_classes=len(classes),
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

    use_focal = getattr(args, "use_focal_loss", False)
    if use_focal:
        gamma = getattr(args, "focal_gamma", 2.0)
        alpha_val = getattr(args, "focal_alpha", [2.5, 1.0])
        alpha_t = torch.tensor(alpha_val, dtype=torch.float32).to(device)
        criterion = FocalLoss(gamma=gamma, alpha=alpha_t)
    else:
        criterion = nn.CrossEntropyLoss()

    batch_size = 32
    epochs = args.epochs
    best_loss = float("inf")
    patience_cnt = 0
    PATIENCE = 20

    print(f"\n[Train] Segment-LSTM (hidden={hidden_size}, layers={num_layers}, dropout={dropout})")
    print(f"        Loss={'FocalLoss' if use_focal else 'CrossEntropy'}  Device={device}")

    for epoch in range(epochs):
        model.train()
        perm = np.random.permutation(len(X_tr_s))
        X_tr_p, y_tr_p, len_tr_p = X_tr_s[perm], y_tr[perm], len_tr[perm]
        train_loss = 0.0

        for i in range(0, len(X_tr_p), batch_size):
            xb = torch.tensor(X_tr_p[i:i+batch_size], dtype=torch.float32).to(device)
            yb = torch.tensor(y_tr_p[i:i+batch_size], dtype=torch.long).to(device)
            lb = torch.tensor(len_tr_p[i:i+batch_size], dtype=torch.long)

            optimizer.zero_grad()
            logits = model(xb, lb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(yb)

        train_loss /= len(X_tr_p)

        # Validation loss (전체 학습 세트)
        model.eval()
        with torch.no_grad():
            val_logits = model(
                torch.tensor(X_tr_s, dtype=torch.float32).to(device),
                torch.tensor(len_tr, dtype=torch.long),
            )
            val_loss = criterion(val_logits, torch.tensor(y_tr, dtype=torch.long).to(device)).item()

        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            patience_cnt = 0
        else:
            patience_cnt += 1

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{epochs}  train={train_loss:.4f}  val={val_loss:.4f}")

        if patience_cnt >= PATIENCE:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    # 평가
    model.eval()
    with torch.no_grad():
        te_logits = model(
            torch.tensor(X_te_s, dtype=torch.float32).to(device),
            torch.tensor(len_te, dtype=torch.long),
        )
        y_pred_enc = te_logits.argmax(dim=1).cpu().numpy()

    y_pred = le.inverse_transform(y_pred_enc)
    y_te_lbl = le.inverse_transform(y_te)
    acc = accuracy_score(y_te_lbl, y_pred)
    report = classification_report(y_te_lbl, y_pred, target_names=classes)
    cm = confusion_matrix(y_te_lbl, y_pred)

    print(f"\n[Result] Test Accuracy: {acc:.4f}")
    print(report)

    # 결과 저장
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    overwrite = getattr(args, "overwrite", True)
    report_fname = "dwt_lstm_report.txt" if overwrite else f"dwt_lstm_report_{ts}.txt"
    report_path  = os.path.join(args.result_dir, report_fname)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("DWT-LSTM (Segment-as-Timestep) Classification Report\n")
        f.write(f"Timestamp : {ts}\n")
        f.write(f"Feature   : DWT flat, segments as timesteps\n")
        f.write(f"Input     : (N, max_segs={X.shape[1]}, feature_dim={feature_dim})\n")
        f.write(f"LSTM      : hidden={hidden_size}, layers={num_layers}, dropout={dropout}\n")
        f.write(f"Epochs    : {args.epochs}  LR={args.lr}\n")
        if use_focal:
            f.write(f"Loss      : FocalLoss(gamma={getattr(args,'focal_gamma',2.0)}, alpha={getattr(args,'focal_alpha',[2.5,1.0])})\n")
        else:
            f.write("Loss      : CrossEntropyLoss\n")
        f.write(f"Train/Test: {len(X_tr)} / {len(X_te)} instances\n\n")
        f.write(f"Test Accuracy: {acc:.4f}\n\n")
        f.write(report)
        f.write(f"\nConfusion Matrix:\n{cm}\n")
    print(f"[Result] Report → {report_path}")

    _plot_confusion(cm, classes, "" if overwrite else ts, args.result_dir)

    # 5-Fold CV
    print("\n[CV] 5-Fold Cross Validation...")
    cv_accs = []
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y_enc)):
        X_tr_f, X_val_f = X[tr_idx], X[val_idx]
        y_tr_f, y_val_f = y_enc[tr_idx], y_enc[val_idx]
        len_tr_f, len_val_f = lengths[tr_idx], lengths[val_idx]

        N_f, T_f, F_f = X_tr_f.shape
        sc_f = StandardScaler()
        sc_f.fit(X_tr_f.reshape(N_f, T_f * F_f))

        def _scale_f(Xb, Lb, sc):
            N_, T_, F_ = Xb.shape
            Xs = sc.transform(Xb.reshape(N_, T_ * F_)).reshape(N_, T_, F_)
            for i, l in enumerate(Lb):
                Xs[i, l:, :] = 0.0
            return Xs

        X_tr_fs = _scale_f(X_tr_f, len_tr_f, sc_f)
        X_val_fs = _scale_f(X_val_f, len_val_f, sc_f)

        m_f = DWTSegLSTM(
            input_size=F_f, hidden_size=hidden_size,
            num_layers=num_layers, dropout=dropout,
            num_classes=len(classes),
        ).to(device)
        opt_f = optim.Adam(m_f.parameters(), lr=args.lr, weight_decay=1e-5)

        for epoch in range(args.epochs):
            m_f.train()
            perm = np.random.permutation(len(X_tr_fs))
            for i in range(0, len(X_tr_fs), batch_size):
                xb = torch.tensor(X_tr_fs[perm[i:i+batch_size]], dtype=torch.float32).to(device)
                yb = torch.tensor(y_tr_f[perm[i:i+batch_size]], dtype=torch.long).to(device)
                lb = torch.tensor(len_tr_f[perm[i:i+batch_size]], dtype=torch.long)
                opt_f.zero_grad()
                loss = criterion(m_f(xb, lb), yb)
                loss.backward()
                opt_f.step()

        m_f.eval()
        with torch.no_grad():
            vl = m_f(
                torch.tensor(X_val_fs, dtype=torch.float32).to(device),
                torch.tensor(len_val_f, dtype=torch.long),
            )
            vp = vl.argmax(dim=1).cpu().numpy()
        fold_acc = accuracy_score(y_val_f, vp)
        cv_accs.append(fold_acc)
        print(f"  Fold {fold+1}: {fold_acc:.4f}")

    cv_mean = np.mean(cv_accs)
    cv_std  = np.std(cv_accs)
    print(f"  CV Mean±Std: {cv_mean:.4f} ± {cv_std:.4f}")

    with open(report_path, "a", encoding="utf-8") as f:
        f.write(f"\n5-Fold CV: {[round(a,4) for a in cv_accs]}\n")
        f.write(f"CV Mean: {cv_mean:.4f}  CV Std: {cv_std:.4f}\n")

    # 모델 저장
    model_path  = os.path.join(args.model_dir, "dwt_lstm.pth")
    scaler_path = os.path.join(args.model_dir, "dwt_lstm_scaler.pkl")
    le_path     = os.path.join(args.model_dir, "dwt_lstm_label_encoder.pkl")
    meta_path   = os.path.join(args.model_dir, "dwt_lstm_meta.json")

    torch.save(model.state_dict(), model_path)
    with open(scaler_path, "wb") as f: pickle.dump(scaler, f)
    with open(le_path,     "wb") as f: pickle.dump(le, f)

    meta = {
        "input_mode": "segment_timestep",
        "feature_dim": int(feature_dim),
        "max_segs": int(X.shape[1]),
        "lstm_hidden": hidden_size,
        "lstm_layers": num_layers,
        "lstm_dropout": dropout,
        "use_focal_loss": use_focal,
        "focal_gamma": getattr(args, "focal_gamma", 2.0),
        "focal_alpha": getattr(args, "focal_alpha", [2.5, 1.0]),
        "classes": list(classes),
        "test_accuracy": float(acc),
        "cv_mean": float(cv_mean),
        "cv_std":  float(cv_std),
        "timestamp": ts,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4, ensure_ascii=False)

    print(f"[Model] Saved → {model_path}")
    print(f"[Model] Meta  → {meta_path}")


def _plot_confusion(cm, classes, ts, result_dir):
    fig, ax = plt.subplots(figsize=(5, 4), facecolor=BG)
    ax.set_facecolor(PANEL); ax.spines[:].set_color(GRID); ax.tick_params(colors=TEXT)
    im = ax.imshow(cm, cmap="Blues", vmin=0)
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, color=TEXT)
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes, color=TEXT, rotation=90, va="center")
    ax.set_xlabel("Predicted", color=TEXT); ax.set_ylabel("True", color=TEXT)
    ax.set_title("DWT-LSTM (Seg-Timestep) Confusion Matrix", color=TEXT, fontsize=11, pad=8)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-10)
    for i in range(len(classes)):
        for j in range(len(classes)):
            tc = "white" if cm_norm[i, j] > 0.5 else TEXT
            ax.text(j, i-0.15, f"{cm_norm[i,j]*100:.1f}%", ha="center", va="center", fontsize=12, color=tc, fontweight="bold")
            ax.text(j, i+0.20, f"({cm[i,j]})", ha="center", va="center", fontsize=9, color=tc)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).ax.tick_params(colors=TEXT, labelsize=8)
    plt.tight_layout()
    fname = "dwt_lstm_confusion.png" if not ts else f"dwt_lstm_confusion_{ts}.png"
    fig.savefig(os.path.join(result_dir, fname), dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"[Result] Confusion matrix → {os.path.join(result_dir, fname)}")
