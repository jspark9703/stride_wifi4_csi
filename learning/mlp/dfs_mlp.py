"""
DFS-MLP Classifier
===================
DFS (전력 스펙트로그램) 특징을 MLP 로 big/small 보폭 이진 분류합니다.

특징 구성
---------
DFS 스펙트로그램 (n_freq × n_time) 2D 매트릭스를 직접 MLP 입력으로 사용하면
차원이 너무 커지므로 두 단계 압축을 수행합니다:

  1) 주파수 축 통계 → 각 주파수 빈에서 [mean, std, max, skew] 4개 통계량
     → (n_freq, 4) = (64, 4) → flatten → 256-dim 통계 벡터
     (시간 분포 정보를 보존하면서 시간 축 차원 제거)

  2) PCA 압축 (n_pca 주성분)
     → 노이즈 제거 + 분류에 효과적인 저차원 표현

  최종 특징 벡터: (~64 × 4) → PCA → n_pca 차원

MLP 구조
--------
  Input (n_pca)
    └─ Dense(256) → BatchNorm → ReLU → Dropout(0.3)
    └─ Dense(128) → ReLU → Dropout(0.2)
    └─ Dense(64)  → ReLU
    └─ Dense(2)   → Softmax

  → sklearn MLPClassifier (StandardScaler + Pipeline)

실행
----
  python dfs_mlp.py
  python dfs_mlp.py --n_pca 32 --hidden 256 128 64
  python dfs_mlp.py --inference --model_path models/dfs_mlp.pkl

출력
----
  models/dfs_mlp.pkl              학습된 모델
  models/dfs_label_encoder.pkl    레이블 인코더
  models/dfs_mlp_meta.json        메타데이터
  results/dfs_mlp_report_*.txt    분류 리포트
  results/dfs_confusion_*.png     혼동행렬 시각화
"""

import os
import sys
import glob
import argparse
import pickle
import json
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.neural_network  import MLPClassifier
from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.decomposition   import PCA
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics         import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline        import Pipeline

# ─── 경로 ────────────────────────────────────────────────────
HERE      = os.path.dirname(os.path.abspath(__file__))
BASE_DIR  = os.path.abspath(os.path.join(HERE, "..", ".."))

DEFAULT_FEAT_DIR   = os.path.join(BASE_DIR, "data", "feature_extraction", "dfs")
DEFAULT_SANIT_DIR  = os.path.join(BASE_DIR, "data", "sanitization", "sanitization")
DEFAULT_MODEL_DIR  = os.path.join(HERE, "models")
DEFAULT_RESULT_DIR = os.path.join(HERE, "results")

# ─── 스타일 ──────────────────────────────────────────────────
BG, PANEL, TEXT, GRID = "#0D1117", "#161B22", "#E6EDF3", "#30363D"
C_BIG, C_SMALL        = "#EF5350", "#42A5F5"

plt.rcParams.update({
    "figure.facecolor": BG,  "axes.facecolor": PANEL,
    "axes.edgecolor":   GRID, "axes.labelcolor": TEXT,
    "xtick.color": TEXT, "ytick.color": TEXT,
    "text.color":  TEXT, "grid.color":  GRID,
    "font.family": "DejaVu Sans", "font.size": 10,
})


# ══════════════════════════════════════════════════════════════
# A. 특징 추출 — 스펙트로그램 → 통계 벡터
# ══════════════════════════════════════════════════════════════

def _spectrogram_to_stat_vector(dfs_power: np.ndarray) -> np.ndarray:
    """
    DFS 전력 스펙트로그램 (n_freq, n_time) → 고정 차원 통계 벡터.

    각 주파수 빈(행)에서 시간 축 통계 4종 추출:
      mean, std, max, 왜도(skewness)

    → (n_freq × 4,) float32

    이유: 시간 길이가 파일마다 다를 수 있으므로 통계로 표준화.
    """
    power = dfs_power.astype(np.float64)   # (n_freq, n_time)

    mu   = power.mean(axis=1)              # (n_freq,)
    sigma = power.std(axis=1)             # (n_freq,)
    mx   = power.max(axis=1)              # (n_freq,)

    # 왜도: (평균과의 차)^3 / std^3
    diff3 = ((power - mu[:, None]) ** 3).mean(axis=1)
    skew  = diff3 / (sigma ** 3 + 1e-10)  # (n_freq,)

    feat = np.concatenate([mu, sigma, mx, skew]).astype(np.float32)
    return feat


def _parse_label(filename: str):
    stem  = os.path.splitext(os.path.basename(filename))[0].lower()
    parts = stem.split("_")
    lraw  = parts[-1]
    if "big"   in lraw: return "big"
    if "small" in lraw or "smal" in lraw: return "small"
    return None


# ── A-1. feature NPZ 로드
def load_from_feature_npz(feat_dir: str):
    """data/feature_extraction/dfs/*.npz → X, y, subjects"""
    files = sorted(glob.glob(os.path.join(feat_dir, "*.npz")))
    X, y, subjects = [], [], []
    for fp in files:
        d     = np.load(fp, allow_pickle=True)
        label = str(d["label"])
        if label not in ("big", "small"):
            continue
        try:
            power = d["dfs_power"].astype(np.float32)   # (n_freq, n_time)
            feat  = _spectrogram_to_stat_vector(power)
            X.append(feat)
            y.append(label)
            subjects.append(str(d["subject"]))
        except Exception as e:
            print(f"  [SKIP] {os.path.basename(fp)}: {e}")
    return np.array(X, dtype=np.float32), np.array(y), np.array(subjects)


# ── A-2. on-the-fly 계산
def load_from_sanit(sanit_dir: str, n_fft: int, hop: int, doppler_hz: float):
    """sanitization/*.npz → on-the-fly DFS 계산 → 통계 벡터"""
    _ext = os.path.join(BASE_DIR, "feature_extraction", "dfs")
    if _ext not in sys.path:
        sys.path.insert(0, _ext)
    from extract_dfs import extract_features_single

    files = sorted(glob.glob(os.path.join(sanit_dir, "*.npz")))
    X, y, subjects = [], [], []
    for fp in files:
        label = _parse_label(fp)
        if label is None:
            continue
        try:
            res  = extract_features_single(fp, n_fft=n_fft, hop=hop, doppler_hz=doppler_hz)
            feat = _spectrogram_to_stat_vector(res["dfs_power"])
            X.append(feat)
            y.append(label)
            parts = os.path.splitext(os.path.basename(fp))[0].split("_")
            subjects.append(parts[-2] if len(parts) >= 2 else "unk")
        except Exception as e:
            print(f"  [SKIP] {os.path.basename(fp)}: {e}")
    return np.array(X, dtype=np.float32), np.array(y), np.array(subjects)


def load_dataset(args):
    feat_files = glob.glob(os.path.join(args.feat_dir, "*.npz"))
    if feat_files:
        print(f"[Data] Loading {len(feat_files)} pre-extracted DFS NPZ from:\n  {args.feat_dir}")
        return load_from_feature_npz(args.feat_dir)
    else:
        print(f"[Data] Computing DFS on-the-fly from:\n  {args.sanit_dir}")
        return load_from_sanit(args.sanit_dir, args.n_fft, args.hop, args.doppler_hz)


# ══════════════════════════════════════════════════════════════
# B. PCA + MLP 파이프라인
# ══════════════════════════════════════════════════════════════

def build_pipeline(n_pca: int, hidden_layers: tuple, activation: str,
                   max_iter: int, lr_init: float) -> Pipeline:
    """
    StandardScaler → PCA(n_pca) → MLPClassifier

    DFS 통계 벡터 (n_freq×4 ≈ 256dim)는 상관 성분이 많으므로
    PCA로 압축하여 분류에 효과적인 주성분만 사용.
    """
    mlp = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation=activation,
        solver="adam",
        alpha=1e-4,            # L2 정규화
        batch_size=32,
        learning_rate="adaptive",
        learning_rate_init=lr_init,
        max_iter=max_iter,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        verbose=False,
    )
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=n_pca, random_state=42)),
        ("mlp",    mlp),
    ])


# ══════════════════════════════════════════════════════════════
# C. 학습 & 평가
# ══════════════════════════════════════════════════════════════

def _plot_confusion(cm, classes, ts, tag, result_dir):
    fig, ax = plt.subplots(figsize=(5, 4), facecolor=BG)
    ax.set_facecolor(PANEL); ax.spines[:].set_color(GRID)
    ax.tick_params(colors=TEXT)

    im = ax.imshow(cm, cmap="Blues", vmin=0)
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, color=TEXT)
    ax.set_yticks(range(len(classes)));
    ax.set_yticklabels(classes, color=TEXT, rotation=90, va="center")
    ax.set_xlabel("Predicted", color=TEXT); ax.set_ylabel("True", color=TEXT)
    ax.set_title(f"DFS-MLP Confusion Matrix", color=TEXT, fontsize=12, pad=8)

    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-10)
    for i in range(len(classes)):
        for j in range(len(classes)):
            tc = "white" if cm_norm[i, j] > 0.5 else TEXT
            ax.text(j, i - 0.15, f"{cm_norm[i,j]*100:.1f}%",
                    ha="center", va="center", fontsize=12, color=tc, fontweight="bold")
            ax.text(j, i + 0.2,  f"({cm[i,j]})",
                    ha="center", va="center", fontsize=9,  color=tc)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.ax.tick_params(colors=TEXT, labelsize=8)
    plt.tight_layout()

    out = os.path.join(result_dir, f"dfs_confusion_{ts}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"[Result] Confusion matrix → {out}")


def train_and_evaluate(X, y, subjects, args):
    os.makedirs(args.model_dir,  exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)

    le      = LabelEncoder()
    y_enc   = le.fit_transform(y)
    classes = le.classes_

    print(f"\n[Train] Classes: {classes}  |  Total: {len(X)}  |  Feature dim: {X.shape[1]}")
    for cls in classes:
        print(f"  {cls}: {(y == cls).sum()} samples")

    # ── Hold-out 80/20
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )
    print(f"\n[Train] Hold-out → Train={len(X_tr)}, Test={len(X_te)}")

    n_pca_actual = min(args.n_pca, X_tr.shape[0], X_tr.shape[1])
    pipe = build_pipeline(
        n_pca=n_pca_actual,
        hidden_layers=tuple(args.hidden),
        activation=args.activation,
        max_iter=args.epochs,
        lr_init=args.lr,
    )
    pipe.fit(X_tr, y_tr)

    y_pred = pipe.predict(X_te)
    acc    = accuracy_score(y_te, y_pred)
    report = classification_report(y_te, y_pred, target_names=classes)
    cm     = confusion_matrix(y_te, y_pred)

    print(f"\n[Result] Test Accuracy: {acc:.4f}")
    print(report)

    # ── 결과 저장
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(args.result_dir, f"dfs_mlp_report_{ts}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("DFS-MLP Classification Report\n")
        f.write(f"Timestamp  : {ts}\n")
        f.write(f"Feature    : DFS Power Spectrogram → Freq-axis stats (mean/std/max/skew per bin)\n")
        f.write(f"             n_fft={args.n_fft}, hop={args.hop}, doppler_hz=±{args.doppler_hz}Hz\n")
        f.write(f"             Raw dim={X.shape[1]}  →  PCA n_components={n_pca_actual}\n")
        f.write(f"Hidden     : {args.hidden}  Activation={args.activation}\n")
        f.write(f"Epochs     : {args.epochs}  LR={args.lr}\n")
        f.write(f"Train/Test : {len(X_tr)} / {len(X_te)}\n\n")
        f.write(f"Test Accuracy: {acc:.4f}\n\n")
        f.write(report)
        f.write(f"\nConfusion Matrix:\n{cm}\n")
    print(f"[Result] Report → {report_path}")

    _plot_confusion(cm, classes, ts, "dfs", args.result_dir)

    # ── 5-Fold CV
    print("\n[CV] 5-Fold Cross Validation...")
    cv_accs = []
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y_enc)):
        cv_pipe = build_pipeline(n_pca_actual, tuple(args.hidden),
                                  args.activation, args.epochs, args.lr)
        cv_pipe.fit(X[tr_idx], y_enc[tr_idx])
        fold_acc = accuracy_score(y_enc[val_idx], cv_pipe.predict(X[val_idx]))
        cv_accs.append(fold_acc)
        print(f"  Fold {fold+1}: {fold_acc:.4f}")
    print(f"  CV Mean±Std: {np.mean(cv_accs):.4f} ± {np.std(cv_accs):.4f}")

    # ── 결과 파일에 CV 추가
    with open(report_path, "a", encoding="utf-8") as f:
        f.write(f"\n5-Fold CV: {[round(a,4) for a in cv_accs]}\n")
        f.write(f"CV Mean: {np.mean(cv_accs):.4f}  CV Std: {np.std(cv_accs):.4f}\n")

    # ── 모델 저장
    model_path = os.path.join(args.model_dir, "dfs_mlp.pkl")
    le_path    = os.path.join(args.model_dir, "dfs_label_encoder.pkl")
    meta_path  = os.path.join(args.model_dir, "dfs_mlp_meta.json")

    with open(model_path, "wb") as f: pickle.dump(pipe, f)
    with open(le_path,    "wb") as f: pickle.dump(le,   f)

    meta = {
        "feature_type": "dfs_stat",
        "n_fft":        args.n_fft,
        "hop":          args.hop,
        "doppler_hz":   args.doppler_hz,
        "raw_feat_dim": int(X.shape[1]),
        "n_pca":        n_pca_actual,
        "hidden":       args.hidden,
        "activation":   args.activation,
        "classes":      list(classes),
        "test_accuracy": float(acc),
        "cv_mean":       float(np.mean(cv_accs)),
        "cv_std":        float(np.std(cv_accs)),
        "timestamp":     ts,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4, ensure_ascii=False)

    print(f"[Model] Saved → {model_path}")
    print(f"[Model] Meta  → {meta_path}")
    return pipe, le


# ══════════════════════════════════════════════════════════════
# D. 추론
# ══════════════════════════════════════════════════════════════

def run_inference(args):
    model_path = args.model_path or os.path.join(args.model_dir, "dfs_mlp.pkl")
    le_path    = os.path.join(args.model_dir, "dfs_label_encoder.pkl")
    meta_path  = os.path.join(args.model_dir, "dfs_mlp_meta.json")

    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found: {model_path}"); sys.exit(1)

    with open(model_path, "rb") as f: pipe = pickle.load(f)
    with open(le_path,    "rb") as f: le   = pickle.load(f)
    with open(meta_path,  "r")  as f: meta = json.load(f)

    n_fft      = meta.get("n_fft",      args.n_fft)
    hop        = meta.get("hop",         args.hop)
    doppler_hz = meta.get("doppler_hz",  args.doppler_hz)
    classes    = meta.get("classes",     le.classes_.tolist())

    # feat 추출 모듈
    _ext = os.path.join(BASE_DIR, "feature_extraction", "dfs")
    if _ext not in sys.path: sys.path.insert(0, _ext)
    from extract_dfs import extract_features_single

    targets = [args.infer_file] if args.infer_file else []
    if not targets:
        import random; random.seed(42)
        files   = sorted(glob.glob(os.path.join(args.feat_dir, "*.npz")))
        if not files:
            files = sorted(glob.glob(os.path.join(args.sanit_dir, "*.npz")))
        targets = random.sample(files, min(20, len(files)))

    print(f"\n{'─'*68}")
    print(f"{'File':<40} {'True':>6} {'Pred':>6} {'Conf':>8}")
    print(f"{'─'*68}")

    correct = 0
    for fp in targets:
        true_label = _parse_label(fp) or "?"
        try:
            # feature NPZ 우선 로드
            fname = os.path.basename(fp)
            feat_fp = os.path.join(args.feat_dir, fname)
            if os.path.exists(feat_fp):
                d    = np.load(feat_fp, allow_pickle=True)
                feat = _spectrogram_to_stat_vector(d["dfs_power"].astype(np.float32))
            else:
                res  = extract_features_single(fp, n_fft=n_fft, hop=hop, doppler_hz=doppler_hz)
                feat = _spectrogram_to_stat_vector(res["dfs_power"])

            prob  = pipe.predict_proba(feat.reshape(1, -1))[0]
            pred  = le.inverse_transform([prob.argmax()])[0]
            conf  = prob.max()
            match = "✓" if pred == true_label else "✗"
            print(f"  {fname[:36]:<40} {true_label:>6} {pred:>6} {conf:>8.3f} {match}")
            if pred == true_label: correct += 1
        except Exception as e:
            print(f"  {os.path.basename(fp)[:36]:<40} ERR: {e}")

    valid = len([f for f in targets if _parse_label(f)])
    if valid > 0:
        print(f"{'─'*68}")
        print(f"Accuracy: {correct}/{valid} = {correct/valid:.4f}")


# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════

def _parse_args():
    p = argparse.ArgumentParser(
        description="DFS-MLP: 전력 스펙트로그램 주파수 통계 특징 기반 보폭 분류",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--feat_dir",    default=DEFAULT_FEAT_DIR,   help="DFS feature NPZ dir")
    p.add_argument("--sanit_dir",   default=DEFAULT_SANIT_DIR,  help="Sanitization NPZ dir")
    # DFS 파라미터 (on-the-fly 시)
    p.add_argument("--n_fft",       type=int,   default=64,    help="STFT FFT 크기 (기본: 64)")
    p.add_argument("--hop",         type=int,   default=4,     help="STFT hop (기본: 4)")
    p.add_argument("--doppler_hz",  type=float, default=50.0,  help="유효 도플러 대역 ±Hz (기본: 50)")
    # PCA
    p.add_argument("--n_pca",       type=int,   default=32,    help="PCA 주성분 수 (기본: 32)")
    # MLP
    p.add_argument("--hidden",      type=int, nargs="+", default=[256, 128, 64])
    p.add_argument("--activation",  default="relu")
    p.add_argument("--epochs",      type=int,   default=300)
    p.add_argument("--lr",          type=float, default=1e-3)
    # 저장
    p.add_argument("--model_dir",   default=DEFAULT_MODEL_DIR)
    p.add_argument("--result_dir",  default=DEFAULT_RESULT_DIR)
    # 추론
    p.add_argument("--inference",   action="store_true")
    p.add_argument("--model_path",  default=None)
    p.add_argument("--infer_file",  default=None)
    return p.parse_args()


def main():
    args = _parse_args()
    if args.inference:
        run_inference(args)
        return

    X, y, subjects = load_dataset(args)
    if len(X) == 0:
        print("[ERROR] No valid samples loaded."); sys.exit(1)

    print(f"[Data] Loaded: X={X.shape}, labels={set(y)}")
    train_and_evaluate(X, y, subjects, args)


if __name__ == "__main__":
    main()
