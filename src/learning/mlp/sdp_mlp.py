"""
SDP-MLP Classifier
===================
SDP (Spatial Doppler Profile) 특징을 추출한 뒤 간단한 MLP 로
big/small 보폭 이진 분류를 수행합니다.

특징 추출 → (option A) data/feature_extraction/sdp/*.npz 가 있으면 그대로 로드
           → (option B) 없으면 sanitization NPZ 에서 on-the-fly ACF 계산

실행 예시
---------
  # 기본 실행 (학습 + 평가 + 저장)
  python sdp_mlp.py

  # feature NPZ 경로 지정
  python sdp_mlp.py --feat_dir ../../data/feature_extraction/sdp

  # 저장 모델로 추론
  python sdp_mlp.py --inference --model_path models/sdp_mlp.pkl

출력
----
  models/sdp_mlp.pkl           학습된 sklearn MLP 모델
  models/sdp_label_encoder.pkl 레이블 인코더
  models/sdp_mlp_meta.json     하이퍼파라미터 메타데이터
  results/sdp_mlp_report_*.txt 분류 리포트
  results/sdp_confusion_*.png  혼동행렬 시각화
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
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics         import (
    classification_report, confusion_matrix, accuracy_score,
)
from sklearn.pipeline import Pipeline

# ─── 경로 ────────────────────────────────────────────────────
HERE      = os.path.dirname(os.path.abspath(__file__))
BASE_DIR  = os.path.abspath(os.path.join(HERE, "..", ".."))

DEFAULT_FEAT_DIR   = os.path.join(BASE_DIR, "data", "feature_extraction", "sdp")
DEFAULT_SANIT_DIR  = os.path.join(BASE_DIR, "data", "sanitization",  "sanitization")
DEFAULT_MODEL_DIR  = os.path.join(HERE, "models")
DEFAULT_RESULT_DIR = os.path.join(HERE, "results")

# ─── 스타일 ──────────────────────────────────────────────────
BG     = "#0D1117"
PANEL  = "#161B22"
TEXT   = "#E6EDF3"
GRID   = "#30363D"
C_BIG  = "#EF5350"
C_SML  = "#42A5F5"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT,
    "xtick.color": TEXT, "ytick.color": TEXT,
    "text.color": TEXT, "grid.color": GRID,
    "font.family": "DejaVu Sans", "font.size": 10,
})


# ══════════════════════════════════════════════════════════════
# A-0. On-the-fly SDP 계산 (extract_sdp.py 와 동일 로직 인라인)
# ══════════════════════════════════════════════════════════════
def _compute_sdp_inline(npz_path, n_lag, wt):
    data      = np.load(npz_path)
    amplitude = np.abs(data["csi"]).astype(np.float32)
    N, NS     = amplitude.shape
    if N < wt:
        pad = np.zeros((wt - N, NS), dtype=np.float32)
        amplitude = np.vstack([amplitude, pad])
        N = wt
    start  = max(0, (N - wt) // 2)
    window = amplitude[start: start + wt]

    w = window - window.mean(axis=0, keepdims=True)
    acf = np.zeros((n_lag, wt, NS), dtype=np.float32)
    for tau in range(1, n_lag + 1):
        valid = wt - tau
        if valid <= 0: break
        acf[tau - 1, :valid, :] = w[:valid, :] * w[tau: tau + valid, :]

    sdp  = np.abs(acf.mean(axis=2))
    csum = sdp.sum(axis=0, keepdims=True) + 1e-10
    sdp  = (sdp / csum).astype(np.float32)
    return sdp.ravel()   # (n_lag * wt,)


# ══════════════════════════════════════════════════════════════
# A. 특징 로드
# ══════════════════════════════════════════════════════════════
def _parse_label(filename):
    stem  = os.path.splitext(os.path.basename(filename))[0].lower()
    parts = stem.split("_")
    lraw  = parts[-1]
    if "big"   in lraw: return "big"
    if "small" in lraw or "smal" in lraw: return "small"
    return None


def load_from_feature_npz(feat_dir):
    files = sorted(glob.glob(os.path.join(feat_dir, "*.npz")))
    X, y, subjects = [], [], []
    for fp in files:
        d = np.load(fp, allow_pickle=True)
        label = str(d["label"])
        if label not in ("big", "small"): continue
        X.append(d["features"].astype(np.float32))
        y.append(label)
        subjects.append(str(d["subject"]))
    return np.array(X, dtype=np.float32), np.array(y), np.array(subjects)


def load_from_sanit(sanit_dir, n_lag, wt):
    """on-the-fly SDP 계산"""
    files = sorted(glob.glob(os.path.join(sanit_dir, "*.npz")))
    X, y, subjects = [], [], []
    for fp in files:
        label = _parse_label(fp)
        if label is None: continue
        try:
            feat = _compute_sdp_inline(fp, n_lag, wt)
            X.append(feat)
            y.append(label)
            parts = os.path.splitext(os.path.basename(fp))[0].split("_")
            subjects.append(parts[-2] if len(parts) >= 2 else "unk")
        except Exception as e:
            print(f"  [SKIP] {os.path.basename(fp)}: {e}")
    return np.array(X, dtype=np.float32), np.array(y), np.array(subjects)


def load_dataset(feat_dir, sanit_dir, n_lag, wt):
    feat_files = glob.glob(os.path.join(feat_dir, "*.npz"))
    if feat_files:
        print(f"[Data] Loading {len(feat_files)} pre-extracted feature NPZ from:\n  {feat_dir}")
        return load_from_feature_npz(feat_dir)
    else:
        print(f"[Data] Feature NPZ not found. Computing SDP on-the-fly from:\n  {sanit_dir}")
        return load_from_sanit(sanit_dir, n_lag, wt)


# ══════════════════════════════════════════════════════════════
# B. MLP Pipeline
# ══════════════════════════════════════════════════════════════
def build_pipeline(hidden_layers=(512, 256, 128),
                   activation="relu",
                   l2_alpha=1e-4,
                   max_iter=300,
                   lr_init=1e-3):
    mlp = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation=activation,
        solver="adam",
        alpha=l2_alpha,
        batch_size=64,
        learning_rate="adaptive",
        learning_rate_init=lr_init,
        max_iter=max_iter,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        verbose=False,
    )
    return Pipeline([("scaler", StandardScaler()), ("mlp", mlp)])


# ══════════════════════════════════════════════════════════════
# C. 학습 & 평가
# ══════════════════════════════════════════════════════════════
def train_and_evaluate(X, y, subjects, args):
    os.makedirs(args.model_dir,  exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)

    le    = LabelEncoder()
    y_enc = le.fit_transform(y)
    classes = le.classes_

    print(f"\n[Train] Classes: {classes}  |  Total samples: {len(X)}")
    for cls in classes:
        print(f"  {cls}: {(y == cls).sum()} samples")

    # ── Hold-out 80/20
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )
    print(f"\n[Train] Hold-out split → Train={len(X_tr)}, Test={len(X_te)}")

    pipe = build_pipeline(
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

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── 텍스트 리포트 저장
    rpt_path = os.path.join(args.result_dir, f"sdp_mlp_report_{ts}.txt")
    with open(rpt_path, "w", encoding="utf-8") as f:
        f.write("SDP-MLP Classification Report\n")
        f.write(f"Timestamp : {ts}\n")
        f.write(f"Feature   : SDP (n_lag={args.n_lag}, wt={args.wt})\n")
        f.write(f"Hidden    : {args.hidden}\n")
        f.write(f"Epochs    : {args.epochs}  LR={args.lr}\n")
        f.write(f"Train/Test: {len(X_tr)} / {len(X_te)}\n\n")
        f.write(f"Test Accuracy: {acc:.4f}\n\n")
        f.write(report)
        f.write(f"\nConfusion Matrix:\n{cm}\n")
    print(f"[Result] Report → {rpt_path}")

    _plot_confusion(cm, classes, ts, args.result_dir)

    # ── 5-Fold CV
    print("\n[CV] 5-Fold Cross Validation...")
    cv_accs = []
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y_enc)):
        cv_pipe = build_pipeline(tuple(args.hidden), args.activation, max_iter=args.epochs, lr_init=args.lr)
        cv_pipe.fit(X[tr_idx], y_enc[tr_idx])
        fa = accuracy_score(y_enc[val_idx], cv_pipe.predict(X[val_idx]))
        cv_accs.append(fa)
        print(f"  Fold {fold+1}: {fa:.4f}")
    print(f"  CV Mean±Std: {np.mean(cv_accs):.4f} ± {np.std(cv_accs):.4f}")

    # ── 모델 저장
    model_path = os.path.join(args.model_dir, "sdp_mlp.pkl")
    le_path    = os.path.join(args.model_dir, "sdp_label_encoder.pkl")
    with open(model_path, "wb") as f: pickle.dump(pipe, f)
    with open(le_path,    "wb") as f: pickle.dump(le,   f)

    meta = {
        "n_lag": args.n_lag, "wt": args.wt,
        "hidden": args.hidden, "activation": args.activation,
        "classes": list(classes),
        "feature_dim": int(X.shape[1]),
        "test_accuracy": float(acc),
        "cv_mean": float(np.mean(cv_accs)),
        "cv_std":  float(np.std(cv_accs)),
        "timestamp": ts,
    }
    meta_path = os.path.join(args.model_dir, "sdp_mlp_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4, ensure_ascii=False)

    print(f"[Model] Saved → {model_path}")
    print(f"[Model] Meta  → {meta_path}")
    return pipe, le


def _plot_confusion(cm, classes, ts, result_dir):
    fig, ax = plt.subplots(figsize=(5, 4), facecolor=BG)
    ax.set_facecolor(PANEL)
    ax.spines[:].set_color(GRID)
    ax.tick_params(colors=TEXT)

    im = ax.imshow(cm, cmap="Blues", vmin=0)
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, color=TEXT)
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes, color=TEXT, rotation=90, va="center")
    ax.set_xlabel("Predicted", color=TEXT); ax.set_ylabel("True", color=TEXT)
    ax.set_title("SDP-MLP Confusion Matrix", color=TEXT, fontsize=12, pad=8)

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
    out = os.path.join(result_dir, f"sdp_confusion_{ts}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"[Result] Confusion matrix → {out}")


# ══════════════════════════════════════════════════════════════
# D. 추론
# ══════════════════════════════════════════════════════════════
def run_inference(args):
    model_path = args.model_path or os.path.join(args.model_dir, "sdp_mlp.pkl")
    le_path    = os.path.join(args.model_dir, "sdp_label_encoder.pkl")
    meta_path  = os.path.join(args.model_dir, "sdp_mlp_meta.json")

    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found: {model_path}")
        print("  → Train first by running without --inference flag.")
        sys.exit(1)

    with open(model_path, "rb") as f: pipe = pickle.load(f)
    with open(le_path,    "rb") as f: le   = pickle.load(f)
    with open(meta_path)        as f: meta = json.load(f)

    n_lag   = meta.get("n_lag", args.n_lag)
    wt      = meta.get("wt",    args.wt)
    classes = meta.get("classes", le.classes_.tolist())

    print(f"[Inference] Model : {model_path}")
    print(f"[Inference] SDP   : n_lag={n_lag}, wt={wt}")
    print(f"[Inference] Classes: {classes}\n")

    targets = []
    if args.infer_file:
        targets = [args.infer_file]
    else:
        files = sorted(glob.glob(os.path.join(args.sanit_dir, "*.npz")))
        import random; random.seed(42)
        targets = random.sample(files, min(20, len(files)))

    print(f"{'─'*64}")
    print(f"{'File':<40} {'True':>6} {'Pred':>6} {'Conf':>8}  ")
    print(f"{'─'*64}")

    correct = 0
    for fp in targets:
        true_label = _parse_label(fp) or "?"
        try:
            feat      = _compute_sdp_inline(fp, n_lag, wt)
            pred_enc  = pipe.predict(feat.reshape(1, -1))[0]
            proba     = pipe.predict_proba(feat.reshape(1, -1))[0]
            pred_lbl  = le.inverse_transform([pred_enc])[0]
            conf      = proba.max()
            match     = "✓" if pred_lbl == true_label else "✗"
            print(f"  {os.path.basename(fp)[:36]:<38} {true_label:>6} {pred_lbl:>6} {conf:>8.3f} {match}")
            if pred_lbl == true_label: correct += 1
        except Exception as e:
            print(f"  {os.path.basename(fp)[:36]:<38} ERR: {e}")

    valid = len([f for f in targets if _parse_label(f)])
    if valid > 0:
        print(f"{'─'*64}")
        print(f"Inference Accuracy: {correct}/{valid} = {correct/valid:.4f}")


# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════
def _parse_args():
    p = argparse.ArgumentParser(description="SDP-MLP Classifier for WiFi CSI stride classification")
    # 데이터
    p.add_argument("--feat_dir",   default=DEFAULT_FEAT_DIR,  help="Pre-extracted feature NPZ dir")
    p.add_argument("--sanit_dir",  default=DEFAULT_SANIT_DIR, help="Sanitization NPZ dir (on-the-fly)")
    # SDP 파라미터
    p.add_argument("--n_lag", type=int, default=20,  help="ACF max lag (default: 20)")
    p.add_argument("--wt",    type=int, default=100, help="Window size in packets (default: 100)")
    # MLP 하이퍼파라미터
    p.add_argument("--hidden",     type=int, nargs="+", default=[512, 256, 128], help="Hidden layer sizes")
    p.add_argument("--activation", default="relu", help="Activation function (relu/tanh/logistic)")
    p.add_argument("--epochs",     type=int,   default=300,  help="Max iterations (default: 300)")
    p.add_argument("--lr",         type=float, default=1e-3, help="Initial learning rate (default: 1e-3)")
    # 저장 경로
    p.add_argument("--model_dir",  default=DEFAULT_MODEL_DIR,  help="Model save directory")
    p.add_argument("--result_dir", default=DEFAULT_RESULT_DIR, help="Result save directory")
    # 추론 모드
    p.add_argument("--inference",  action="store_true", help="Run inference with saved model")
    p.add_argument("--model_path", default=None, help="Explicit .pkl path for inference")
    p.add_argument("--infer_file", default=None, help="Single NPZ file to infer (optional)")
    return p.parse_args()


def main():
    args = _parse_args()

    if args.inference:
        run_inference(args)
    else:
        X, y, subjects = load_dataset(
            args.feat_dir, args.sanit_dir, args.n_lag, args.wt
        )
        if len(X) == 0:
            print("[ERROR] No valid samples loaded."); sys.exit(1)

        print(f"[Data] Loaded: X={X.shape}, unique labels={set(y)}")
        train_and_evaluate(X, y, subjects, args)


if __name__ == "__main__":
    main()
