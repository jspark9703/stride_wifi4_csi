"""
TD-DFS-MLP Classifier
======================
TD-DFS (순간 도플러 속도 시계열) 특징을 MLP 로 big/small 보폭 이진 분류합니다.

특징 구성
---------
TD-DFS 출력은 (N_packets - Δt,) shape의 1D 속도 시계열입니다.
파일마다 시계열 길이가 다르므로 통계 기반 고정 차원 특징 벡터로 변환합니다.

  추출 통계 (19-dim 핵심 고정 특징 벡터):
  ─────────────────────────────────────────
  [전체 속도 통계] (7)
    mean, std, median, min, max, rms(= √mean(v²)), iqr(Q75-Q25)

  [방향 정보] (3)
    접근 비율 (v > 0), 후퇴 비율 (v < 0), 제로 크로싱 수 (정규화)

  [분포 형태] (2)
    왜도(skewness), 첨도(kurtosis)

  [에너지 분포] (4)
    하위 25% / 상위 75% 속도 에너지 비율,
    피크 속도 (max |v|), 피크 도플러 주파수 (|fdop|.max())

  [시간 변화율] (3)
    속도 변화율(diff) 의 mean_abs, std, max_abs

  총 19-dim 특징 벡터 → MLP 입력

MLP 구조
--------
  Input (19)
    └─ Dense(128) → ReLU → Dropout (L2=1e-4)
    └─ Dense(64)  → ReLU → Dropout
    └─ Dense(32)  → ReLU
    └─ Dense(2)   → Softmax

  → sklearn MLPClassifier (StandardScaler + Pipeline without PCA: 차원이 충분히 작음)

실행
----
  python tddfs_mlp.py
  python tddfs_mlp.py --hidden 128 64 32
  python tddfs_mlp.py --inference --model_path models/tddfs_mlp.pkl

출력
----
  models/tddfs_mlp.pkl              학습된 모델
  models/tddfs_label_encoder.pkl    레이블 인코더
  models/tddfs_mlp_meta.json        메타데이터
  results/tddfs_mlp_report_*.txt    분류 리포트
  results/tddfs_confusion_*.png     혼동행렬 시각화
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
from sklearn.metrics         import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline        import Pipeline

# ─── 경로 ────────────────────────────────────────────────────
HERE      = os.path.dirname(os.path.abspath(__file__))
BASE_DIR  = os.path.abspath(os.path.join(HERE, "..", ".."))

DEFAULT_FEAT_DIR   = os.path.join(BASE_DIR, "data", "feature_extraction", "td-dfs")
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

FEAT_DIM = 19   # 고정 특징 벡터 차원


# ══════════════════════════════════════════════════════════════
# A. 특징 추출 — 속도 시계열 → 19-dim 통계 벡터
# ══════════════════════════════════════════════════════════════

def _velocity_to_stat_vector(velocity: np.ndarray,
                              fdop: np.ndarray = None) -> np.ndarray:
    """
    TD-DFS 속도 시계열 → 고정 19-dim 통계 벡터.

    Parameters
    ----------
    velocity : (M,) float32  [cm/s]
    fdop     : (M,) float32  [Hz]  (없으면 속도에서 추정)

    Returns
    -------
    feat : (19,) float32
    """
    v = velocity.astype(np.float64)
    N = len(v)

    # ── [1~7] 전체 속도 통계
    mean_v   = v.mean()
    std_v    = v.std()
    med_v    = np.median(v)
    min_v    = v.min()
    max_v    = v.max()
    rms_v    = np.sqrt((v ** 2).mean())
    iqr_v    = np.percentile(v, 75) - np.percentile(v, 25)

    # ── [8~10] 방향 정보
    pos_ratio  = (v > 0).mean()                          # 접근 비율
    neg_ratio  = (v < 0).mean()                          # 후퇴 비율
    sign_v     = np.sign(v)
    zero_cross = float(np.sum(np.diff(sign_v) != 0)) / max(N - 1, 1)  # 정규화

    # ── [11~12] 분포 형태
    diff_v = v - mean_v
    skew   = (diff_v ** 3).mean() / (std_v ** 3 + 1e-10)
    kurt   = (diff_v ** 4).mean() / (std_v ** 4 + 1e-10) - 3.0

    # ── [13~16] 에너지 분포
    abs_v      = np.abs(v)
    q25, q75   = np.percentile(abs_v, 25), np.percentile(abs_v, 75)
    low_energy = (abs_v[abs_v <= q25] ** 2).mean() if (abs_v <= q25).any() else 0.0
    high_energy= (abs_v[abs_v >= q75] ** 2).mean() if (abs_v >= q75).any() else 0.0
    peak_speed = abs_v.max()

    if fdop is not None and len(fdop) == N:
        peak_fdop = np.abs(fdop.astype(np.float64)).max()
    else:
        peak_fdop = peak_speed   # 대체값

    # ── [17~19] 시간 변화율
    dv          = np.diff(v)
    diff_mean_abs = np.abs(dv).mean() if len(dv) > 0 else 0.0
    diff_std      = dv.std()          if len(dv) > 0 else 0.0
    diff_max_abs  = np.abs(dv).max()  if len(dv) > 0 else 0.0

    feat = np.array([
        mean_v, std_v, med_v, min_v, max_v, rms_v, iqr_v,   # 1-7
        pos_ratio, neg_ratio, zero_cross,                     # 8-10
        skew, kurt,                                           # 11-12
        low_energy, high_energy, peak_speed, peak_fdop,       # 13-16
        diff_mean_abs, diff_std, diff_max_abs,                 # 17-19
    ], dtype=np.float32)

    return feat   # (19,)


def _parse_label(filename: str):
    stem  = os.path.splitext(os.path.basename(filename))[0].lower()
    parts = stem.split("_")
    lraw  = parts[-1]
    if "big"   in lraw: return "big"
    if "small" in lraw or "smal" in lraw: return "small"
    return None


# ── A-1. feature NPZ 로드
def load_from_feature_npz(feat_dir: str):
    """data/feature_extraction/td-dfs/*.npz → X, y, subjects"""
    files = sorted(glob.glob(os.path.join(feat_dir, "*.npz")))
    X, y, subjects = [], [], []
    for fp in files:
        d     = np.load(fp, allow_pickle=True)
        label = str(d["label"])
        if label not in ("big", "small"):
            continue
        try:
            velocity = d["velocity"].astype(np.float32)
            fdop     = d["fdop"].astype(np.float32) if "fdop" in d else None
            feat     = _velocity_to_stat_vector(velocity, fdop)
            X.append(feat)
            y.append(label)
            subjects.append(str(d["subject"]))
        except Exception as e:
            print(f"  [SKIP] {os.path.basename(fp)}: {e}")
    return np.array(X, dtype=np.float32), np.array(y), np.array(subjects)


# ── A-2. on-the-fly 계산
def load_from_sanit(sanit_dir: str, delta_t_min: int,
                    delta_t_max: int, fc_hz: float):
    """sanitization/*.npz → on-the-fly TD-DFS 계산 → 통계 벡터"""
    _ext = os.path.join(BASE_DIR, "feature_extraction", "td-dfs")
    if _ext not in sys.path:
        sys.path.insert(0, _ext)
    from extract_tddfs import extract_features_single

    files = sorted(glob.glob(os.path.join(sanit_dir, "*.npz")))
    X, y, subjects = [], [], []
    for fp in files:
        label = _parse_label(fp)
        if label is None:
            continue
        try:
            res  = extract_features_single(fp, delta_t_min=delta_t_min,
                                            delta_t_max=delta_t_max, fc_hz=fc_hz)
            feat = _velocity_to_stat_vector(res["velocity"], res["fdop"])
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
        print(f"[Data] Loading {len(feat_files)} pre-extracted TD-DFS NPZ from:\n  {args.feat_dir}")
        return load_from_feature_npz(args.feat_dir)
    else:
        print(f"[Data] Computing TD-DFS on-the-fly from:\n  {args.sanit_dir}")
        return load_from_sanit(args.sanit_dir, args.delta_t_min,
                                args.delta_t_max, args.fc_hz)


# ══════════════════════════════════════════════════════════════
# B. MLP 파이프라인  (차원이 작으므로 PCA 없이 직접 MLP)
# ══════════════════════════════════════════════════════════════

def build_pipeline(hidden_layers: tuple, activation: str,
                   max_iter: int, lr_init: float) -> Pipeline:
    """
    StandardScaler → MLPClassifier

    TD-DFS 통계 벡터는 19-dim 으로 이미 저차원이므로
    PCA 없이 StandardScaler + MLP 파이프라인 사용.
    """
    mlp = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation=activation,
        solver="adam",
        alpha=1e-3,             # DWT 보다 강한 L2 (작은 데이터셋 과적합 방지)
        batch_size=32,
        learning_rate="adaptive",
        learning_rate_init=lr_init,
        max_iter=max_iter,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=25,
        verbose=False,
    )
    return Pipeline([
        ("scaler", StandardScaler()),
        ("mlp",    mlp),
    ])


# ══════════════════════════════════════════════════════════════
# C. 학습 & 평가
# ══════════════════════════════════════════════════════════════

FEAT_NAMES = [
    "mean_v", "std_v", "median_v", "min_v", "max_v", "rms_v", "iqr_v",
    "pos_ratio", "neg_ratio", "zero_cross",
    "skewness", "kurtosis",
    "low_energy", "high_energy", "peak_speed", "peak_fdop",
    "diff_mean_abs", "diff_std", "diff_max_abs",
]


def _plot_confusion(cm, classes, ts, result_dir):
    fig, ax = plt.subplots(figsize=(5, 4), facecolor=BG)
    ax.set_facecolor(PANEL); ax.spines[:].set_color(GRID)
    ax.tick_params(colors=TEXT)

    im = ax.imshow(cm, cmap="Blues", vmin=0)
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, color=TEXT)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes, color=TEXT, rotation=90, va="center")
    ax.set_xlabel("Predicted", color=TEXT); ax.set_ylabel("True", color=TEXT)
    ax.set_title("TD-DFS-MLP Confusion Matrix", color=TEXT, fontsize=12, pad=8)

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

    out = os.path.join(result_dir, f"tddfs_confusion_{ts}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"[Result] Confusion matrix → {out}")


def _plot_feature_importance(X, y, classes, ts, result_dir):
    """클래스별 특징 평균 비교 막대그래프 (특징 중요도 대리 지표)."""
    mask_big   = y == "big"
    mask_small = y == "small"
    mu_big     = X[mask_big].mean(axis=0)
    mu_small   = X[mask_small].mean(axis=0)

    # StandardScaler 없이 원시 평균 차이 (상대 크기 비교용)
    diff = np.abs(mu_big - mu_small)
    order = np.argsort(diff)[::-1]

    fig, ax = plt.subplots(figsize=(14, 5), facecolor=BG)
    ax.set_facecolor(PANEL); ax.spines[:].set_color(GRID)
    ax.tick_params(colors=TEXT, labelsize=8)

    x = np.arange(FEAT_DIM)
    ax.bar(x - 0.2, mu_big[order],   0.35, color=C_BIG,   alpha=0.85, label="Big")
    ax.bar(x + 0.2, mu_small[order], 0.35, color=C_SMALL, alpha=0.85, label="Small")

    ax.set_xticks(x)
    ax.set_xticklabels([FEAT_NAMES[i] for i in order],
                       rotation=45, ha="right", fontsize=7)
    ax.set_title("TD-DFS Feature Mean Comparison  ─  Big vs Small (sorted by |diff|)",
                 color=TEXT, fontsize=11, pad=8)
    ax.set_ylabel("Feature Mean Value", color=TEXT, fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=9)

    plt.tight_layout()
    out = os.path.join(result_dir, f"tddfs_feature_importance_{ts}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"[Result] Feature importance → {out}")


def train_and_evaluate(X, y, subjects, args):
    os.makedirs(args.model_dir,  exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)

    le      = LabelEncoder()
    y_enc   = le.fit_transform(y)
    classes = le.classes_

    print(f"\n[Train] Classes: {classes}  |  Total: {len(X)}  |  Feature dim: {X.shape[1]}")
    for cls in classes:
        print(f"  {cls}: {(y == cls).sum()} samples")
    print(f"  Feature names: {FEAT_NAMES}")

    # ── Hold-out 80/20
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )
    print(f"\n[Train] Hold-out → Train={len(X_tr)}, Test={len(X_te)}")

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

    # ── 결과 저장
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(args.result_dir, f"tddfs_mlp_report_{ts}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("TD-DFS-MLP Classification Report\n")
        f.write(f"Timestamp  : {ts}\n")
        f.write(f"Feature    : TD-DFS velocity time series → {FEAT_DIM}-dim statistical vector\n")
        f.write(f"             Δt search [{args.delta_t_min}, {args.delta_t_max}]  fc={args.fc_hz/1e9:.3f}GHz\n")
        f.write(f"Feature names: {FEAT_NAMES}\n")
        f.write(f"Hidden     : {args.hidden}  Activation={args.activation}\n")
        f.write(f"Epochs     : {args.epochs}  LR={args.lr}\n")
        f.write(f"Train/Test : {len(X_tr)} / {len(X_te)}\n\n")
        f.write(f"Test Accuracy: {acc:.4f}\n\n")
        f.write(report)
        f.write(f"\nConfusion Matrix:\n{cm}\n")
    print(f"[Result] Report → {report_path}")

    _plot_confusion(cm, classes, ts, args.result_dir)
    _plot_feature_importance(X, y, classes, ts, args.result_dir)

    # ── 5-Fold CV
    print("\n[CV] 5-Fold Cross Validation...")
    cv_accs = []
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y_enc)):
        cv_pipe = build_pipeline(tuple(args.hidden), args.activation, args.epochs, args.lr)
        cv_pipe.fit(X[tr_idx], y_enc[tr_idx])
        fold_acc = accuracy_score(y_enc[val_idx], cv_pipe.predict(X[val_idx]))
        cv_accs.append(fold_acc)
        print(f"  Fold {fold+1}: {fold_acc:.4f}")
    print(f"  CV Mean±Std: {np.mean(cv_accs):.4f} ± {np.std(cv_accs):.4f}")

    with open(report_path, "a", encoding="utf-8") as f:
        f.write(f"\n5-Fold CV: {[round(a,4) for a in cv_accs]}\n")
        f.write(f"CV Mean: {np.mean(cv_accs):.4f}  CV Std: {np.std(cv_accs):.4f}\n")

    # ── 모델 저장
    model_path = os.path.join(args.model_dir, "tddfs_mlp.pkl")
    le_path    = os.path.join(args.model_dir, "tddfs_label_encoder.pkl")
    meta_path  = os.path.join(args.model_dir, "tddfs_mlp_meta.json")

    with open(model_path, "wb") as f: pickle.dump(pipe, f)
    with open(le_path,    "wb") as f: pickle.dump(le,   f)

    meta = {
        "feature_type":  "tddfs_stat",
        "feat_dim":       FEAT_DIM,
        "feat_names":     FEAT_NAMES,
        "delta_t_min":    args.delta_t_min,
        "delta_t_max":    args.delta_t_max,
        "fc_hz":          args.fc_hz,
        "hidden":         args.hidden,
        "activation":     args.activation,
        "classes":        list(classes),
        "test_accuracy":  float(acc),
        "cv_mean":        float(np.mean(cv_accs)),
        "cv_std":         float(np.std(cv_accs)),
        "timestamp":      ts,
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
    model_path = args.model_path or os.path.join(args.model_dir, "tddfs_mlp.pkl")
    le_path    = os.path.join(args.model_dir, "tddfs_label_encoder.pkl")
    meta_path  = os.path.join(args.model_dir, "tddfs_mlp_meta.json")

    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found: {model_path}"); sys.exit(1)

    with open(model_path, "rb") as f: pipe = pickle.load(f)
    with open(le_path,    "rb") as f: le   = pickle.load(f)
    with open(meta_path,  "r")  as f: meta = json.load(f)

    delta_t_min = meta.get("delta_t_min", args.delta_t_min)
    delta_t_max = meta.get("delta_t_max", args.delta_t_max)
    fc_hz       = meta.get("fc_hz",        args.fc_hz)
    classes     = meta.get("classes",       le.classes_.tolist())

    _ext = os.path.join(BASE_DIR, "feature_extraction", "td-dfs")
    if _ext not in sys.path: sys.path.insert(0, _ext)
    from extract_tddfs import extract_features_single

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
            fname   = os.path.basename(fp)
            feat_fp = os.path.join(args.feat_dir, fname)
            if os.path.exists(feat_fp):
                d    = np.load(feat_fp, allow_pickle=True)
                feat = _velocity_to_stat_vector(
                    d["velocity"].astype(np.float32),
                    d["fdop"].astype(np.float32) if "fdop" in d else None,
                )
            else:
                res  = extract_features_single(fp, delta_t_min=delta_t_min,
                                               delta_t_max=delta_t_max, fc_hz=fc_hz)
                feat = _velocity_to_stat_vector(res["velocity"], res["fdop"])

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
        description="TD-DFS-MLP: 순간 도플러 속도 통계 특징 기반 보폭 분류",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--feat_dir",     default=DEFAULT_FEAT_DIR,   help="TD-DFS feature NPZ dir")
    p.add_argument("--sanit_dir",    default=DEFAULT_SANIT_DIR,  help="Sanitization NPZ dir")
    # TD-DFS 파라미터 (on-the-fly 시)
    p.add_argument("--delta_t_min",  type=int,   default=1,       help="Δt 탐색 최솟값 (기본: 1)")
    p.add_argument("--delta_t_max",  type=int,   default=10,      help="Δt 탐색 최댓값 (기본: 10)")
    p.add_argument("--fc_hz",        type=float, default=5.18e9,  help="WiFi 중심 주파수 Hz (기본: 5.18e9)")
    # MLP
    p.add_argument("--hidden",       type=int, nargs="+", default=[128, 64, 32])
    p.add_argument("--activation",   default="relu")
    p.add_argument("--epochs",       type=int,   default=300)
    p.add_argument("--lr",           type=float, default=1e-3)
    # 저장
    p.add_argument("--model_dir",    default=DEFAULT_MODEL_DIR)
    p.add_argument("--result_dir",   default=DEFAULT_RESULT_DIR)
    # 추론
    p.add_argument("--inference",    action="store_true")
    p.add_argument("--model_path",   default=None)
    p.add_argument("--infer_file",   default=None)
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
