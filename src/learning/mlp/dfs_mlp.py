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
# PyTorch MLP Adapter
# ══════════════════════════════════════════════════════════════
import sys
if os.path.join(BASE_DIR, "src", "learning", "mlp") not in sys.path:
    sys.path.append(os.path.join(BASE_DIR, "src", "learning", "mlp"))
from pytorch_mlp_utils import train_and_evaluate_pytorch, run_inference_pytorch

def train_and_evaluate(X, y, subjects, args):
    feature_desc = f"Feature    : DFS Power Spectrogram → Freq-axis stats (mean/std/max/skew per bin)\n             n_fft={args.n_fft}, hop={args.hop}, doppler_hz=±{args.doppler_hz}Hz"
    meta_extras = {"n_fft": args.n_fft, "hop": args.hop, "doppler_hz": args.doppler_hz}
    return train_and_evaluate_pytorch(X, y, subjects, args, "dfs", feature_desc, meta_extras=meta_extras, use_pca=True)

def _ext(fp, args, meta):
    _ext_path = os.path.join(BASE_DIR, "feature_extraction", "dfs")
    if _ext_path not in sys.path: sys.path.insert(0, _ext_path)
    from extract_dfs import extract_features_single
    res = extract_features_single(fp, n_fft=meta.get("n_fft", args.n_fft), hop=meta.get("hop", args.hop), doppler_hz=meta.get("doppler_hz", args.doppler_hz))
    return _spectrogram_to_stat_vector(res["dfs_power"])

def run_inference(args):
    import random
    random.seed(42)
    targets = [args.infer_file] if args.infer_file else []
    if not targets:
        files = sorted(glob.glob(os.path.join(args.feat_dir, "*.npz")))
        if not files:
            files = sorted(glob.glob(os.path.join(args.sanit_dir, "*.npz")))
        targets = random.sample(files, min(20, len(files)))
        
    run_inference_pytorch(args, "dfs", _ext, targets, use_pca=True)

def main():
    args = _parse_args()
    if getattr(args, "inference", False):
        run_inference(args)
        return

    X, y, subjects = load_dataset(args)
    if len(X) == 0:
        print("[ERROR] No valid samples loaded."); sys.exit(1)

    train_and_evaluate(X, y, subjects, args)

if __name__ == "__main__":
    main()
