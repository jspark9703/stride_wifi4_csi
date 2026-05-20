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
# PyTorch MLP Adapter
# ══════════════════════════════════════════════════════════════
import sys
if os.path.join(BASE_DIR, "src", "learning", "mlp") not in sys.path:
    sys.path.append(os.path.join(BASE_DIR, "src", "learning", "mlp"))
from pytorch_mlp_utils import train_and_evaluate_pytorch, run_inference_pytorch

def train_and_evaluate(X, y, subjects, args):
    feature_desc = f"Feature    : SDP Polar Histogram ({args.r_bins}x{args.theta_bins})\n             n_lag={args.n_lag}, wt={args.wt}, hop={args.hop}"
    meta_extras = {"r_bins": args.r_bins, "theta_bins": args.theta_bins, "n_lag": args.n_lag, "wt": args.wt, "hop": args.hop}
    return train_and_evaluate_pytorch(X, y, subjects, args, "sdp", feature_desc, meta_extras=meta_extras, use_pca=True)

def _ext(fp, args, meta):
    _ext_path = os.path.join(BASE_DIR, "feature_extraction", "sdp")
    if _ext_path not in sys.path: sys.path.insert(0, _ext_path)
    from extract_sdp import extract_features_single
    res = extract_features_single(fp, n_lag=meta.get("n_lag", args.n_lag), wt=meta.get("wt", args.wt), hop=meta.get("hop", args.hop), r_bins=meta.get("r_bins", args.r_bins), theta_bins=meta.get("theta_bins", args.theta_bins))
    return res["sdp_hist"]

def run_inference(args):
    import random
    random.seed(42)
    targets = [args.infer_file] if args.infer_file else []
    if not targets:
        files = sorted(glob.glob(os.path.join(args.feat_dir, "*.npz")))
        if not files:
            files = sorted(glob.glob(os.path.join(args.sanit_dir, "*.npz")))
        targets = random.sample(files, min(20, len(files)))
        
    run_inference_pytorch(args, "sdp", _ext, targets, use_pca=True)

def main():
    args = _parse_args()
    if getattr(args, "inference", False):
        run_inference(args)
        return

    X, y, subjects = load_dataset(args.feat_dir, args.sanit_dir, args.n_lag, args.wt)
    if len(X) == 0:
        print("[ERROR] No valid samples loaded."); sys.exit(1)

    train_and_evaluate(X, y, subjects, args)

if __name__ == "__main__":
    main()
