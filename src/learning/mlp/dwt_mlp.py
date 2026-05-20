"""
DWT-MLP Classifier
===================
DWT 특징을 추출한 뒤 간단한 MLP 로 big/small 보폭 이진 분류를 수행합니다.

특징 추출 → (option A) data/feature_extraction/dwt/*.npz 가 있으면 그대로 로드
           → (option B) 없으면 sanitization NPZ 에서 on-the-fly 계산

실행 예시
---------
  # 기본 (학습 + 평가 + 모델 저장)
  python dwt_mlp.py

  # feature NPZ 가 이미 있을 때 경로 지정
  python dwt_mlp.py --feat_dir ../../data/feature_extraction/dwt

  # 학습 없이 저장된 모델로 추론만
  python dwt_mlp.py --inference --model_path models/dwt_mlp.pkl
  
  # 추론 대상 파일 또는 디렉토리 지정
    python dwt_mlp.py --inference --model_path models/dwt_mlp.pkl --infer_file data/sanitization/sanitization/sample_big_01.npz
    python dwt_mlp.py --inference --is_feat --infer_dir ../../data/feature_extraction/dwt/260406
출력
----
  models/dwt_mlp.pkl           학습된 sklearn MLP 모델
  models/dwt_label_encoder.pkl 레이블 인코더
  results/dwt_mlp_report.txt   분류 리포트 + 혼동행렬
  results/dwt_confusion.png    혼동행렬 시각화
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

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.model_selection  import train_test_split, StratifiedKFold
from sklearn.metrics          import (
    classification_report, confusion_matrix, accuracy_score,
)
from sklearn.pipeline import Pipeline

# ─── 경로 ────────────────────────────────────────────────────
HERE      = os.path.dirname(os.path.abspath(__file__))
BASE_DIR  = os.path.abspath(os.path.join(HERE, "..", ".."))

DEFAULT_FEAT_DIR   = os.path.join(BASE_DIR, "data", "feature_extraction", "dwt","260406")
DEFAULT_SANIT_DIR  = os.path.join(BASE_DIR, "data", "sanitization",  "sanitization")
DEFAULT_MODEL_DIR  = os.path.join(HERE, "models")
DEFAULT_RESULT_DIR = os.path.join(HERE, "results")

# ─── 스타일 ──────────────────────────────────────────────────
BG        = "#0D1117"
PANEL     = "#161B22"
TEXT      = "#E6EDF3"
GRID      = "#30363D"
C_BIG     = "#EF5350"
C_SMALL   = "#42A5F5"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT,
    "xtick.color": TEXT, "ytick.color": TEXT,
    "text.color": TEXT, "grid.color": GRID,
    "font.family": "DejaVu Sans", "font.size": 10,
})


# ══════════════════════════════════════════════════════════════
# A. 특징 로드 — feature NPZ 또는 on-the-fly 계산
# ══════════════════════════════════════════════════════════════
def _parse_label(filename):
    stem = os.path.splitext(os.path.basename(filename))[0].lower()
    parts = stem.split("_")
    lraw = parts[-1]
    if "big"   in lraw: return "big"
    if "small" in lraw or "smal" in lraw: return "small"
    return None


def load_from_feature_npz(feat_dir):
    """data/feature_extraction/dwt/*.npz → X, y, subjects"""
    files = sorted(glob.glob(os.path.join(feat_dir, "*.npz")))
    X, y, subjects = [], [], []
    for fp in files:
        d = np.load(fp, allow_pickle=True)
        label = str(d["label"])
        if label not in ("big", "small"):
            continue
        X.append(d["features"].astype(np.float32))
        y.append(label)
        subjects.append(str(d["subject"]))
    return np.array(X), np.array(y), np.array(subjects)


def load_from_sanit(sanit_dir,
                    wavelet="sym3", level=10, n_pca=6):
    """sanitization/*.npz → on-the-fly DWT 특징 계산"""
    # feature extraction 모듈 동적 import
    _ext_path = os.path.join(BASE_DIR, "feature_extraction", "dwt")
    if _ext_path not in sys.path:
        sys.path.insert(0, _ext_path)
    from extract_dwt import extract_features_single

    files = sorted(glob.glob(os.path.join(sanit_dir, "*.npz")))
    X, y, subjects = [], [], []
    for fp in files:
        label = _parse_label(fp)
        if label is None:
            continue
        try:
            feat = extract_features_single(fp, wavelet=wavelet, level=level, n_pca=n_pca)
            X.append(feat)
            y.append(label)
            parts = os.path.splitext(os.path.basename(fp))[0].split("_")
            subjects.append(parts[-2] if len(parts) >= 2 else "unk")
        except Exception as e:
            print(f"  [SKIP] {os.path.basename(fp)}: {e}")
    return np.array(X, dtype=np.float32), np.array(y), np.array(subjects)


def load_dataset(feat_dir, sanit_dir, wavelet, level, n_pca):
    """feature NPZ 우선, 없으면 on-the-fly 계산"""
    feat_files = glob.glob(os.path.join(feat_dir, "*.npz"))
    if feat_files:
        print(f"[Data] Loading {len(feat_files)} pre-extracted feature NPZ from:\n  {feat_dir}")
        return load_from_feature_npz(feat_dir)
    else:
        print(f"[Data] Feature NPZ not found. Computing on-the-fly from:\n  {sanit_dir}")
        return load_from_sanit(sanit_dir, wavelet, level, n_pca)



# ══════════════════════════════════════════════════════════════
# PyTorch MLP Adapter
# ══════════════════════════════════════════════════════════════
import sys
if os.path.join(BASE_DIR, "src", "learning", "mlp") not in sys.path:
    sys.path.append(os.path.join(BASE_DIR, "src", "learning", "mlp"))
from pytorch_mlp_utils import train_and_evaluate_pytorch, run_inference_pytorch

def train_and_evaluate(X, y, subjects, args):
    feature_desc = f"Feature   : DWT (wavelet={args.wavelet}, level={args.level}, n_pca={args.n_pca})"
    meta_extras = {"wavelet": args.wavelet, "level": args.level, "n_pca": args.n_pca}
    return train_and_evaluate_pytorch(X, y, subjects, args, "dwt", feature_desc, meta_extras=meta_extras, use_pca=False)

def _ext(fp, args, meta):
    _ext_path = os.path.join(BASE_DIR, "feature_extraction", "dwt")
    if _ext_path not in sys.path: sys.path.insert(0, _ext_path)
    from extract_dwt import extract_features_single
    return extract_features_single(fp, wavelet=meta.get("wavelet", args.wavelet), level=meta.get("level", args.level), n_pca=meta.get("n_pca", args.n_pca))

def run_inference(args):
    import random
    random.seed(42)
    targets = [args.infer_file] if args.infer_file else []
    if not targets:
        files = sorted(glob.glob(os.path.join(args.feat_dir, "*.npz")))
        if not files:
            files = sorted(glob.glob(os.path.join(args.sanit_dir, "*.npz")))
        targets = random.sample(files, min(20, len(files)))
        
    run_inference_pytorch(args, "dwt", _ext, targets, use_pca=False)

def main():
    args = _parse_args()
    if getattr(args, "inference", False):
        run_inference(args)
        return

    X, y, subjects = load_dataset(args.feat_dir, args.sanit_dir, args.wavelet, args.level, args.n_pca)
    if len(X) == 0:
        print("[ERROR] No valid samples loaded."); sys.exit(1)

    train_and_evaluate(X, y, subjects, args)

if __name__ == "__main__":
    main()
