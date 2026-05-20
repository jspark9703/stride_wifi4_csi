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
# PyTorch MLP Adapter
# ══════════════════════════════════════════════════════════════
import sys
if os.path.join(BASE_DIR, "src", "learning", "mlp") not in sys.path:
    sys.path.append(os.path.join(BASE_DIR, "src", "learning", "mlp"))
from pytorch_mlp_utils import train_and_evaluate_pytorch, run_inference_pytorch

def train_and_evaluate(X, y, subjects, args):
    feature_desc = f"Feature    : TD-DFS velocity time series → 19-dim statistical vector\n             Δt search [{args.delta_t_min}, {args.delta_t_max}]  fc={args.fc_hz/1e9:.3f}GHz"
    meta_extras = {"delta_t_min": args.delta_t_min, "delta_t_max": args.delta_t_max, "fc_hz": args.fc_hz}
    return train_and_evaluate_pytorch(X, y, subjects, args, "tddfs", feature_desc, meta_extras=meta_extras, use_pca=False)

def _ext(fp, args, meta):
    _ext_path = os.path.join(BASE_DIR, "feature_extraction", "td-dfs")
    if _ext_path not in sys.path: sys.path.insert(0, _ext_path)
    from extract_tddfs import extract_features_single
    res = extract_features_single(fp, delta_t_min=meta.get("delta_t_min", args.delta_t_min), delta_t_max=meta.get("delta_t_max", args.delta_t_max), fc_hz=meta.get("fc_hz", args.fc_hz))
    return _velocity_to_stat_vector(res["velocity"], res.get("fdop"))

def run_inference(args):
    import random
    random.seed(42)
    targets = [args.infer_file] if args.infer_file else []
    if not targets:
        files = sorted(glob.glob(os.path.join(args.feat_dir, "*.npz")))
        if not files:
            files = sorted(glob.glob(os.path.join(args.sanit_dir, "*.npz")))
        targets = random.sample(files, min(20, len(files)))
        
    run_inference_pytorch(args, "tddfs", _ext, targets, use_pca=False)

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
