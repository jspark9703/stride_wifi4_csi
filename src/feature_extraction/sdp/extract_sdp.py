"""
SDP (Spatial Doppler Profile) Feature Extractor
=================================================
입력  : data/sanitization/sanitization/*.npz  (keys: 'time', 'csi')
출력  : data/feature_extraction/sdp/*.npz     (keys: 'features', 'label', 'subject', 'filename')

SDP 알고리즘 개요
------------------
SDP는 CSI 신호의 자기상관함수(ACF)를 기반으로 산란체(사람의 신체)의 속도 분포를
도메인 불변(Domain-independent) 방식으로 추출합니다.

STEP 1. 진폭(Amplitude) 추출
    - 위상 대신 진폭만 사용하여 도메인 쉬프트(위치/방향 변화)에 강인한 특징 확보
    - shape: (N_packets, N_subcarriers)

STEP 2. 슬라이딩 윈도우 분할 (Window Segmentation)
    - 전체 시계열을 W_T 크기의 겹치는(overlapping) 윈도우로 분할
    - 각 윈도우 → (W_T × N_S) 2D 행렬

STEP 3. ACF 텐서 생성  [N_lag × W_T × N_S]
    - 각 윈도우의 각 서브캐리어 시계열에 대해 lag τ = 0..N_lag-1 까지 ACF 계산
    - ACF(τ, t, s) = Σ_t [ x(t,s) · x(t+τ, s) ]  (정규화 포함)
    - 복잡도 O(N_lag × W_T × N_S) → NumPy 벡터화로 최적화

STEP 4. 서브캐리어 축 평균 + 확률적 정규화
    - N_lag × W_T 행렬로 압축 (서브캐리어 평균)
    - 각 열(패킷 축)을 합이 1이 되도록 확률 정규화 (Probabilistic normalization)
    - 결과: N_lag × W_T 2D SDP 이미지

STEP 5. 평탄화 → 특징 벡터 (N_lag × W_T,) → ML 입력

파라미터
--------
--n_lag     : ACF 최대 lag 샘플 수 (기본: 20)
--wt        : 슬라이딩 윈도우 패킷 수 (기본: 100, 100Hz × 1s)
--hop       : 윈도우 hop (기본: 50, 50% 겹침)
--n_windows : 시계열에서 추출할 최대 윈도우 수 (기본: 1, 중앙 윈도우 사용)
--sanit_dir : 입력 NPZ 디렉터리
--out_dir   : 출력 NPZ 디렉터리
"""

import os
import glob
import argparse
import time
import json
from datetime import datetime

import numpy as np

# ─────────────────────────────────────────────
# 기본 경로
# ─────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_SANIT_DIR = os.path.join(BASE_DIR, "data", "sanitization", "sanitization")
DEFAULT_OUT_DIR   = os.path.join(BASE_DIR, "data", "feature_extraction", "sdp")
DEFAULT_LOG_DIR   = os.path.join(BASE_DIR, "data", "result", "feature_extraction", "sdp")

# ─────────────────────────────────────────────
# 유틸리티
# ─────────────────────────────────────────────
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _parse_label_subject(filename: str):
    """파일명 패턴: csi_YYMMDD_HHMMSS_<subject>_<label>.npz"""
    stem = os.path.splitext(os.path.basename(filename))[0]
    parts = stem.split("_")
    label_raw = parts[-1].lower()
    if "big" in label_raw:
        label = "big"
    elif "small" in label_raw or "smal" in label_raw:
        label = "small"
    else:
        label = "unknown"
    subject = parts[-2] if len(parts) >= 2 else "unknown"
    return label, subject


# ─────────────────────────────────────────────
# STEP 1: 진폭 추출
# ─────────────────────────────────────────────
def load_amplitude(npz_path: str) -> np.ndarray:
    """
    Returns
    -------
    amplitude : (N_packets, N_subcarriers=108) float32
    """
    data = np.load(npz_path)
    csi = data["csi"]                       # complex, shape (N, 108)
    return np.abs(csi).astype(np.float32)


# ─────────────────────────────────────────────
# STEP 2: 슬라이딩 윈도우 분할
# ─────────────────────────────────────────────
def sliding_windows(amplitude: np.ndarray, wt: int, hop: int, n_windows: int):
    """
    Parameters
    ----------
    amplitude : (N, NS)
    wt        : 윈도우 크기 (패킷 수)
    hop       : 윈도우 이동 간격
    n_windows : 추출할 최대 윈도우 수

    Returns
    -------
    windows : list of (wt, NS) arrays
    """
    N, NS = amplitude.shape
    if N < wt:
        # zero-pad
        pad = np.zeros((wt - N, NS), dtype=np.float32)
        amplitude = np.vstack([amplitude, pad])
        N = wt

    starts = list(range(0, N - wt + 1, hop))
    if not starts:
        starts = [0]

    # 중앙 윈도우 우선 사용 (파일 중간 구간이 보행 동작 밀도 높음)
    mid = len(starts) // 2
    # n_windows 개 선택 (중앙 기준)
    half = n_windows // 2
    idx_range = range(
        max(0, mid - half),
        min(len(starts), mid - half + n_windows)
    )
    selected = [starts[i] for i in idx_range]
    if not selected:
        selected = starts[:n_windows]

    windows = [amplitude[s : s + wt] for s in selected]
    return windows


# ─────────────────────────────────────────────
# STEP 3: ACF 텐서 → SDP 2D 매트릭스
# ─────────────────────────────────────────────
def compute_sdp(window: np.ndarray, n_lag: int) -> np.ndarray:
    """
    ACF 기반 SDP 2D 매트릭스 계산 (NumPy 벡터화, O(N_lag × W_T × N_S) 최적화).

    Parameters
    ----------
    window : (W_T, N_S) float  — 단일 윈도우 진폭 행렬
    n_lag  : ACF lag 수 (τ = 1 .. n_lag)

    Returns
    -------
    sdp_matrix : (n_lag, W_T) float32  — 정규화된 2D SDP
    """
    WT, NS = window.shape

    # 평균 제거 (centering) — 도메인 무관 통계 특성 강화
    w = window - window.mean(axis=0, keepdims=True)   # (WT, NS)

    # ACF 텐서: (n_lag, WT, NS)
    # acf_tensor[τ, t, s] = w[t, s] * w[t+τ, s]  (lag 기준 cross-product)
    # 벡터화 구현: lag 별로 슬라이싱 후 element-wise product
    acf_tensor = np.zeros((n_lag, WT, NS), dtype=np.float32)

    for tau in range(1, n_lag + 1):
        # t = 0 .. WT-tau-1 에서 유효한 곱 계산, 나머지는 0
        valid_len = WT - tau
        if valid_len <= 0:
            break
        acf_tensor[tau - 1, :valid_len, :] = (
            w[:valid_len, :] * w[tau : tau + valid_len, :]
        )

    # STEP 4a. 서브캐리어 축 평균 → (n_lag, WT)
    sdp_2d = acf_tensor.mean(axis=2)   # (n_lag, WT)

    # STEP 4b. 절댓값 취하기 (음수 ACF 값 처리)
    sdp_2d = np.abs(sdp_2d)

    # STEP 4c. 확률적 정규화: 각 열(시간 축, WT 방향)을 합이 1이 되도록 정규화
    col_sum = sdp_2d.sum(axis=0, keepdims=True) + 1e-10   # (1, WT)
    sdp_2d = sdp_2d / col_sum                              # (n_lag, WT)

    return sdp_2d.astype(np.float32)


# ─────────────────────────────────────────────
# 파일 하나 처리
# ─────────────────────────────────────────────
def extract_features_single(
    npz_path: str,
    n_lag: int = 20,
    wt: int = 100,
    hop: int = 50,
    n_windows: int = 1,
) -> np.ndarray:
    """
    Returns
    -------
    feature_vec : (n_windows * n_lag * wt,) float32
        n_windows > 1 이면 여러 윈도우 SDP 를 이어 붙인 벡터
    """
    amplitude = load_amplitude(npz_path)      # (N, 108)
    windows = sliding_windows(amplitude, wt=wt, hop=hop, n_windows=n_windows)

    sdp_list = []
    for win in windows:
        sdp = compute_sdp(win, n_lag=n_lag)   # (n_lag, wt)
        sdp_list.append(sdp.ravel())          # (n_lag * wt,)

    # 부족한 윈도우 zero-pad
    expected_dim = n_lag * wt
    while len(sdp_list) < n_windows:
        sdp_list.append(np.zeros(expected_dim, dtype=np.float32))

    feature_vec = np.concatenate(sdp_list).astype(np.float32)
    return feature_vec


# ─────────────────────────────────────────────
# 배치 처리
# ─────────────────────────────────────────────
def run_sdp_extraction(
    sanit_dir: str = DEFAULT_SANIT_DIR,
    out_dir: str = DEFAULT_OUT_DIR,
    log_dir: str = DEFAULT_LOG_DIR,
    n_lag: int = 20,
    wt: int = 100,
    hop: int = 50,
    n_windows: int = 1,
):
    ensure_dir(out_dir)
    ensure_dir(log_dir)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_subdir = os.path.join(log_dir, run_id.split("_")[0])
    ensure_dir(log_subdir)

    all_files = sorted(glob.glob(os.path.join(sanit_dir, "*.npz")))
    total = len(all_files)
    if total == 0:
        print(f"[SDP] No NPZ files found in {sanit_dir}")
        return

    feature_dim = n_windows * n_lag * wt
    print(
        f"[SDP] Processing {total} files  |  "
        f"n_lag={n_lag}, wt={wt}, hop={hop}, n_windows={n_windows}"
    )
    print(f"[SDP] Output feature dimension = {feature_dim}")

    start = time.time()
    success, failed_list = 0, []

    for i, fp in enumerate(all_files):
        fname = os.path.basename(fp)
        try:
            feat = extract_features_single(
                fp, n_lag=n_lag, wt=wt, hop=hop, n_windows=n_windows
            )
            label, subject = _parse_label_subject(fp)

            out_path = os.path.join(out_dir, fname)
            np.savez_compressed(
                out_path,
                features=feat,
                label=np.array(label),
                subject=np.array(subject),
                filename=np.array(fname),
            )
            success += 1
        except Exception as e:
            failed_list.append({"file": fname, "error": str(e)})
            print(f"  [FAIL] {fname}: {e}")

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{total}] done...")

    elapsed = time.time() - start

    # ─── 로그 저장
    log_payload = {
        "execution_timestamp": datetime.now().isoformat(),
        "pipeline_stage": "feature_extraction_sdp",
        "params": {
            "n_lag": n_lag,
            "wt": wt,
            "hop": hop,
            "n_windows": n_windows,
            "feature_dim": feature_dim,
        },
        "summary": {
            "total_files": total,
            "success": success,
            "failed": len(failed_list),
            "elapsed_sec": round(elapsed, 2),
        },
        "failures": failed_list,
    }
    log_path = os.path.join(log_subdir, f"{run_id}_sdp_extraction_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_payload, f, indent=4, ensure_ascii=False)

    print(
        f"\n[SDP] Done. Success={success}, Failed={len(failed_list)}, "
        f"Elapsed={elapsed:.1f}s\n"
        f"  Features saved → {out_dir}\n"
        f"  Log saved      → {log_path}"
    )


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def _parse_args():
    parser = argparse.ArgumentParser(
        description="SDP (Spatial Doppler Profile) CSI Amplitude Feature Extractor"
    )
    parser.add_argument("--n_lag",     type=int, default=20,  help="ACF max lag samples (default: 20)")
    parser.add_argument("--wt",        type=int, default=100, help="Sliding window size in packets (default: 100)")
    parser.add_argument("--hop",       type=int, default=50,  help="Slide hop in packets (default: 50)")
    parser.add_argument("--n_windows", type=int, default=1,   help="# of windows per file to use (default: 1)")
    parser.add_argument("--sanit_dir", default=DEFAULT_SANIT_DIR, help="Sanitized NPZ input directory")
    parser.add_argument("--out_dir",   default=DEFAULT_OUT_DIR,   help="Feature NPZ output directory")
    parser.add_argument("--log_dir",   default=DEFAULT_LOG_DIR,   help="JSON log output directory")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_sdp_extraction(
        sanit_dir=args.sanit_dir,
        out_dir=args.out_dir,
        log_dir=args.log_dir,
        n_lag=args.n_lag,
        wt=args.wt,
        hop=args.hop,
        n_windows=args.n_windows,
    )
