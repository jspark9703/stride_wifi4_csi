"""
DWT (Discrete Wavelet Transform) Feature Extractor
====================================================
입력  : data/sanitization/sanitization/*.npz  (keys: 'time', 'csi')
출력  : data/feature_extraction/dwt/*.npz     (keys: 'features', 'label', 'subject', 'filename')

파이프라인 개요
---------------
1. NPZ 로드 → 복소 CSI 에서 **진폭(Amplitude)** 만 추출  [N_packets × 108]
2. PCA 로 서브캐리어 축 압축 → 상위 K개 주성분 시계열  [N_packets × K]
3. 각 PC 시계열에 DWT 분해 (sym3, L 레벨)
   - 최상위 레벨 coefficient(detail) 에서 MAD 기반 Soft Thresholding (노이즈 제거)
4. Soft Thresholding 적용 후 재합성된 신호를 다시 DWT
5. 각 레벨의 Detail/Approximation coefficients 에너지를 특징 벡터로 구성
   - 최종 shape : (K × (L+1),) → 평탄화된 1-D 에너지 벡터

파라미터 (CLI / 함수 인자)
--------------------------
--wavelet     : 웨이블릿 함수 (기본: 'sym3')
--level       : 분해 레벨 (기본: 10)
--n_pca       : PCA 주성분 수 (기본: 6)
--sanit_dir   : 입력 NPZ 디렉터리
--out_dir     : 출력 NPZ 디렉터리
"""

import os
import glob
import argparse
import time
import json
from datetime import datetime

import numpy as np
import pywt
from sklearn.decomposition import PCA

# ─────────────────────────────────────────────
# 기본 경로
# ─────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_SANIT_DIR = os.path.join(BASE_DIR, "data", "sanitization", "sanitization","260406")
DEFAULT_OUT_DIR   = os.path.join(BASE_DIR, "data", "feature_extraction", "dwt", "260406")
DEFAULT_LOG_DIR   = os.path.join(BASE_DIR, "data", "result", "feature_extraction", "dwt")

# ─────────────────────────────────────────────
# 유틸리티
# ─────────────────────────────────────────────
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _parse_label_subject(filename: str):
    """파일명 패턴: csi_YYMMDD_HHMMSS_<subject>_<label>.npz
    label : 'big' 또는 'small'
    subject : 사람 이름
    """
    stem = os.path.splitext(os.path.basename(filename))[0]  # csi_260331_011319_minhyeok_small
    parts = stem.split("_")
    # 마지막 파트가 label, 그 전이 subject
    # 파일명에 'smal' 같은 오타 허용
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
    amplitude : (N_packets, N_subcarriers) float32
    """
    data = np.load(npz_path)
    csi = data["csi"]           # complex128, shape (N, 108)
    amplitude = np.abs(csi).astype(np.float32)
    return amplitude


# ─────────────────────────────────────────────
# STEP 2: PCA 降維
# ─────────────────────────────────────────────
def apply_pca(amplitude: np.ndarray, n_components: int = 6) -> np.ndarray:
    """
    서브캐리어 108차원 → n_components 주성분

    Returns
    -------
    pca_streams : (N_packets, n_components) float32
    """
    if amplitude.shape[0] < n_components:
        n_components = amplitude.shape[0]
    pca = PCA(n_components=n_components)
    pca_streams = pca.fit_transform(amplitude)   # (N, K)
    return pca_streams.astype(np.float32)


# ─────────────────────────────────────────────
# STEP 3: DWT 노이즈 제거 (Soft Thresholding)
# ─────────────────────────────────────────────
def soft_threshold(coeffs, threshold):
    """Soft thresholding of a coefficient array."""
    return np.sign(coeffs) * np.maximum(np.abs(coeffs) - threshold, 0.0)


def dwt_denoise(signal_1d: np.ndarray, wavelet: str, level: int) -> np.ndarray:
    """
    단일 1-D 시계열에 대해 Soft Thresholding 기반 DWT 노이즈 제거.

    Parameters
    ----------
    signal_1d : (T,) float
    wavelet   : 웨이블릿 이름 (예: 'sym3')
    level     : 분해 레벨

    Returns
    -------
    denoised : (T,) float  — 재합성된 노이즈 제거 신호
    """
    # 실제 최대 가능 레벨 제한
    max_level = pywt.dwt_max_level(len(signal_1d), wavelet)
    level = min(level, max_level)

    coeffs = pywt.wavedec(signal_1d, wavelet, level=level)

    # 최상위 레벨(detail) 에서 MAD 기반 σ 추정
    highest_detail = coeffs[-1]
    sigma = np.median(np.abs(highest_detail)) / 0.6745  # Donoho & Johnstone estimator
    threshold = sigma * np.sqrt(2 * np.log(len(signal_1d)))  # universal threshold

    # Detail coefficients (coeffs[1:]) 에만 soft thresholding 적용
    # coeffs[0] = approximation (저주파, 신호 보존)
    thresholded = [coeffs[0]] + [soft_threshold(c, threshold) for c in coeffs[1:]]

    denoised = pywt.waverec(thresholded, wavelet)
    # 재합성 후 길이가 1 길어질 수 있으므로 crop
    return denoised[: len(signal_1d)]


# ─────────────────────────────────────────────
# STEP 4: DWT 에너지 특징 추출
# ─────────────────────────────────────────────
def dwt_energy_features(signal_1d: np.ndarray, wavelet: str, level: int) -> np.ndarray:
    """
    노이즈 제거된 신호에 DWT 를 수행하여 각 스케일의 에너지를 계산.

    Returns
    -------
    energy_vec : (level + 1,) float32
        [E_approx, E_detail_level, ..., E_detail_1]
        인덱스 0 = 최저주파 근사(approximation),
        인덱스 1..L = 각 detail 레벨 (레벨 순차: high→low frequency)
    """
    max_level = pywt.dwt_max_level(len(signal_1d), wavelet)
    level = min(level, max_level)

    coeffs = pywt.wavedec(signal_1d, wavelet, level=level)
    # coeffs[0] = cA_L, coeffs[1..L] = cD_L .. cD_1
    energy_vec = np.array(
        [np.sum(c ** 2) for c in coeffs], dtype=np.float32
    )  # (level+1,)

    # L2 정규화 (0으로 나누기 방지)
    norm = np.linalg.norm(energy_vec)
    if norm > 1e-10:
        energy_vec = energy_vec / norm

    return energy_vec


# ─────────────────────────────────────────────
# 파일 하나 처리
# ─────────────────────────────────────────────
def extract_features_single(
    npz_path: str,
    wavelet: str = "sym3",
    level: int = 10,
    n_pca: int = 6,
) -> np.ndarray:
    """
    Returns
    -------
    feature_vec : (n_pca * (level + 1),) float32
    """
    amplitude = load_amplitude(npz_path)     # (N, 108)

    # 너무 짧은 시계열 예외 처리
    min_len = 2 ** (level + 1)
    if amplitude.shape[0] < min_len:
        # zero-pad
        pad = np.zeros((min_len - amplitude.shape[0], amplitude.shape[1]), dtype=np.float32)
        amplitude = np.vstack([amplitude, pad])

    pca_streams = apply_pca(amplitude, n_components=n_pca)  # (N, K)

    feature_parts = []
    for k in range(pca_streams.shape[1]):
        sig = pca_streams[:, k].astype(np.float64)  # pywt needs float64

        # ① 노이즈 제거
        denoised = dwt_denoise(sig, wavelet, level)

        # ② 에너지 특징 추출
        energy = dwt_energy_features(denoised, wavelet, level)   # (level+1,)
        feature_parts.append(energy)

    feature_vec = np.concatenate(feature_parts).astype(np.float32)  # (K*(L+1),)
    return feature_vec


# ─────────────────────────────────────────────
# 배치 처리
# ─────────────────────────────────────────────
def run_dwt_extraction(
    sanit_dir: str = DEFAULT_SANIT_DIR,
    out_dir: str = DEFAULT_OUT_DIR,
    log_dir: str = DEFAULT_LOG_DIR,
    wavelet: str = "sym3",
    level: int = 10,
    n_pca: int = 6,
):
    ensure_dir(out_dir)
    ensure_dir(log_dir)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_subdir = os.path.join(log_dir, run_id.split("_")[0])
    ensure_dir(log_subdir)

    all_files = sorted(glob.glob(os.path.join(sanit_dir, "**", "*.npz"), recursive=True))
    total = len(all_files)
    if total == 0:
        print(f"[DWT] No NPZ files found in {sanit_dir}")
        return

    print(f"[DWT] Processing {total} files  |  wavelet={wavelet}, level={level}, n_pca={n_pca}")
    feature_dim = n_pca * (level + 1)
    print(f"[DWT] Output feature dimension = {feature_dim}")

    start = time.time()
    results = []
    success, failed_list = 0, []

    for i, fp in enumerate(all_files):
        fname = os.path.basename(fp)
        try:
            feat = extract_features_single(fp, wavelet=wavelet, level=level, n_pca=n_pca)
            label, subject = _parse_label_subject(fp)

            out_path = os.path.join(out_dir, fname)
            np.savez_compressed(
                out_path,
                features=feat,
                label=np.array(label),
                subject=np.array(subject),
                filename=np.array(fname),
            )
            results.append({"file": fname, "label": label, "subject": subject, "feature_dim": len(feat)})
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
        "pipeline_stage": "feature_extraction_dwt",
        "params": {
            "wavelet": wavelet,
            "level": level,
            "n_pca": n_pca,
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
    log_path = os.path.join(log_subdir, f"{run_id}_dwt_extraction_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_payload, f, indent=4, ensure_ascii=False)

    print(
        f"\n[DWT] Done. Success={success}, Failed={len(failed_list)}, "
        f"Elapsed={elapsed:.1f}s\n"
        f"  Features saved → {out_dir}\n"
        f"  Log saved      → {log_path}"
    )


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def _parse_args():
    parser = argparse.ArgumentParser(
        description="DWT-based CSI Amplitude Feature Extractor"
    )
    parser.add_argument("--wavelet",   default="sym3", help="PyWavelets wavelet name (default: sym3)")
    parser.add_argument("--level",     type=int, default=10, help="DWT decomposition levels (default: 10)")
    parser.add_argument("--n_pca",     type=int, default=6,  help="# of PCA components (default: 6)")
    parser.add_argument("--sanit_dir", default=DEFAULT_SANIT_DIR, help="Sanitized NPZ input directory")
    parser.add_argument("--out_dir",   default=DEFAULT_OUT_DIR,   help="Feature NPZ output directory")
    parser.add_argument("--log_dir",   default=DEFAULT_LOG_DIR,   help="JSON log output directory")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_dwt_extraction(
        sanit_dir=args.sanit_dir,
        out_dir=args.out_dir,
        log_dir=args.log_dir,
        wavelet=args.wavelet,
        level=args.level,
        n_pca=args.n_pca,
    )
