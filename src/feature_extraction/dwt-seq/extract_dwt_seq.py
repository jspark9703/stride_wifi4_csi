"""
DWT-seq (Denoised Sequence) Feature Extractor
==============================================
입력  : data/sanitization/sanitization/*.npz  (keys: 'time', 'csi')
출력  : data/feature_extraction/dwt-seq/*.npz (keys: 'features', 'label', 'subject', 'filename')

파이프라인 개요
---------------
1. NPZ 로드 → 복소 CSI 에서 **진폭(Amplitude)** 만 추출  [N_packets × 108]
2. PCA 로 서브캐리어 축 압축 → 상위 K개 주성분 시계열  [N_packets × K]
3. 각 PC 시계열에 DWT 분해 (sym3, L 레벨)
   - 최상위 레벨 coefficient(detail) 에서 MAD 기반 Soft Thresholding (노이즈 제거)
4. Soft Thresholding 적용 후 재합성된 신호
   → 원본 길이로 크롭 (zero-padding 제거)
   → **최종 특징: 노이즈 제거된 PCA 시계열 (N_orig, K)**

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
DEFAULT_SANIT_DIR = os.path.join(BASE_DIR, "data", "sanitization", "sanitization", "260406")
DEFAULT_OUT_DIR   = os.path.join(BASE_DIR, "data", "feature_extraction", "dwt-seq", "260406")
DEFAULT_LOG_DIR   = os.path.join(BASE_DIR, "results", "baseline", "feature_extraction")

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
    """Returns amplitude : (N_packets, N_subcarriers) float32"""
    data = np.load(npz_path)
    csi = data["csi"]           # complex128, shape (N, 108)
    amplitude = np.abs(csi).astype(np.float32)
    return amplitude


# ─────────────────────────────────────────────
# STEP 2: PCA 降維
# ─────────────────────────────────────────────
def apply_pca(amplitude: np.ndarray, n_components: int = 6, del_pca_1: bool = False) -> np.ndarray:
    """
    서브캐리어 108차원 → n_components 주성분

    Parameters
    ----------
    del_pca_1 : bool
        True면 첫 번째 주성분(PC1) 제거 후 나머지 n_components-1개 반환.

    Returns
    -------
    pca_streams : (N_packets, n_components) or (N_packets, n_components-1) float32
    """
    if del_pca_1:
        n_calc = min(n_components + 1, amplitude.shape[0])
    else:
        n_calc = n_components if amplitude.shape[0] >= n_components else amplitude.shape[0]

    pca = PCA(n_components=n_calc)
    pca_streams = pca.fit_transform(amplitude)

    if del_pca_1:
        pca_streams = pca_streams[:, 1:]
        pca_streams = pca_streams[:, :n_components]

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
    max_level = pywt.dwt_max_level(len(signal_1d), wavelet)
    level = min(level, max_level)

    coeffs = pywt.wavedec(signal_1d, wavelet, level=level)

    # 최상위 레벨(detail) 에서 MAD 기반 σ 추정
    highest_detail = coeffs[-1]
    sigma = np.median(np.abs(highest_detail)) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(signal_1d)))

    # Detail coefficients 에만 soft thresholding 적용
    thresholded = [coeffs[0]] + [soft_threshold(c, threshold) for c in coeffs[1:]]

    denoised = pywt.waverec(thresholded, wavelet)
    return denoised[: len(signal_1d)]


# ─────────────────────────────────────────────
# 파일 하나 처리
# ─────────────────────────────────────────────
def extract_features_single(
    npz_path:  str,
    wavelet:   str  = "sym3",
    level:     int  = 10,
    n_pca:     int  = 6,
    del_pca_1: bool = False,
) -> np.ndarray:
    """
    DWT 노이즈 제거된 PCA 시퀀스 추출.

    Returns
    -------
    seq : (N_orig, n_pca_used) float32
        N_orig = 원본 패킷 수 (zero-padding 제거)
        n_pca_used = n_pca-1 if del_pca_1 else n_pca
    """
    amplitude = load_amplitude(npz_path)     # (N_orig, 108)
    N_orig = amplitude.shape[0]

    # 짧은 시계열 zero-padding
    filter_len = pywt.Wavelet(wavelet).dec_len
    min_len    = (filter_len - 1) * (2 ** level)
    if amplitude.shape[0] < min_len:
        pad = np.zeros((min_len - amplitude.shape[0], amplitude.shape[1]), dtype=np.float32)
        amplitude = np.vstack([amplitude, pad])

    pca_streams = apply_pca(amplitude, n_components=n_pca, del_pca_1=del_pca_1)  # (N_padded, K)
    K = pca_streams.shape[1]

    denoised_parts = []
    for k in range(K):
        sig = pca_streams[:, k].astype(np.float64)
        denoised = dwt_denoise(sig, wavelet, level)
        denoised_parts.append(denoised)

    # Stack → crop to original length
    denoised_matrix = np.stack(denoised_parts, axis=1)  # (N_padded, K)
    seq = denoised_matrix[:N_orig, :].astype(np.float32)  # (N_orig, K)

    return seq


# ─────────────────────────────────────────────
# 배치 처리
# ─────────────────────────────────────────────
def run_seq_extraction(
    sanit_dir: str = DEFAULT_SANIT_DIR,
    out_dir:   str = DEFAULT_OUT_DIR,
    log_dir:   str = DEFAULT_LOG_DIR,
    wavelet:   str = "sym3",
    level:     int = 10,
    n_pca:     int = 6,
    del_pca_1: bool = False,
):
    ensure_dir(out_dir)
    ensure_dir(log_dir)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_subdir = os.path.join(log_dir, run_id.split("_")[0])
    ensure_dir(log_subdir)

    all_files = sorted(glob.glob(os.path.join(sanit_dir, "**", "*.npz"), recursive=True))
    total = len(all_files)
    if total == 0:
        print(f"[DWT-seq] No NPZ files found in {sanit_dir}")
        return

    n_used = n_pca - 1 if del_pca_1 else n_pca
    print(f"[DWT-seq] Processing {total} files  |  wavelet={wavelet}, level={level}, n_pca={n_pca}, del_pca_1={del_pca_1}")
    print(f"[DWT-seq] Output feature shape = (variable_length, {n_used})")

    start = time.time()
    results = []
    success, failed_list = 0, []
    seq_lens = []

    for i, fp in enumerate(all_files):
        fname = os.path.basename(fp)
        try:
            seq = extract_features_single(fp, wavelet=wavelet, level=level, n_pca=n_pca, del_pca_1=del_pca_1)
            label, subject = _parse_label_subject(fp)

            out_path = os.path.join(out_dir, fname)
            np.savez_compressed(
                out_path,
                features=seq,
                label=np.array(label),
                subject=np.array(subject),
                filename=np.array(fname),
            )
            results.append({"file": fname, "label": label, "subject": subject, "seq_len": seq.shape[0]})
            seq_lens.append(seq.shape[0])
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
        "pipeline_stage": "feature_extraction_dwt_seq",
        "params": {
            "wavelet":   wavelet,
            "level":     level,
            "n_pca":     n_pca,
            "del_pca_1": del_pca_1,
        },
        "summary": {
            "total_files": total,
            "success": success,
            "failed": len(failed_list),
            "elapsed_sec": round(elapsed, 2),
            "seq_len_min": int(min(seq_lens)) if seq_lens else 0,
            "seq_len_max": int(max(seq_lens)) if seq_lens else 0,
            "seq_len_mean": float(np.mean(seq_lens)) if seq_lens else 0.0,
        },
        "failures": failed_list,
    }
    log_path = os.path.join(log_subdir, f"{run_id}_dwt_seq_extraction_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_payload, f, indent=4, ensure_ascii=False)

    print(
        f"\n[DWT-seq] Done. Success={success}, Failed={len(failed_list)}, "
        f"Elapsed={elapsed:.1f}s\n"
        f"  Seq lengths: min={min(seq_lens) if seq_lens else 0}, "
        f"max={max(seq_lens) if seq_lens else 0}, "
        f"mean={np.mean(seq_lens):.1f}\n"
        f"  Features saved → {out_dir}\n"
        f"  Log saved      → {log_path}"
    )


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def _parse_args():
    parser = argparse.ArgumentParser(
        description="DWT-seq (Denoised Sequence) CSI Feature Extractor"
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
    run_seq_extraction(
        sanit_dir=args.sanit_dir,
        out_dir=args.out_dir,
        log_dir=args.log_dir,
        wavelet=args.wavelet,
        level=args.level,
        n_pca=args.n_pca,
    )
