"""
pipeline/windowing.py
======================
온라인 학습 대응 슬라이딩 윈도우 전처리 모듈.

Sanitized NPZ (전체 보행 신호) → 고정 길이 윈도우 단위 NPZ 파일들.
각 윈도우는 독립 샘플로 저장되어 기존 feature extraction 모듈과 호환.

윈도우 파일 네이밍 규칙:
  원본: csi_260406_011319_minhyeok_big.npz
  윈도우: csi_260406_011319_minhyeok_w00_big.npz
           → parts[-1] = 'big' (기존 _parse_label_subject 호환)

파라미터:
  window_sec : 윈도우 크기 [초] — 0.8 | 0.9 | 1.0
  hop_sec    : hop 크기 [초]    — overlap = 1 - hop / window
  fs         : 샘플링 주파수 [Hz] (전처리 target_fs 와 동일)

online 추론 가정:
  window_sec=1.0 → 100샘플 → 1초마다 보폭 분류 1회
  window_sec=0.8 → 80샘플  → 0.8초마다 분류 (빠른 반응)
  hop_sec=0.1    → 90% overlap → 최대 데이터 증강
"""

import os
import glob
import json
import time
from datetime import datetime

import numpy as np


def _split_stem(stem: str):
    """
    파일 stem에서 label(마지막)과 base(나머지)를 분리.
    예: 'csi_260406_011319_minhyeok_big' → ('csi_260406_011319_minhyeok', 'big')
    """
    parts = stem.split("_")
    label_raw = parts[-1].lower()
    if "big" in label_raw:
        label = "big"
    elif "small" in label_raw or "smal" in label_raw:
        label = "small"
    else:
        label = "unknown"
    base = "_".join(parts[:-1])
    return base, label


def segment_single(
    npz_path: str,
    window_samples: int,
    hop_samples: int,
) -> list:
    """
    단일 sanitized NPZ를 슬라이딩 윈도우로 분할.

    Returns
    -------
    list of dict: [{'stem': str, 'time': ndarray, 'csi': ndarray}, ...]
    """
    data = np.load(npz_path, allow_pickle=True)
    time_arr = data["time"].astype(np.float64)   # (N,)
    csi_arr  = data["csi"]                        # (N, 108) complex

    N = csi_arr.shape[0]
    stem = os.path.splitext(os.path.basename(npz_path))[0]
    base, label = _split_stem(stem)

    windows = []

    if N < window_samples:
        # 신호가 윈도우보다 짧으면 zero-pad 후 단일 윈도우로 처리
        pad = window_samples - N
        csi_pad  = np.vstack([
            csi_arr,
            np.zeros((pad, csi_arr.shape[1]), dtype=csi_arr.dtype),
        ])
        dt = (time_arr[-1] - time_arr[0]) / max(N - 1, 1)
        time_pad = np.append(
            time_arr,
            time_arr[-1] + dt * np.arange(1, pad + 1),
        )
        windows.append({
            "stem":  f"{base}_w00_{label}",
            "time":  time_pad,
            "csi":   csi_pad,
        })
        return windows

    starts = list(range(0, N - window_samples + 1, hop_samples))
    if not starts:
        starts = [0]

    for i, s in enumerate(starts):
        windows.append({
            "stem":  f"{base}_w{i:02d}_{label}",
            "time":  time_arr[s : s + window_samples],
            "csi":   csi_arr[s  : s + window_samples],
        })

    return windows


def apply_sliding_window(
    sanit_dir: str,
    out_dir: str,
    window_sec: float = 1.0,
    hop_sec: float    = 0.5,
    fs: int           = 100,
    log_dir: str      = None,
) -> str:
    """
    sanit_dir 내 모든 NPZ를 슬라이딩 윈도우로 분할하여 out_dir에 저장.

    Parameters
    ----------
    sanit_dir    : sanitized NPZ 입력 디렉토리
    out_dir      : 윈도우 NPZ 출력 디렉토리
    window_sec   : 윈도우 크기 [초] (0.8 | 0.9 | 1.0)
    hop_sec      : hop 크기 [초]
    fs           : 샘플링 주파수 (default 100 Hz)
    log_dir      : JSON 로그 저장 디렉토리 (None 이면 로그 저장 안 함)

    Returns
    -------
    out_dir : 결과 디렉토리 경로
    """
    window_samples = int(round(window_sec * fs))
    hop_samples    = int(round(hop_sec    * fs))

    os.makedirs(out_dir, exist_ok=True)

    all_files = sorted(glob.glob(os.path.join(sanit_dir, "*.npz")))
    if not all_files:
        print(f"[Window] No NPZ files found in {sanit_dir}")
        return out_dir

    print(
        f"[Window] window={window_sec}s ({window_samples}samp)  "
        f"hop={hop_sec}s ({hop_samples}samp)  "
        f"overlap={1 - hop_sec/window_sec:.0%}  "
        f"files={len(all_files)}"
    )

    t0 = time.time()
    total_windows = 0
    success, failed = 0, []

    for fp in all_files:
        try:
            segs = segment_single(fp, window_samples, hop_samples)
            for seg in segs:
                out_path = os.path.join(out_dir, seg["stem"] + ".npz")
                np.savez_compressed(out_path, time=seg["time"], csi=seg["csi"])
            total_windows += len(segs)
            success += 1
        except Exception as e:
            failed.append({"file": os.path.basename(fp), "error": str(e)})
            print(f"  [FAIL] {os.path.basename(fp)}: {e}")

    elapsed = time.time() - t0
    print(
        f"[Window] Done → {total_windows} windows from {success} files  "
        f"({elapsed:.1f}s)  → {out_dir}"
    )

    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        run_id  = datetime.now().strftime("%Y%m%d_%H%M%S")
        payload = {
            "timestamp": datetime.now().isoformat(),
            "params": {
                "window_sec": window_sec,
                "window_samples": window_samples,
                "hop_sec": hop_sec,
                "hop_samples": hop_samples,
                "fs": fs,
                "overlap_ratio": round(1 - hop_sec / window_sec, 4),
            },
            "summary": {
                "input_files": len(all_files),
                "success": success,
                "failed": len(failed),
                "total_windows": total_windows,
                "avg_windows_per_file": round(total_windows / max(success, 1), 2),
                "elapsed_sec": round(elapsed, 2),
            },
            "failures": failed,
        }
        log_path = os.path.join(log_dir, f"{run_id}_windowing_log.json")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4, ensure_ascii=False)
        print(f"[Window] Log → {log_path}")

    return out_dir
