"""
pipeline/windowing.py
======================
슬라이딩 윈도우 모듈 — raw CSV 기반 (Pipeline 첫 번째 스텝).

파이프라인 순서
--------------
  raw/          → [window]     → windowed/        (CSV)
  windowed/     → [preprocess] → preprocessed/    (NPZ)
  preprocessed/ → [sanitize]   → sanitization/    (NPZ)
  sanitization/ → [extract]    → feature_extraction/
  feature_extraction/ → [train] → models/

window 스텝 역할
----------------
- raw CSV 파일을 시간(초) 기준으로 슬라이딩 윈도우 분할
- 각 윈도우를 독립 CSV 파일로 저장
- preprocess 스텝이 windowed CSV를 읽어 NPZ로 변환

윈도우 파일 네이밍 규칙:
  원본  : csi_260331_011319_minhyeok_big.csv
  윈도우: csi_260331_011319_minhyeok_w00_big.csv
           → label은 마지막 '_' 다음 (기존 _parse_label_subject 호환)

파라미터:
  window_sec : 윈도우 크기 [초] — 예) 2.5
  hop_sec    : hop 크기 [초]    — overlap = 1 - hop / window
"""

import os
import glob
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════
#  공통 유틸리티
# ══════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════
#  CSV 기반 슬라이딩 윈도우 (MAIN — raw → windowed CSV)
# ══════════════════════════════════════════════════════════════

def apply_sliding_window_csv(
    raw_dir:          str,
    out_dir:          str,
    window_sec:       float = 2.5,
    hop_sec:          float = 0.5,
    min_segment_size: float = 0.0,
    log_dir:          str   = None,
) -> str:
    """
    raw_dir 내 모든 raw CSV를 시간 기반 슬라이딩 윈도우로 분할하고
    windowed CSV 파일로 저장. (preprocess 스텝 입력 호환)

    Parameters
    ----------
    raw_dir           : 원본 CSV 디렉토리 (하위 폴더 포함 재귀 탐색)
    out_dir           : windowed CSV 저장 디렉토리
    window_sec        : 윈도우 크기 [초]
    hop_sec           : hop 크기 [초]
    min_segment_size  : 세그먼트 유효 최소 길이 [초] (0.0 = 비활성)
                        window 쪽로 잘라나고 남은 마지막 부분 세그먼트의
                        실제 시간 길이가 이 값 미만이면 펴기.
    log_dir           : JSON 로그 저장 디렉토리 (None 이면 저장 안 함)

    Returns
    -------
    out_dir : 결과 디렉토리 경로
    """
    os.makedirs(out_dir, exist_ok=True)

    # raw CSV 파일 탐색 (재귀 포함)
    all_files = sorted(glob.glob(os.path.join(raw_dir, "**", "*.csv"), recursive=True))
    if not all_files:
        all_files = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))

    if not all_files:
        print(f"[Window] No CSV files found in {raw_dir}")
        return out_dir

    print(
        f"[Window] CSV mode  window={window_sec}s  hop={hop_sec}s  "
        f"overlap={1 - hop_sec/window_sec:.0%}  files={len(all_files)}"
    )

    t0            = time.time()
    total_windows = 0
    success       = 0
    failed        = []

    for fp in all_files:
        try:
            df   = pd.read_csv(fp)
            stem = os.path.splitext(os.path.basename(fp))[0]
            base, label = _split_stem(stem)

            if "timestamp" not in df.columns:
                failed.append({"file": os.path.basename(fp), "error": "no timestamp column"})
                continue

            # 시작 시점 기준 경과 시간(초) 계산
            ts    = pd.to_datetime(df["timestamp"], format="ISO8601")
            t_rel = (ts - ts.iloc[0]).dt.total_seconds().values
            T     = t_rel[-1]   # 전체 지속 시간 [초]

            windows = []
            start_t = 0.0
            w_idx   = 0
            skipped = 0

            while start_t + window_sec <= T + 1e-6:
                end_t = start_t + window_sec
                mask  = (t_rel >= start_t - 1e-9) & (t_rel < end_t - 1e-9)
                if mask.sum() > 0:
                    w_df = df[mask].copy()
                    # min_segment_size 필터링:
                    # 실제 시간 기간이 최소 길이 미만이면 펴기
                    if min_segment_size > 0.0:
                        w_ts    = pd.to_datetime(w_df["timestamp"], format="ISO8601")
                        seg_dur = (w_ts.iloc[-1] - w_ts.iloc[0]).total_seconds()
                        if seg_dur < min_segment_size:
                            skipped += 1
                            start_t += hop_sec
                            w_idx   += 1
                            continue
                    windows.append((w_idx, w_df))
                start_t += hop_sec
                w_idx   += 1

            # 신호가 window_sec보다 짧으면 전체를 단일 윈도우로 처리
            if not windows:
                windows = [(0, df.copy())]
                skipped = 0

            for i, w_df in windows:
                out_name = f"{base}_w{i:02d}_{label}.csv"
                w_df.to_csv(os.path.join(out_dir, out_name), index=False)

            total_windows += len(windows)
            success       += 1
            if skipped:
                print(f"  [Window] {os.path.basename(fp)}: {skipped} segment(s) skipped (< {min_segment_size}s)")

        except Exception as e:
            failed.append({"file": os.path.basename(fp), "error": str(e)})
            print(f"  [FAIL] {os.path.basename(fp)}: {e}")

    elapsed = time.time() - t0
    print(
        f"[Window] Done → {total_windows} windowed CSV from {success} files  "
        f"({elapsed:.1f}s)  → {out_dir}"
    )

    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        run_id  = datetime.now().strftime("%Y%m%d_%H%M%S")
        payload = {
            "timestamp": datetime.now().isoformat(),
            "mode": "csv",
            "params": {
                "window_sec":       window_sec,
                "hop_sec":          hop_sec,
                "min_segment_size": min_segment_size,
                "overlap_ratio":    round(1 - hop_sec / window_sec, 4),
            },
            "summary": {
                "input_files":   len(all_files),
                "success":       success,
                "failed":        len(failed),
                "total_windows": total_windows,
                "elapsed_sec":   round(elapsed, 2),
            },
            "failures": failed,
        }
        log_path = os.path.join(log_dir, f"{run_id}_windowing_log.json")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4, ensure_ascii=False)
        print(f"[Window] Log → {log_path}")

    return out_dir


# ══════════════════════════════════════════════════════════════
#  NPZ 기반 슬라이딩 윈도우 (Legacy — sanitized NPZ 분할)
# ══════════════════════════════════════════════════════════════

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

    N    = csi_arr.shape[0]
    stem = os.path.splitext(os.path.basename(npz_path))[0]
    base, label = _split_stem(stem)

    windows = []

    if N < window_samples:
        # 신호가 윈도우보다 짧으면 zero-pad 후 단일 윈도우
        pad = window_samples - N
        csi_pad = np.vstack([
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
    sanit_dir:  str,
    out_dir:    str,
    window_sec: float = 1.0,
    hop_sec:    float = 0.5,
    fs:         int   = 100,
    log_dir:    str   = None,
) -> str:
    """
    [Legacy] sanit_dir 내 모든 NPZ를 슬라이딩 윈도우로 분할하여 out_dir에 저장.
    현재 파이프라인에서는 apply_sliding_window_csv() 를 사용.
    """
    window_samples = int(round(window_sec * fs))
    hop_samples    = int(round(hop_sec    * fs))

    os.makedirs(out_dir, exist_ok=True)

    all_files = sorted(glob.glob(os.path.join(sanit_dir, "*.npz")))
    if not all_files:
        print(f"[Window/NPZ] No NPZ files found in {sanit_dir}")
        return out_dir

    print(
        f"[Window/NPZ] window={window_sec}s ({window_samples}samp)  "
        f"hop={hop_sec}s ({hop_samples}samp)  "
        f"overlap={1 - hop_sec/window_sec:.0%}  files={len(all_files)}"
    )

    t0            = time.time()
    total_windows = 0
    success, failed = 0, []

    for fp in all_files:
        try:
            segs = segment_single(fp, window_samples, hop_samples)
            for seg in segs:
                out_path = os.path.join(out_dir, seg["stem"] + ".npz")
                np.savez_compressed(out_path, time=seg["time"], csi=seg["csi"])
            total_windows += len(segs)
            success       += 1
        except Exception as e:
            failed.append({"file": os.path.basename(fp), "error": str(e)})
            print(f"  [FAIL] {os.path.basename(fp)}: {e}")

    elapsed = time.time() - t0
    print(
        f"[Window/NPZ] Done → {total_windows} windows from {success} files  "
        f"({elapsed:.1f}s)  → {out_dir}"
    )

    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        run_id  = datetime.now().strftime("%Y%m%d_%H%M%S")
        payload = {
            "timestamp": datetime.now().isoformat(),
            "mode": "npz",
            "params": {
                "window_sec":    window_sec,
                "window_samples": window_samples,
                "hop_sec":       hop_sec,
                "hop_samples":   hop_samples,
                "fs":            fs,
                "overlap_ratio": round(1 - hop_sec / window_sec, 4),
            },
            "summary": {
                "input_files":   len(all_files),
                "success":       success,
                "failed":        len(failed),
                "total_windows": total_windows,
                "elapsed_sec":   round(elapsed, 2),
            },
            "failures": failed,
        }
        log_path = os.path.join(log_dir, f"{run_id}_windowing_log.json")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4, ensure_ascii=False)
        print(f"[Window/NPZ] Log → {log_path}")

    return out_dir
