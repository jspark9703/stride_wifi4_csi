import os
import glob
import ast
import asyncio
import time
import json
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt

# ==========================================
# CONFIGURATION PARAMETERS
# ==========================================
TARGET_FS = 100   # Target sampling frequency (Hz)
MAX_GAP_MS = 20.0 # Only interpolate gaps larger than or equal to this (ms)

# --- Hampel Filter ---
HAMPEL_ENABLED   = True  # Outlier removal (default: true)
HAMPEL_WINDOW    = 7     # Filter window size (3-11, odd numbers)
HAMPEL_THRESHOLD = 5.0   # Outlier detection threshold (1.0-10.0)

# --- Low-Pass Filter ---
LOWPASS_ENABLED = False  # Noise reduction (default: false)
LOWPASS_CUTOFF  = 11.0   # Cutoff frequency in Hz (5.0-20.0)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PREPROCESSED_DIR = os.path.join(BASE_DIR, "data", "sanitization", "preprocessed")
JSON_LOG_DIR = os.path.join(BASE_DIR, "data", "result", "sanitization", "json")

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

class TimeSeriesPreprocessor:
    def __init__(
        self,
        target_fs=TARGET_FS,
        max_gap_ms=MAX_GAP_MS,
        hampel_enabled=HAMPEL_ENABLED,
        hampel_window=HAMPEL_WINDOW,
        hampel_threshold=HAMPEL_THRESHOLD,
        lowpass_enabled=LOWPASS_ENABLED,
        lowpass_cutoff=LOWPASS_CUTOFF,
    ):
        self.target_fs        = target_fs
        self.max_gap_ms       = max_gap_ms
        self.hampel_enabled   = hampel_enabled
        self.hampel_window    = hampel_window
        self.hampel_threshold = hampel_threshold
        self.lowpass_enabled  = lowpass_enabled
        self.lowpass_cutoff   = lowpass_cutoff
        # 802.11n 40MHz 유효 부반송파 (가드밴드 제외: 2~58, 70~126 = 114개)
        nz = list(range(2, 59)) + list(range(70, 127))
        # 파일럿 부반송파 위치 (11, 25, 53, -53(75), -25(103), -11(117))
        pilots = [11, 25, 53, 75, 103, 117]
        # 최종 108개 유효 데이터 부반송파 선별
        self.valid_indices = [x for x in nz if x not in pilots]

    def parse_csi_string(self, csi_str):
        # json_loads 대신 성능 최적화: ast.literal_eval 유지하지만 dtype float32
        csi_list = ast.literal_eval(csi_str)
        csi_array = np.array(csi_list, dtype=np.float32)
        
        if len(csi_array) != 384:
            raise ValueError(f"Expected 384, but got {len(csi_array)}")
            
        ht_ltf_flat = csi_array[128:]
        complex_ht_ltf = ht_ltf_flat[::2] + 1j * ht_ltf_flat[1::2]
        valid_ht_ltf = complex_ht_ltf[self.valid_indices]
        return valid_ht_ltf
        
    def process_dataframe(self, df):
        csi_matrix = np.array([self.parse_csi_string(s) for s in df['data']])
        amplitudes = np.abs(csi_matrix)
        is_zero_subcarrier = np.all(amplitudes == 0, axis=0)
        active_matrix = csi_matrix[:, ~is_zero_subcarrier]
        return active_matrix

    def extract_gap_stats(self, diff_us, csi_matrix):
        """JSON으로 저장할 Time Gap 및 기초 통계 반환"""
        if len(diff_us) == 0:
            return {}
            
        q10, q25, q50, q75, q90, q95, q99 = np.percentile(diff_us, [10, 25, 50, 75, 90, 95, 99])
        
        # P99 이하 Trimmed Stats
        trimmed_diff = diff_us[diff_us <= q99]
        
        # Large / Small group definition (using q25 and q75)
        small_mask = diff_us <= q25
        large_mask = diff_us >= q75
        
        # For Amp diff, we check packet-to-packet amplitude diff in original matrix
        amp_diffs = np.mean(np.abs(np.diff(np.abs(csi_matrix), axis=0)), axis=1)
        
        small_amp_diffs = amp_diffs[small_mask] if len(amp_diffs) > 0 else [0]
        large_amp_diffs = amp_diffs[large_mask] if len(amp_diffs) > 0 else [0]

        stats = {
            "overall_gap_stats_all_pairs": {
                "pair_count": len(diff_us),
                "mean_us": float(np.mean(diff_us)),
                "std_us": float(np.std(diff_us)),
                "median_us": float(q50),
                "p90_us": float(q90),
                "p95_us": float(q95),
                "p99_us": float(q99),
                "max_us": float(np.max(diff_us))
            },
            "overall_gap_stats_trimmed_le_p99": {
                "pair_count": len(trimmed_diff),
                "mean_us": float(np.mean(trimmed_diff)) if len(trimmed_diff) > 0 else 0,
                "std_us": float(np.std(trimmed_diff)) if len(trimmed_diff) > 0 else 0,
                "median_us": float(np.median(trimmed_diff)) if len(trimmed_diff) > 0 else 0
            },
            "valid_csi_pair_gap_quantiles_us": {
                "q10": float(q10), "q25": float(q25), "q50": float(q50),
                "q75": float(q75), "q90": float(q90), "q95": float(q95), "q99": float(q99)
            },
            "small_gap_group": {
                "definition": "gap <= q25",
                "count": int(np.sum(small_mask)),
                "gap_mean_us": float(np.mean(diff_us[small_mask])) if np.sum(small_mask) > 0 else 0,
                "amp_diff_mean": float(np.mean(small_amp_diffs)) if len(small_amp_diffs) > 0 else 0,
            },
            "large_gap_group": {
                "definition": "gap >= q75",
                "count": int(np.sum(large_mask)),
                "gap_mean_us": float(np.mean(diff_us[large_mask])) if np.sum(large_mask) > 0 else 0,
                "amp_diff_mean": float(np.mean(large_amp_diffs)) if len(large_amp_diffs) > 0 else 0,
            }
        }
        return stats

    # ------------------------------------------------------------------
    # Hampel Filter (per-subcarrier outlier removal)
    # ------------------------------------------------------------------
    def hampel_filter(self, signal: np.ndarray) -> np.ndarray:
        """
        Hampel identifier: 윈도우 내 중앙값 절대 편차(MAD) 기반 이상값 탐지 및 대체.
        signal: 1-D float array (amplitude of a single subcarrier)
        반환: 필터링된 signal (이상값은 윈도우 중앙값으로 대체)
        """
        k = 1.4826  # MAD → σ 변환 상수 (정규분포 가정)
        n = len(signal)
        half = self.hampel_window // 2
        out = signal.copy()

        for i in range(n):
            lo = max(0, i - half)
            hi = min(n, i + half + 1)
            window = signal[lo:hi]
            med = np.median(window)
            mad = k * np.median(np.abs(window - med))
            if mad > 0 and np.abs(signal[i] - med) > self.hampel_threshold * mad:
                out[i] = med
        return out

    def apply_hampel(self, amplitude: np.ndarray) -> np.ndarray:
        """
        amplitude: shape (N_samples, N_subcarriers)
        각 부반송파에 독립적으로 Hampel 필터 적용
        """
        result = np.empty_like(amplitude)
        for sc in range(amplitude.shape[1]):
            result[:, sc] = self.hampel_filter(amplitude[:, sc])
        return result

    # ------------------------------------------------------------------
    # Butterworth Low-Pass Filter (per-subcarrier noise reduction)
    # ------------------------------------------------------------------
    def apply_lowpass(self, amplitude: np.ndarray) -> np.ndarray:
        """
        amplitude: shape (N_samples, N_subcarriers)
        Butterworth 2차 low-pass 필터 (zero-phase, filtfilt)를 각 부반송파에 적용
        """
        nyq = self.target_fs / 2.0
        normalized_cutoff = min(self.lowpass_cutoff / nyq, 0.9999)  # 0 < Wn < 1
        b, a = butter(2, normalized_cutoff, btype='low', analog=False)
        result = np.empty_like(amplitude)
        for sc in range(amplitude.shape[1]):
            result[:, sc] = filtfilt(b, a, amplitude[:, sc])
        return result

    # ------------------------------------------------------------------
    # Selective Interpolation
    # ------------------------------------------------------------------
    def selective_interpolate(self, timestamps, csi_matrix):
        """
        간격이 max_gap_ms (예: 20ms) 이상인 곳에만 패킷을 선형 보간하여 채우고,
        그 이하인 곳의 원본 데이터는 보존합니다.
        보간 후 활성화된 필터(Hampel / Low-pass)를 순서대로 진폭에 적용합니다.
        """
        # Convert timestamps to relative absolute time in seconds
        rel_time = (timestamps - timestamps.iloc[0]).dt.total_seconds().values
        diffs_sec = np.diff(rel_time)

        # 원본 타임라인을 그대로 보유
        target_times = list(rel_time)

        # 임계치를 초과하는 Gaps (>= max_gap_ms) 탐색 후 중간 지점들 생성
        threshold_sec = self.max_gap_ms / 1000.0
        step_sec = 1.0 / self.target_fs

        for i, df_sec in enumerate(diffs_sec):
            if df_sec >= threshold_sec:
                # 시작점(rel_time[i])과 끝점(rel_time[i+1]) 사이에 step_sec 간격으로 추가
                # endpoints는 제외 (이미 원본 배열에 있으므로)
                new_points = np.arange(rel_time[i] + step_sec, rel_time[i + 1], step_sec)
                target_times.extend(new_points)

        # 섞인 시간들을 정렬하여 최종 타임라인 수립
        target_times_array = np.sort(np.unique(target_times))

        # 보간 함수 생성
        f_real = interp1d(rel_time, np.real(csi_matrix), axis=0, kind='linear', fill_value="extrapolate")
        f_imag = interp1d(rel_time, np.imag(csi_matrix), axis=0, kind='linear', fill_value="extrapolate")

        # 전체에 대해 새 타임라인 적용하여 csi 획득
        csi_interpolated = f_real(target_times_array) + 1j * f_imag(target_times_array)

        # ----------------------------------------------------------
        # Post-interpolation filtering (amplitude domain)
        # ----------------------------------------------------------
        amplitude = np.abs(csi_interpolated)
        phase     = np.angle(csi_interpolated)

        if self.hampel_enabled:
            amplitude = self.apply_hampel(amplitude)

        if self.lowpass_enabled:
            amplitude = self.apply_lowpass(amplitude)

        # 필터 적용 후 진폭/위상으로 복소 행렬 재조합
        csi_filtered = amplitude * np.exp(1j * phase)

        # 통계 추출용 차이 (Microseconds 단위)
        diffs_us = diffs_sec * 1e6
        gap_stats = self.extract_gap_stats(diffs_us, csi_matrix)

        return target_times_array, csi_filtered, gap_stats


# --- WORKER FUNCTION ---
def process_single_csv(
    file_path,
    out_dir,
    target_fs,
    max_gap_ms,
    hampel_enabled,
    hampel_window,
    hampel_threshold,
    lowpass_enabled,
    lowpass_cutoff,
):
    basename = os.path.basename(file_path)
    ts_preprocessor = TimeSeriesPreprocessor(
        target_fs=target_fs,
        max_gap_ms=max_gap_ms,
        hampel_enabled=hampel_enabled,
        hampel_window=hampel_window,
        hampel_threshold=hampel_threshold,
        lowpass_enabled=lowpass_enabled,
        lowpass_cutoff=lowpass_cutoff,
    )

    try:
        df = pd.read_csv(file_path)
        if 'timestamp' not in df.columns:
            return basename, False, "Missing timestamp column", {}

        timestamps = pd.to_datetime(df['timestamp'], format='ISO8601')
        # Preprocessing
        csi_raw_complex = ts_preprocessor.process_dataframe(df)

        # Selective Interpolation & Post-filtering & Stats
        prep_t, prep_csi, gap_stats = ts_preprocessor.selective_interpolate(timestamps, csi_raw_complex)

        # Save Preprocessed (.npz)
        prep_path = os.path.join(out_dir, basename.replace('.csv', '.npz'))
        np.savez(prep_path, time=prep_t, csi=prep_csi)

        return basename, True, "Success", gap_stats
    except Exception as e:
        return basename, False, str(e), {}


# --- ASYNC ORCHESTRATOR ---
async def run_preprocessing(
    raw_dir=RAW_DIR,
    out_dir=PREPROCESSED_DIR,
    json_dir=JSON_LOG_DIR,
    target_fs=TARGET_FS,
    max_gap_ms=MAX_GAP_MS,
    hampel_enabled=HAMPEL_ENABLED,
    hampel_window=HAMPEL_WINDOW,
    hampel_threshold=HAMPEL_THRESHOLD,
    lowpass_enabled=LOWPASS_ENABLED,
    lowpass_cutoff=LOWPASS_CUTOFF,
    run_id=None,
):
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name = run_id.split('_')[0]
    else:
        dir_name = run_id

    json_dir = os.path.join(json_dir, dir_name)
    ensure_dir(out_dir)
    ensure_dir(json_dir)

    all_files = glob.glob(os.path.join(raw_dir, "*.csv"))
    total_files = len(all_files)

    if total_files == 0:
        print(f"No CSV files found in {raw_dir}")
        return

    print(f"[Preprocessing] Starting batch processing for {total_files} files...")
    start_time = time.time()

    loop = asyncio.get_running_loop()
    results = []

    with ProcessPoolExecutor() as pool:
        func = partial(
            process_single_csv,
            out_dir=out_dir,
            target_fs=target_fs,
            max_gap_ms=max_gap_ms,
            hampel_enabled=hampel_enabled,
            hampel_window=hampel_window,
            hampel_threshold=hampel_threshold,
            lowpass_enabled=lowpass_enabled,
            lowpass_cutoff=lowpass_cutoff,
        )
        futures = [loop.run_in_executor(pool, func, fp) for fp in all_files]

        for idx, f in enumerate(asyncio.as_completed(futures)):
            basename, success, msg, gap_stats = await f
            results.append((basename, success, msg, gap_stats))
            if (idx + 1) % 50 == 0:
                print(f"[Preprocessing] Processed {idx + 1}/{total_files} files...")

    end_time = time.time()
    success_count = sum(1 for r in results if r[1])
    fail_count = total_files - success_count

    # ------------------
    # SAVE GLOBAL JSON
    # ------------------
    json_path = os.path.join(json_dir, f"{run_id}_preprocess_log.json")

    # Aggregate successful gap stats (simple global merge for JSON report)
    # We will log the first successful file's detailed stats to prevent bloating,
    # and simply list the basic file processing results.
    first_valid_stats = {}
    for r in results:
        if r[1] and r[3]:
            first_valid_stats = r[3]
            break

    log_data = {
        "execution_timestamp": datetime.now().isoformat(),
        "pipeline_stage": "preprocessing",
        "hyperparameters": {
            "target_fs": target_fs,
            "max_interp_gap_ms": max_gap_ms,
            "hampel_enabled": hampel_enabled,
            "hampel_window": hampel_window,
            "hampel_threshold": hampel_threshold,
            "lowpass_enabled": lowpass_enabled,
            "lowpass_cutoff_hz": lowpass_cutoff,
        },
        "summary": {
            "total_files": total_files,
            "success": success_count,
            "failed": fail_count,
            "execution_time_sec": round(end_time - start_time, 2)
        },
        "sample_gap_stats_from_data": first_valid_stats,
        "failures": [{"file": r[0], "error": r[2]} for r in results if not r[1]]
    }

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=4, ensure_ascii=False)

    print(f"[Preprocessing] Complete! Success: {success_count}, Failed: {fail_count}. Logs saved to {json_path}")

if __name__ == "__main__":
    asyncio.run(run_preprocessing())
