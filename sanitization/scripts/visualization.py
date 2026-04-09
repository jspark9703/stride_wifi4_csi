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
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURATION PARAMETERS
# ==========================================
HIST_BINS = 100
HIST_RANGE = (0, 40)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PREPROCESSED_DIR = os.path.join(BASE_DIR, "data", "sanitization", "preprocessed")
SANITIZATION_DIR = os.path.join(BASE_DIR, "data", "sanitization", "sanitization")
RESULT_DIR = os.path.join(BASE_DIR, "data", "result", "sanitization")
JSON_LOG_DIR = os.path.join(RESULT_DIR, "json")
PLOTS_DIR = os.path.join(RESULT_DIR, "plots")

def ensure_dir(path):
    if not os.path.exists(path):    
        os.makedirs(path)

# Extract 114 valid indices to parse raw CSV quickly in Visualization
VALID_INDICES = list(range(5, 59)) + list(range(70, 127))

def parse_csi_fast(csi_str):
    csi_list = ast.literal_eval(csi_str)
    csi_array = np.array(csi_list, dtype=np.float32)
    if len(csi_array) != 384: return None
    ht_ltf = csi_array[128:]
    complex_ht_ltf = ht_ltf[::2] + 1j * ht_ltf[1::2]
    return complex_ht_ltf[VALID_INDICES]

# --- WORKER FUNCTION ---
def extract_stats_single(csv_path, prep_dir, sanit_dir):
    basename = os.path.basename(csv_path)
    prep_path = os.path.join(prep_dir, basename.replace('.csv', '.npz'))
    sanit_path = os.path.join(sanit_dir, basename.replace('.csv', '.npz'))
    
    try:
        # 1. Read Raw CSV (for raw gaps and raw amplitude limits)
        df = pd.read_csv(csv_path)
        if 'timestamp' not in df.columns:
            return basename, False, "Missing timestamp", None
            
        timestamps = pd.to_datetime(df['timestamp'], format='ISO8601')
        raw_intervals = timestamps.diff().dt.total_seconds().values * 1000
        raw_intervals = raw_intervals[~np.isnan(raw_intervals)]
        
        csi_raw_list = []
        for s in df['data']:
            parsed = parse_csi_fast(s)
            if parsed is not None: csi_raw_list.append(parsed)
        csi_raw_matrix = np.array(csi_raw_list)
        
        amplitudes = np.abs(csi_raw_matrix)
        is_zero_subcarrier = np.all(amplitudes == 0, axis=0)
        active_raw_matrix = csi_raw_matrix[:, ~is_zero_subcarrier]
        mean_raw_amp = np.mean(np.abs(active_raw_matrix), axis=0)
        
        # 2. Read Preprocessed NPZ (for unwrapped phase baseline)
        if not os.path.exists(prep_path):
            return basename, False, f"Preprocessed file not found: {prep_path}", None
        prep_data = np.load(prep_path)
        prep_t = prep_data['time']
        prep_csi = prep_data['csi']
        
        # Calculate Unwrapped Phase manually over the first few packets for frequency curve
        # (Averaging unwrapped phase over all packets)
        unwrapped_phases = []
        for p in np.angle(prep_csi):
            unwrapped_phases.append(np.unwrap(p))
        mean_unwrapped_phase = np.mean(unwrapped_phases, axis=0)
        
        # 3. Read Sanitized NPZ
        if not os.path.exists(sanit_path):
            return basename, False, f"Sanitized file not found: {sanit_path}", None
            
        sanit_data = np.load(sanit_path)
        sanitized_full_csi = sanit_data['csi']
        
        prep_intervals = np.diff(prep_t) * 1000
        mean_sanitized_amp = np.mean(np.abs(sanitized_full_csi), axis=0)
        mean_sanitized_phase = np.mean(np.angle(sanitized_full_csi), axis=0)
        std_sanitized_phase = np.std(np.angle(sanitized_full_csi), axis=0)
        
        stats = {
            'raw_intervals': raw_intervals,
            'prep_intervals': prep_intervals,
            'mean_raw_amp': mean_raw_amp,
            'mean_sanit_amp': mean_sanitized_amp,
            'mean_unwrapped_phase': mean_unwrapped_phase,
            'mean_sanit_phase': mean_sanitized_phase,
            'std_sanit_phase': std_sanitized_phase
        }
        return basename, True, "Success", stats
        
    except Exception as e:
        return basename, False, str(e), None

# --- ASYNC ORCHESTRATOR ---
async def run_visualization(raw_dir=RAW_DIR, prep_dir=PREPROCESSED_DIR, sanit_dir=SANITIZATION_DIR, plots_dir=PLOTS_DIR, json_dir=JSON_LOG_DIR, hist_bins=HIST_BINS, hist_range=HIST_RANGE, run_id=None):
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name = run_id.split('_')[0]
    else:
        dir_name = run_id
        
    plots_dir = os.path.join(plots_dir, dir_name)
    json_dir = os.path.join(json_dir, dir_name)
    ensure_dir(plots_dir)
    ensure_dir(json_dir)
    
    all_files = glob.glob(os.path.join(raw_dir, "*.csv"))
    total_files = len(all_files)
    
    if total_files == 0:
        print(f"No CSV files found in {raw_dir}")
        return
        
    print(f"[Visualization] Extracting statistics for {total_files} files...")
    start_time = time.time()
    
    loop = asyncio.get_running_loop()
    results = []
    
    with ProcessPoolExecutor() as pool:
        func = partial(extract_stats_single, prep_dir=prep_dir, sanit_dir=sanit_dir)
        futures = [loop.run_in_executor(pool, func, fp) for fp in all_files]
        
        for idx, f in enumerate(asyncio.as_completed(futures)):
            basename, success, msg, stats = await f
            results.append((basename, success, msg, stats))
            if (idx + 1) % 50 == 0:
                print(f"[Visualization] Processed {idx + 1}/{total_files} files...")

    end_time = time.time()
    success_count = sum(1 for r in results if r[1])
    fail_count = total_files - success_count
    
    # --- AGGREGATE MATRICES BY CLASS ---
    classes = ['big', 'small']
    aggs = {
        cls: {
            'raw_int': [], 'prep_int': [],
            'raw_amp': [], 'sanit_amp': [],
            'unwrap_ph': [], 'sanit_ph': []
        } for cls in classes
    }
    
    any_data = False
    for res in results:
        if res[1]:
            basename = res[0].lower()
            stats = res[3]
            
            # Determine class
            s_class = None
            if 'big' in basename:
                s_class = 'big'
            elif 'small' in basename:
                s_class = 'small'
                
            if s_class:
                any_data = True
                aggs[s_class]['raw_int'].extend(stats['raw_intervals'])
                aggs[s_class]['prep_int'].extend(stats['prep_intervals'])
                aggs[s_class]['raw_amp'].append(stats['mean_raw_amp'])
                aggs[s_class]['sanit_amp'].append(stats['mean_sanit_amp'])
                aggs[s_class]['unwrap_ph'].append(stats['mean_unwrapped_phase'])
                aggs[s_class]['sanit_ph'].append(stats['mean_sanit_phase'])

    # --- COMPUTE VISUALIZED METRICS ---
    visualized_metrics = {}
    if any_data:
        for cls in classes:
            metrics_cls = {}
            raw_int = aggs[cls]['raw_int']
            prep_int = aggs[cls]['prep_int']
            if len(raw_int) > 0:
                metrics_cls['raw_interval_mean_ms'] = float(np.mean(raw_int))
                metrics_cls['raw_interval_std_ms'] = float(np.std(raw_int))
            if len(prep_int) > 0:
                metrics_cls['prep_interval_mean_ms'] = float(np.mean(prep_int))
                metrics_cls['prep_interval_std_ms'] = float(np.std(prep_int))
                
            r_amps, s_amps = aggs[cls]['raw_amp'], aggs[cls]['sanit_amp']
            if r_amps and s_amps:
                c_len = len(r_amps[0])
                v_r = [a for a in r_amps if len(a) == c_len]
                v_s = [a for a in s_amps if len(a) == c_len]
                if v_r and v_s:
                    metrics_cls['raw_amplitude_avg'] = np.mean(v_r, axis=0).tolist()
                    metrics_cls['sanitized_amplitude_avg'] = np.mean(v_s, axis=0).tolist()
                    
            u_phs, s_phs = aggs[cls]['unwrap_ph'], aggs[cls]['sanit_ph']
            if u_phs and s_phs:
                c_len = len(u_phs[0])
                v_u = [a for a in u_phs if len(a) == c_len]
                v_s = [a for a in s_phs if len(a) == c_len]
                if v_u and v_s:
                    metrics_cls['unwrapped_phase_avg'] = np.mean(v_u, axis=0).tolist()
                    metrics_cls['sanitized_phase_avg'] = np.mean(v_s, axis=0).tolist()
                    
            if metrics_cls:
                visualized_metrics[cls] = metrics_cls
                
    # --- SAVE TO COMMON JSON LOG ---
    json_path = os.path.join(json_dir, f"{run_id}_visualization_log.json")
    
    log_data = {
        "execution_timestamp": datetime.now().isoformat(),
        "pipeline_stage": "visualization",
        "summary": {
            "total_files": total_files,
            "success": success_count,
            "failed": fail_count,
            "execution_time_sec": round(end_time - start_time, 2)
        },
        "visualized_metrics": visualized_metrics,
        "failures": [{"file": r[0], "error": r[2]} for r in results if not r[1]]
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=4)
                 
    print(f"[Visualization] Statistics Complete. Logs saved to {json_path}")
    
    if not any_data:
        print("No successful big/small classified data to plot.")
        return
        
    print("[Visualization] Rendering Global Class-specific Plots...")
    
    # 1. Time Gap Histogram (Raw vs Preprocessed in 2x2 Subplots)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Time Gap Histogram", fontsize=16)
    
    for i, cls in enumerate(classes):
        raw = aggs[cls]['raw_int']
        prep = aggs[cls]['prep_int']
        
        # Row 0: Raw
        if len(raw) > 0:
            axes[0, i].hist(raw, bins=hist_bins, range=hist_range, color='salmon', alpha=0.7, edgecolor='black')
        axes[0, i].set_title(f"Raw Packet Intervals ({cls.capitalize()})")
        axes[0, i].set_xlabel("Packet Interval (ms)")
        axes[0, i].set_ylabel("Frequency")
        axes[0, i].grid(axis='y', alpha=0.5)
        
        # Row 1: Preprocessed
        if len(prep) > 0:
            axes[1, i].hist(prep, bins=hist_bins, range=hist_range, color='dodgerblue', alpha=0.7, edgecolor='black')
        axes[1, i].set_title(f"Preprocessed Intervals ({cls.capitalize()})")
        axes[1, i].set_xlabel("Packet Interval (ms)")
        axes[1, i].set_ylabel("Frequency")
        axes[1, i].grid(axis='y', alpha=0.5)
        
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "01_time_gap_subplots.png"), dpi=300)
    plt.close()
    
    # 2. Amplitude Distribution (M-shape) in 1x2 Subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Average Subcarrier Amplitude Distribution", fontsize=16)
    
    for i, cls in enumerate(classes):
        raw_amps = aggs[cls]['raw_amp']
        sanit_amps = aggs[cls]['sanit_amp']
        
        if raw_amps and sanit_amps:
            common_length = len(raw_amps[0])
            valid_raw = [a for a in raw_amps if len(a) == common_length]
            valid_sanit = [a for a in sanit_amps if len(a) == common_length]
            
            if valid_raw and valid_sanit:
                avg_raw = np.mean(valid_raw, axis=0)
                avg_sanit = np.mean(valid_sanit, axis=0)
                sub_indices = np.arange(common_length)
                
                axes[i].plot(sub_indices, avg_raw, label="Raw Amp", color='orange', linewidth=2)
                axes[i].plot(sub_indices, avg_sanit, label="Sanitized Amp", color='green', linewidth=2, linestyle='dashed')
                
        axes[i].set_title(f"Amplitude ({cls.capitalize()})")
        axes[i].set_xlabel("Valid Subcarrier Index")
        axes[i].set_ylabel("Amplitude")
        axes[i].legend()
        axes[i].grid(True)
        
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "02_amplitude_distortion_subplots.png"), dpi=300)
    plt.close()
    
    # 3. Phase-Frequency Curve (Unwrapped vs Sanitized in 1x2 Subplots)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Phase-Frequency Curve: Unwrapping & Linear Fit Effect", fontsize=16)
    
    for i, cls in enumerate(classes):
        unwrap_phs = aggs[cls]['unwrap_ph']
        sanit_phs = aggs[cls]['sanit_ph']
        
        if unwrap_phs and sanit_phs:
            common_length = len(unwrap_phs[0])
            valid_unwrap = [a for a in unwrap_phs if len(a) == common_length]
            valid_sanit = [a for a in sanit_phs if len(a) == common_length]
            
            if valid_unwrap and valid_sanit:
                avg_unwrap = np.mean(valid_unwrap, axis=0)
                avg_sanit = np.mean(valid_sanit, axis=0)
                sub_indices = np.arange(common_length)
                
                axes[i].plot(sub_indices, avg_unwrap, label="Unwrapped Phase", color='gray', linewidth=2, linestyle='dotted')
                axes[i].plot(sub_indices, avg_sanit, label="Sanitized Phase", color='purple', linewidth=2)
                
        axes[i].set_title(f"Phase Curve ({cls.capitalize()})")
        axes[i].set_xlabel("Valid Subcarrier Index")
        axes[i].set_ylabel("Phase (Radian)")
        axes[i].legend()
        axes[i].grid(True)
        
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "03_phase_frequency_subplots.png"), dpi=300)
    plt.close()

    print("[Visualization] Outputs saved to data/plots/sanitization/ successfully.")

if __name__ == "__main__":
    asyncio.run(run_visualization())
