import os
import glob
import asyncio
import time
import json
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

import numpy as np
import pandas as pd

# ==========================================
# CONFIGURATION PARAMETERS
# ==========================================
ENABLE_RATIO = False
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PREPROCESSED_DIR = os.path.join(BASE_DIR, "data", "sanitization", "preprocessed")
SANITIZATION_DIR = os.path.join(BASE_DIR, "data", "sanitization", "sanitization")
JSON_LOG_DIR = os.path.join(BASE_DIR, "data", "result", "sanitization", "json")

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

class PhaseSanitizer:
    def __init__(self, enable_ratio=False, template_phase=None):
        self.enable_ratio = enable_ratio
        self.template_phase = template_phase

    def unwrap_and_linear_fit(self, csi_matrix):
        sanitized_matrix = np.zeros_like(csi_matrix, dtype=complex)
        amplitudes = np.abs(csi_matrix)
        phases = np.angle(csi_matrix)
        
        N_packets, N_subcarriers = csi_matrix.shape
        subcarrier_indices = np.arange(N_subcarriers)
        
        r_squared_list = []
        
        for i in range(N_packets):
            unwrapped_phase = np.unwrap(phases[i])
            A = np.vstack([subcarrier_indices, np.ones(N_subcarriers)]).T
            
            # 피팅 및 평가
            m, c = np.linalg.lstsq(A, unwrapped_phase, rcond=None)[0]
            fitted_phase = m * subcarrier_indices + c
            
            # R-squared calculation
            ss_res = np.sum((unwrapped_phase - fitted_phase)**2)
            ss_tot = np.sum((unwrapped_phase - np.mean(unwrapped_phase))**2)
            r2 = 1 - (ss_res / (ss_tot + 1e-10))
            r_squared_list.append(r2)
            
            sanitized_phase = unwrapped_phase - fitted_phase
            if self.template_phase is not None:
                sanitized_phase -= self.template_phase
                
            sanitized_matrix[i] = amplitudes[i] * np.exp(1j * sanitized_phase)
            
        return sanitized_matrix, float(np.mean(r_squared_list))
        
    def compute_csi_ratio(self, csi_matrix_ant1, csi_matrix_ant2):
        if not self.enable_ratio:
            return csi_matrix_ant1
        return csi_matrix_ant1 / (csi_matrix_ant2 + 1e-10)

class AmplitudeSanitizer:
    def __init__(self, template_amplitude=None):
        self.template_amplitude = template_amplitude

    def correct_nonlinear_distortion(self, csi_matrix):
        if self.template_amplitude is None:
            return csi_matrix
        amplitudes = np.abs(csi_matrix)
        phases = np.angle(csi_matrix)
        corrected_amplitudes = amplitudes / (self.template_amplitude + 1e-10)
        return corrected_amplitudes * np.exp(1j * phases)


def extract_quality_metrics(original_matrix, sanitized_matrix, mean_r_squared):
    """정제 파이프라인 이후 품질 분석"""
    sanit_amp = np.abs(sanitized_matrix)
    sanit_phase = np.angle(sanitized_matrix)
    
    # 1. 진폭 변동률 (Coefficient of Variation)
    mean_amp = np.mean(sanit_amp, axis=0) + 1e-10
    std_amp = np.std(sanit_amp, axis=0)
    cv_mean = float(np.mean(std_amp / mean_amp))
    
    # 2. 위상 안정도 (Phase Stability) - 분산이 적을수록 안정적 
    phase_std_mean = float(np.mean(np.std(sanit_phase, axis=0)))
    phase_stability_score = max(0.0, 100.0 - phase_std_mean * 10.0) # 대략적인 환산 (기준용)
    
    return {
        "mean_r_squared": float(mean_r_squared),
        "phase_stability_score": round(phase_stability_score, 2),
        "amplitude_cv_mean": round(cv_mean, 4)
    }

# --- WORKER FUNCTION ---
def process_single_npz(file_path, out_dir, enable_ratio, template_phase=None, template_amplitude=None):
    basename = os.path.basename(file_path)
    phase_sanitizer = PhaseSanitizer(enable_ratio=enable_ratio, template_phase=template_phase)
    amplitude_sanitizer = AmplitudeSanitizer(template_amplitude=template_amplitude)
    
    try:
        data = np.load(file_path)
        prep_t = data['time']
        prep_csi = data['csi']
        
        # Phase Sanitization 
        sanitized_phase_csi, r2 = phase_sanitizer.unwrap_and_linear_fit(prep_csi)
        
        # Amplitude Sanitization
        sanitized_full_csi = amplitude_sanitizer.correct_nonlinear_distortion(sanitized_phase_csi)
        
        # Extract Quality JSON 
        metrics = extract_quality_metrics(prep_csi, sanitized_full_csi, r2)
        
        # Save Sanitized NPZ
        sanit_path = os.path.join(out_dir, basename)
        np.savez(sanit_path, time=prep_t, csi=sanitized_full_csi)
        
        return basename, True, "Success", metrics
    except Exception as e:
        return basename, False, str(e), {}


# --- ASYNC ORCHESTRATOR ---
async def run_sanitization(prep_dir=PREPROCESSED_DIR, sanit_dir=SANITIZATION_DIR, json_dir=JSON_LOG_DIR, enable_ratio=ENABLE_RATIO, run_id=None, template_csv=None):
    in_dir = prep_dir
    out_dir = sanit_dir
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name = run_id.split('_')[0]
    else:
        dir_name = run_id
        
    json_dir = os.path.join(json_dir, dir_name)
    ensure_dir(out_dir)
    ensure_dir(json_dir)
    
    template_phase, template_amplitude = None, None
    algorithm_amp_info = "Disabled (No Template)"
    algorithm_phase_info = "Least-Squares Linear Fitting Only"
    
    if template_csv and os.path.exists(template_csv):
        print(f"[Sanitization] Loading hardware calibration template from: {template_csv}")
        df_temp = pd.read_csv(template_csv)
        if 'phase' in df_temp.columns and 'amplitude' in df_temp.columns:
            template_phase = df_temp['phase'].values
            template_amplitude = df_temp['amplitude'].values
            algorithm_amp_info = f"Template Division ({os.path.basename(template_csv)})"
            algorithm_phase_info = f"Linear Fit + S-Shape Subtraction ({os.path.basename(template_csv)})"
        else:
            print("[Sanitization] WARNING: Invalid template format. 'phase' and 'amplitude' columns required.")
    else:
        print("[Sanitization] No hardware template specified or found. Skipping nonlinear template correction.")
    
    all_files = glob.glob(os.path.join(in_dir, "**", "*.npz"), recursive=True)
    total_files = len(all_files)
    
    if total_files == 0:
        print(f"No NPZ files found in {in_dir}")
        return
        
    print(f"[Sanitization] Starting batch processing for {total_files} files...")
    start_time = time.time()
    
    loop = asyncio.get_running_loop()
    results = []
    
    with ProcessPoolExecutor() as pool:
        func = partial(process_single_npz, out_dir=out_dir, enable_ratio=enable_ratio, template_phase=template_phase, template_amplitude=template_amplitude)
        futures = [loop.run_in_executor(pool, func, fp) for fp in all_files]
        
        for idx, f in enumerate(asyncio.as_completed(futures)):
            basename, success, msg, metrics = await f
            results.append((basename, success, msg, metrics))
            if (idx + 1) % 50 == 0:
                print(f"[Sanitization] Processed {idx + 1}/{total_files} files...")

    end_time = time.time()
    success_count = sum(1 for r in results if r[1])
    fail_count = total_files - success_count
    
    # ------------------
    # SAVE GLOBAL JSON
    # ------------------
    json_path = os.path.join(json_dir, f"{run_id}_sanitization_log.json")
    
    # Calculate average metrics across all successful files
    avg_r2, avg_phase_stab, avg_cv = 0, 0, 0
    if success_count > 0:
        avg_r2 = np.mean([r[3].get('mean_r_squared', 0) for r in results if r[1]])
        avg_phase_stab = np.mean([r[3].get('phase_stability_score', 0) for r in results if r[1]])
        avg_cv = np.mean([r[3].get('amplitude_cv_mean', 0) for r in results if r[1]])
    
    log_data = {
        "execution_timestamp": datetime.now().isoformat(),
        "pipeline_stage": "sanitization",
        "sanitization_quality_avg": {
            "mean_r_squared": float(avg_r2),
            "phase_stability_score": float(avg_phase_stab),
            "amplitude_cv_mean": float(avg_cv)
        },
        "algorithm_info": {
            "phase": algorithm_phase_info,
            "amplitude": algorithm_amp_info,
            "outlier_filter": "Disabled"
        },
        "summary": {
            "total_files": total_files,
            "success": success_count,
            "failed": fail_count,
            "execution_time_sec": round(end_time - start_time, 2)
        },
        "failures": [{"file": r[0], "error": r[2]} for r in results if not r[1]]
    }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=4, ensure_ascii=False)
        
    print(f"[Sanitization] Complete! Success: {success_count}, Failed: {fail_count}. Logs saved to {json_path}")

if __name__ == "__main__":
    asyncio.run(run_sanitization())
