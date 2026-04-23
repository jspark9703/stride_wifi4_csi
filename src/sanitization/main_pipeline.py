import argparse
import asyncio
import time
import os
import numpy as np
from datetime import datetime

import scripts.preprocess as preprocess
import scripts.sanitization as sanitization
import scripts.visualization as visualization
import scripts.make_template as make_template

def prepare_directories():
    """Ensure all required input, intermediate, and output directories exist."""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    dirs = {
        "raw_dir": os.path.join(base_dir, "data", "raw", "260406" ),
        "prep_dir": os.path.join(base_dir, "data", "sanitization", "preprocessed", "260406"),
        "sanit_dir": os.path.join(base_dir, "data", "sanitization", "sanitization", "260406"),
        "json_dir": os.path.join(base_dir, "sanitization", "result", "json"),
        "plots_dir": os.path.join(base_dir, "sanitization", "result", "plots"),
        "template_dir": os.path.join(base_dir, "sanitization", "template")
    }
    
    for key, d in dirs.items():
        if not os.path.exists(d):
            os.makedirs(d)
            
    return dirs


async def run_all(args, dirs):
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("="*50)
    print("     WIFI CSI FULL PIPELINE ORCHESTRATOR     ")
    print("="*50)
    print(f"[*] RUN ID (Timestamp): {run_id}")
    
    start_time = time.time()
    
    print(f"\n[STEP 0] Template Generation (Coaxial Calibration)")
    calib_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", args.calib_dir))
    linear_interval = np.arange(args.linear_start, args.linear_end)
    make_template.run_make_template(
        calib_dir=calib_dir,
        out_path=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", args.template_csv)),
        linear_interval=linear_interval,
        plot=True
    )

    print(f"\n[STEP 1] Preprocessing (Target FS: {args.target_fs}Hz, Max Gap: {args.max_gap_ms}ms, "
          f"Hampel: {args.hampel_enabled}, LowPass: {args.lowpass_enabled})")
    await preprocess.run_preprocessing(
        target_fs=args.target_fs,
        max_gap_ms=args.max_gap_ms,
        hampel_enabled=args.hampel_enabled,
        lowpass_enabled=args.lowpass_enabled,
        raw_dir=dirs["raw_dir"],
        out_dir=dirs["prep_dir"],
        json_dir=dirs["json_dir"],
        run_id=run_id
    )
    
    print(f"\n[STEP 2] Sanitization (Phase Fitting & Amplitude Scaling, CSI Ratio: {args.enable_ratio})")
    
    template_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", args.template_csv))
    await sanitization.run_sanitization(
        enable_ratio=args.enable_ratio,
        prep_dir=dirs["prep_dir"],
        sanit_dir=dirs["sanit_dir"],
        json_dir=dirs["json_dir"],
        run_id=run_id,
        template_csv=template_path
    )
    
    hist_range = (0, args.hist_range_max)
    print(f"\n[STEP 3] Visualization (JSON Logging, Constellation, Phase/Amp Curve)")
    await visualization.run_visualization(
        raw_dir=dirs["raw_dir"],
        prep_dir=dirs["prep_dir"],
        sanit_dir=dirs["sanit_dir"],
        plots_dir=dirs["plots_dir"],
        json_dir=dirs["json_dir"],
        hist_bins=args.hist_bins, 
        hist_range=hist_range,
        run_id=run_id
    )
    
    end_time = time.time()
    
    print("="*50)
    print(f"PIPELINE COMPLETED in {end_time - start_time:.2f} seconds.")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WIFI CSI Data Preprocessing, Sanitization, and Visualization Pipeline")
    
    # Preprocessing Parameters
    parser.add_argument("--target_fs", type=int, default=100, help="Target sampling frequency for interpolation (Hz)")
    parser.add_argument("--max_gap_ms", type=float, default=20.0, help="Maximum gap threshold for interpolation (ms)")
    parser.add_argument("--hampel_enabled", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable Hampel outlier filter (default: enabled). Use --no-hampel_enabled to disable.")
    parser.add_argument("--lowpass_enabled", action=argparse.BooleanOptionalAction, default=False,
                        help="Enable Butterworth low-pass filter (default: disabled). Use --lowpass_enabled to enable.")
    
    # Sanitization / Calibration Parameters
    parser.add_argument("--enable_ratio", action="store_true", help="Enable CSI Ratio (for multiple antennas)")
    parser.add_argument("--template_csv", type=str, default="sanitization/template/template.csv", help="Path to coaxial hardware template CSV (relative to project root)")
    parser.add_argument("--calib_dir", type=str, default="calibration", help="Directory containing raw coaxial calibration CSVs")
    parser.add_argument("--linear_start", type=int, default=30, help="Start index of linear subcarrier interval for template fitting")
    parser.add_argument("--linear_end", type=int, default=78, help="End index (exclusive) of linear subcarrier interval")

    # Visualization Parameters
    parser.add_argument("--hist_bins", type=int, default=100, help="Number of bins for histogram visualization")
    parser.add_argument("--hist_range_max", type=int, default=40, help="Max range component of histogram X-axis (ms)")

    args = parser.parse_args()

    # Define centralized directories directly from main pipeline to avoid mapping bugs
    dirs = prepare_directories()

    # Windows 환경에서 asyncio / multiprocessing 충돌을 막기 위한 안전 진입점
    asyncio.run(run_all(args, dirs))
