"""
make_template.py
================
MATLAB `set_template` 함수를 완벽히 이식한 파이썬 구현체.

동축 케이블(Coaxial Cable)로 수집된 CSI 데이터에서
하드웨어 고유의 비선형 왜곡 템플릿(M-Shape 진폭, S-Shape 위상)을 추출합니다.

Usage (캘리브레이션 데이터가 있을 때):
    python make_template.py --calib_dir ../data/calibration --out_path ../data/sanitization/template/template.csv

Usage (캘리브레이션 데이터가 없을 때):
    -- main_pipeline.py에서 template_csv 인자를 비워두면 자동 건너뜀 --
"""
import os
import sys
import glob
import argparse
import ast

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==========================================
# 802.11n 40MHz 유효 서브캐리어 설정
# 가드밴드 제외: 2~58, 70~126 → 114개
# 파일럿 제거: 11, 25, 53, 75, 103, 117
# 최종 유효 데이터 서브캐리어: 108개
# ==========================================
_NZ = list(range(2, 59)) + list(range(70, 127))
_PILOTS = [11, 25, 53, 75, 103, 117]
VALID_INDICES = [x for x in _NZ if x not in _PILOTS]  # 108개


def extract_valid_108(csi_str: str) -> np.ndarray:
    """
    CSV의 'data' 컬럼 문자열에서 108개의 유효 복소 CSI 서브캐리어를 추출합니다.
    """
    csi_array = np.array(ast.literal_eval(csi_str), dtype=np.float32)
    if len(csi_array) != 384:
        raise ValueError(f"Expected 384 elements, got {len(csi_array)}")
    ht_ltf_flat = csi_array[128:]
    complex_ht_ltf = ht_ltf_flat[::2] + 1j * ht_ltf_flat[1::2]
    return complex_ht_ltf[VALID_INDICES]


def set_template(csi_calib: np.ndarray, linear_interval: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    MATLAB set_template() 함수의 완전한 파이썬 이식.

    Parameters
    ----------
    csi_calib : np.ndarray, shape [T, S]
        동축 케이블에서 수집된 복소 CSI 행렬 (T=패킷 수, S=108 서브캐리어)
    linear_interval : np.ndarray
        비선형 왜곡이 없는(직선에 가까운) 부반송파 인덱스 배열.
        이 구간만을 참조해 선형 피팅 기울기를 계산합니다.

    Returns
    -------
    csi_amp_template : np.ndarray, shape [S]
        M-Shape 진폭 템플릿 (정규화됨, 평균=1.0 기준)
    csi_phase_template : np.ndarray, shape [S]
        S-Shape 위상 에러 템플릿 (비선형 잔류 위상)
        linear_interval 구간은 0으로 클램프됨.
    """
    N_packets, subcarrier_num = csi_calib.shape

    # ── 1. Amplitude Template ────────────────────────────────────────────────
    # MATLAB: csi_amp_template = mean(csi_amp ./ mean(csi_amp, 2), 1)
    csi_amp = np.abs(csi_calib)                                          # [T, S]
    mean_amp_per_packet = np.mean(csi_amp, axis=1, keepdims=True)        # [T, 1]
    csi_amp_norm = csi_amp / (mean_amp_per_packet + 1e-10)               # [T, S]
    csi_amp_template = np.mean(csi_amp_norm, axis=0)                     # [S]

    # ── 2. Phase (S-Shape Nonlinear Error) Template ──────────────────────────
    # MATLAB: csi_phase = unwrap(angle(csi_calib), [], 2)   → axis=1 (subcarrier axis)
    csi_phase = np.unwrap(np.angle(csi_calib), axis=1)                  # [T, S]
    nonlinear_phase_error = np.zeros_like(csi_phase)                     # [T, S]

    x_fit = linear_interval.astype(float)
    x_all = np.arange(subcarrier_num, dtype=float)

    for p in range(N_packets):
        y_fit = csi_phase[p, linear_interval]
        # MATLAB: fit(linear_interval, squeeze(csi_phase(p,:)), 'poly1')
        m, c = np.polyfit(x_fit, y_fit, 1)
        linear_model = m * x_all + c                                     # [S]
        nonlinear_phase_error[p, :] = csi_phase[p, :] - linear_model    # [S]

    csi_phase_template = np.mean(nonlinear_phase_error, axis=0)         # [S]
    # MATLAB: csi_phase_template(1, linear_interval, :, :) = 0
    csi_phase_template[linear_interval] = 0.0

    return csi_amp_template, csi_phase_template


def run_make_template(calib_dir: str, out_path: str, linear_interval: np.ndarray, plot: bool = True):
    """
    calib_dir 안의 모든 캘리브레이션 CSV 파일을 읽어 템플릿을 생성하고 저장합니다.

    캘리브레이션 데이터가 없으면 종료 메시지를 출력하고 반환합니다.
    (파이프라인에서는 template_csv 인자를 None/빈 값으로 두면 건너뜁니다.)
    """
    csv_files = glob.glob(os.path.join(calib_dir, "*.csv"))

    if not csv_files:
        print(f"[make_template] ⚠ No calibration CSV files found in: {calib_dir}")
        print("[make_template] → Skipping template generation.")
        print("[make_template] → When calibration data is available, place CSVs in the above directory and re-run.")
        return

    print(f"[make_template] Loading {len(csv_files)} calibration file(s)...")
    all_csi = []

    for fpath in csv_files:
        df = pd.read_csv(fpath)
        if 'data' not in df.columns:
            print(f"[make_template] WARNING: 'data' column missing in {os.path.basename(fpath)}, skipping.")
            continue
        for csi_str in df['data']:
            try:
                all_csi.append(extract_valid_108(csi_str))
            except Exception as e:
                pass  # noisy packet, skip silently

    if len(all_csi) == 0:
        print("[make_template] ERROR: No valid CSI packets could be parsed. Check input format.")
        return

    csi_calib = np.array(all_csi)    # [T, 108]
    print(f"[make_template] Total valid packets: {csi_calib.shape[0]}, Subcarriers: {csi_calib.shape[1]}")

    # Generate template using MATLAB-equivalent logic
    csi_amp_template, csi_phase_template = set_template(csi_calib, linear_interval)

    # Save to CSV
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    template_df = pd.DataFrame({
        "amplitude": csi_amp_template,
        "phase": csi_phase_template
    })
    template_df.to_csv(out_path, index=False)
    print(f"[make_template] ✓ Template saved to: {out_path}")

    # Optional: save a diagnostic plot
    if plot:
        plot_path = os.path.splitext(out_path)[0] + "_preview.png"
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(csi_amp_template, color="#4A90D9", linewidth=1.5)
        axes[0].axhline(1.0, color='gray', linestyle='--', linewidth=0.8)
        axes[0].set_title("Amplitude Template (M-Shape)")
        axes[0].set_xlabel("Subcarrier Index (0~107)")
        axes[0].set_ylabel("Normalized Amplitude")
        axes[0].grid(alpha=0.3)

        axes[1].plot(csi_phase_template, color="#E85D04", linewidth=1.5)
        axes[1].axhspan(linear_interval[0], linear_interval[-1],
                        alpha=0.1, color='green', label=f"Linear Interval [{linear_interval[0]}:{linear_interval[-1]}]")
        axes[1].set_title("Phase Template (S-Shape Nonlinear Error)")
        axes[1].set_xlabel("Subcarrier Index (0~107)")
        axes[1].set_ylabel("Phase Error (rad)")
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.3)

        plt.suptitle("Hardware Calibration Template (Coaxial Cable)", fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(plot_path, dpi=120)
        plt.close()
        print(f"[make_template] ✓ Preview plot saved to: {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Hardware Calibration Template from Coaxial CSI Data (MATLAB set_template port)"
    )
    parser.add_argument(
        "--calib_dir", type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "data", "calibration"),
        help="Directory containing raw coaxial cable CSI .csv files"
    )
    parser.add_argument(
        "--out_path", type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "data", "sanitization", "template", "template.csv"),
        help="Output path for the generated template.csv"
    )
    parser.add_argument(
        "--linear_start", type=int, default=30,
        help="Start index of linear (non-distorted) subcarrier interval (default: 30)"
    )
    parser.add_argument(
        "--linear_end", type=int, default=78,
        help="End index (exclusive) of linear subcarrier interval (default: 78)"
    )
    parser.add_argument(
        "--no_plot", action="store_true",
        help="Disable diagnostic preview plot generation"
    )
    args = parser.parse_args()

    linear_interval = np.arange(args.linear_start, args.linear_end)

    run_make_template(
        calib_dir=args.calib_dir,
        out_path=args.out_path,
        linear_interval=linear_interval,
        plot=not args.no_plot
    )
