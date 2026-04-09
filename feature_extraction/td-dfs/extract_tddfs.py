"""
TD-DFS (Time-Domain Doppler Frequency Spectrum) Feature Extractor
==================================================================
입력  : data/sanitization/sanitization/*.npz  (keys: 'time', 'csi')
        csi : (N_packets, 108) complex64,  fs = 100 Hz
출력  : data/feature_extraction/td-dfs/*.npz

알고리즘 개요
--------------
STFT의 긴 윈도우 크기 한계를 극복하기 위해, 시간 영역(Time-Domain)에서
두 CSI 샘플 간의 차분(Difference)만으로 순간 도플러 속도를 추출합니다.
진폭(Amplitude)과 위상(Phase) 정보를 모두 사용하여 방향과 절대 속도를 역산합니다.

파이프라인
-----------
STEP 1. 복소 CSI 로드  →  (N_packets, 108) complex64

STEP 2. TD-CSI 생성 (차분 — 정적 성분 완벽 제거)
        TD(t) = H(t + Δt) − H(t)
        ● 정적 반사파(벽, 가구): H_static(t) ≈ const
          → H_static(t+Δt) − H_static(t) ≈ 0  (완전 상쇄)
        ● 동적 성분(신체 움직임): 시간 변화를 가지므로 TD에 그대로 남음

STEP 3. 최적 Δt 탐색 (위상 노이즈 분산 최소화)
        평가 기준: σ²(Δt) = Var[ ∠( subcarrier_mean(TD(t, Δt)) ) ] over t
        ● Δt 너무 작음 → |TD| ≈ 0, 위상 추정이 노이즈에 지배됨
        ● Δt 너무 큼  → 위상이 ±π 초과, 언래핑 오류(aliasing) 발생
        → 위상 분산이 최소인 Δt 선택

STEP 4. 위상 기반 방향 추정
        v_sign(t) = sign( Im[ mean_subcarrier(TD(t)) ] )
        ● 양수(+): 접근 방향  /  음수(-): 후퇴 방향

STEP 5. 진폭 기반 절대 도플러 속도 역산 (물리 수식)
        물리 모델:
          H_dyn(t) = A · exp(j·2π·fd·t/fs)
          TD(t) = H_dyn(t+Δt) − H_dyn(t)
               = A · exp(j·2π·fd·t/fs) · (exp(j·2π·fd·Δt/fs) − 1)
          |TD(t)| = 2A · |sin(π·fd·Δt/fs)|

        도플러 주파수 역산:
          fd(t) = arcsin( |TD(t)| / (2·|H_ref(t)|) ) × fs / (π·Δt)  [Hz]

        도플러 속도 역산:
          v(t) = fd(t) × λ / 2  [m/s]   (단방향 RX, λ = c/fc)
          → cm/s 단위로 출력

출력 NPZ 키
-----------
  velocity  : (N_packets - Δt,) float32  — 순간 도플러 속도 [cm/s]
  fdop      : (N_packets - Δt,) float32  — 도플러 주파수 [Hz]
  delta_t   : scalar int                 — 사용된 최적 Δt [샘플]
  label     : scalar str                 — 'big' / 'small'
  subject   : scalar str
  filename  : scalar str

추가 선택 저장 (--save_npy / --save_csv):
  <stem>_tddfs.npy  — 속도 1D 배열 (.npy)
  <stem>_tddfs.csv  — 속도 1D 시계열 (.csv)

파라미터
---------
  --delta_t_min  : Δt 탐색 최솟값 [샘플]  (기본: 1)
  --delta_t_max  : Δt 탐색 최댓값 [샘플]  (기본: 10)
  --fc_hz        : WiFi 중심 주파수 [Hz]   (기본: 5.18e9)
  --sanit_dir    : 입력 NPZ 디렉터리
  --out_dir      : 출력 NPZ 디렉터리
  --log_dir      : JSON 로그 디렉터리
  --save_npy     : .npy 파일 별도 저장
  --save_csv     : .csv 파일 별도 저장

실행 예시
---------
  python extract_tddfs.py
  python extract_tddfs.py --delta_t_min 1 --delta_t_max 15
  python extract_tddfs.py --save_npy --save_csv
"""

import os
import glob
import argparse
import time
import json
from datetime import datetime

import numpy as np

# ─────────────────────────────────────────────
# 경로 및 물리 상수
# ─────────────────────────────────────────────
BASE_DIR          = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_SANIT_DIR = os.path.join(BASE_DIR, "data", "sanitization", "sanitization")
DEFAULT_OUT_DIR   = os.path.join(BASE_DIR, "data", "feature_extraction", "td-dfs")
DEFAULT_LOG_DIR   = os.path.join(BASE_DIR, "data", "result", "feature_extraction", "td-dfs")
DEFAULT_NPY_DIR   = os.path.join(BASE_DIR, "data", "feature_extraction", "td-dfs_npy")
DEFAULT_CSV_DIR   = os.path.join(BASE_DIR, "data", "feature_extraction", "td-dfs_csv")

FS    = 100    # 샘플링 주파수 [Hz]
C_MPS = 3e8   # 광속 [m/s]


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _parse_label_subject(filename: str):
    """파일명 패턴: csi_YYMMDD_HHMMSS_<subject>_<label>.npz"""
    stem      = os.path.splitext(os.path.basename(filename))[0]
    parts     = stem.split("_")
    label_raw = parts[-1].lower()
    if "big" in label_raw:
        label = "big"
    elif "small" in label_raw or "smal" in label_raw:
        label = "small"
    else:
        label = "unknown"
    subject = parts[-2] if len(parts) >= 2 else "unknown"
    return label, subject


# ══════════════════════════════════════════════════════════════
#  STEP 1: 복소 CSI 로드
# ══════════════════════════════════════════════════════════════

def load_csi(npz_path: str) -> np.ndarray:
    """
    Returns
    -------
    csi : (N_packets, N_subcarriers) complex64
    """
    data = np.load(npz_path)
    return data["csi"].astype(np.complex64)


# ══════════════════════════════════════════════════════════════
#  STEP 2: TD-CSI 생성 (시간 영역 차분)
# ══════════════════════════════════════════════════════════════

def compute_td_csi(csi: np.ndarray, delta_t: int) -> np.ndarray:
    """
    시간 영역 차분으로 정적 배경 성분을 완전히 제거.
      TD(t) = H(t + Δt) − H(t)

    Returns
    -------
    td_csi : (N - delta_t, NS) complex64
    """
    return (csi[delta_t:] - csi[:-delta_t]).astype(np.complex64)


# ══════════════════════════════════════════════════════════════
#  STEP 3: 최적 Δt 탐색
# ══════════════════════════════════════════════════════════════

def find_optimal_delta_t(
    csi: np.ndarray,
    delta_t_min: int = 1,
    delta_t_max: int = 10,
) -> int:
    """
    위상 노이즈 분산  σ²(Δt) = Var[∠ mean_s(TD(t, Δt))]  최소화 Δt 탐색.

    Returns
    -------
    best_dt : int
    """
    best_dt  = delta_t_min
    best_var = np.inf

    for dt in range(delta_t_min, delta_t_max + 1):
        td           = compute_td_csi(csi, dt)
        z            = td.mean(axis=1)
        phase_series = np.angle(z)
        variance     = float(np.var(phase_series))
        if variance < best_var:
            best_var = variance
            best_dt  = dt

    return best_dt


# ══════════════════════════════════════════════════════════════
#  STEP 4 & 5: 방향 추정 + 진폭 기반 속도 역산
# ══════════════════════════════════════════════════════════════

def estimate_velocity(
    csi: np.ndarray,
    delta_t: int,
    fc_hz: float,
) -> tuple:
    """
    STEP 4 — 위상 기반 방향 추정:
      direction(t) = sign( Im[ subcarrier_mean(TD(t)) ] )

    STEP 5 — 진폭 기반 절대 속도 역산:
      fd(t)  = arcsin( |TD| / (2|H_ref|) ) × fs / (π Δt)  [Hz]
      v(t)   = fd(t) × λ / 2  [m/s]  → [cm/s]

    Returns
    -------
    velocity : (N - delta_t,) float32  [cm/s]
    fdop     : (N - delta_t,) float32  [Hz]
    """
    lam_m = C_MPS / fc_hz

    td = compute_td_csi(csi, delta_t)   # (M, NS)
    M  = td.shape[0]

    A_ref   = np.abs(csi[:M]).mean(axis=1)         # (M,)  참조 진폭
    TD_mean = td.mean(axis=1)                       # (M,)  서브캐리어 평균
    TD_amp  = np.abs(TD_mean)                       # (M,)

    # 방향 (허수부 부호)
    direction          = np.sign(np.sin(np.angle(TD_mean)))
    direction[direction == 0] = 1.0

    # arcsin 역산 (수치 안정 클리핑)
    arg    = np.clip(TD_amp / (2.0 * A_ref + 1e-10), 0.0, 1.0)
    fd_abs = np.arcsin(arg) * FS / (np.pi * delta_t)
    fdop   = (fd_abs * direction).astype(np.float32)

    velocity = (fdop * lam_m / 2.0 * 100.0).astype(np.float32)
    return velocity, fdop


# ══════════════════════════════════════════════════════════════
#  파일 하나 처리
# ══════════════════════════════════════════════════════════════

def extract_features_single(
    npz_path: str,
    delta_t_min: int = 1,
    delta_t_max: int = 10,
    fc_hz: float     = 5.18e9,
) -> dict:
    """
    단일 NPZ → TD-DFS 특징 추출.

    Returns
    -------
    {'velocity': (M,) float32, 'fdop': (M,) float32, 'delta_t': int}
    """
    csi     = load_csi(npz_path)
    best_dt = find_optimal_delta_t(csi, delta_t_min, delta_t_max)
    velocity, fdop = estimate_velocity(csi, best_dt, fc_hz)
    return {"velocity": velocity, "fdop": fdop, "delta_t": best_dt}


# ══════════════════════════════════════════════════════════════
#  저장 헬퍼 (선택)
# ══════════════════════════════════════════════════════════════

def _save_npy(velocity: np.ndarray, fname_stem: str, npy_dir: str):
    """TD-DFS 1D 속도 시계열을 .npy로 별도 저장 (Stage 4: 포맷 분리)."""
    ensure_dir(npy_dir)
    np.save(os.path.join(npy_dir, f"{fname_stem}_tddfs.npy"), velocity)


def _save_csv(velocity: np.ndarray, fdop: np.ndarray,
              delta_t: int, fname_stem: str, csv_dir: str):
    """TD-DFS 1D 시계열을 .csv로 별도 저장 (Stage 4: 포맷 분리)."""
    try:
        import csv
        ensure_dir(csv_dir)
        out_path = os.path.join(csv_dir, f"{fname_stem}_tddfs.csv")
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([f"# delta_t={delta_t} samples"])
            writer.writerow(["time_index", "velocity_cms", "fdop_hz"])
            for idx, (v, fd) in enumerate(zip(velocity, fdop)):
                writer.writerow([idx, f"{v:.6f}", f"{fd:.6f}"])
    except Exception as e:
        print(f"  [WARN] CSV 저장 실패: {e}")


# ══════════════════════════════════════════════════════════════
#  배치 처리
# ══════════════════════════════════════════════════════════════

def run_tddfs_extraction(
    sanit_dir:   str   = DEFAULT_SANIT_DIR,
    out_dir:     str   = DEFAULT_OUT_DIR,
    log_dir:     str   = DEFAULT_LOG_DIR,
    delta_t_min: int   = 1,
    delta_t_max: int   = 10,
    fc_hz:       float = 5.18e9,
    save_npy:    bool  = False,
    npy_dir:     str   = DEFAULT_NPY_DIR,
    save_csv:    bool  = False,
    csv_dir:     str   = DEFAULT_CSV_DIR,
):
    ensure_dir(out_dir)
    ensure_dir(log_dir)

    run_id     = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_subdir = os.path.join(log_dir, run_id.split("_")[0])
    ensure_dir(log_subdir)

    all_files = sorted(glob.glob(os.path.join(sanit_dir, "*.npz")))
    total     = len(all_files)
    if total == 0:
        print(f"[TD-DFS] No NPZ files found in {sanit_dir}")
        return

    lam_m = C_MPS / fc_hz
    print(
        f"[TD-DFS] Processing {total} files\n"
        f"         Δt search [{delta_t_min}, {delta_t_max}]  "
        f"fc={fc_hz/1e9:.3f}GHz  λ={lam_m*100:.2f}cm"
    )

    start_t      = time.time()
    success      = 0
    failed_list  = []
    delta_t_hist = {}

    for i, fp in enumerate(all_files):
        fname      = os.path.basename(fp)
        fname_stem = os.path.splitext(fname)[0]
        try:
            result = extract_features_single(
                fp, delta_t_min=delta_t_min, delta_t_max=delta_t_max, fc_hz=fc_hz
            )
            label, subject = _parse_label_subject(fp)

            # ── 메인 NPZ 저장
            np.savez_compressed(
                os.path.join(out_dir, fname),
                velocity = result["velocity"],
                fdop     = result["fdop"],
                delta_t  = np.array(result["delta_t"]),
                label    = np.array(label),
                subject  = np.array(subject),
                filename = np.array(fname),
            )

            # ── 선택 저장
            if save_npy:
                _save_npy(result["velocity"], fname_stem, npy_dir)
            if save_csv:
                _save_csv(result["velocity"], result["fdop"],
                          result["delta_t"], fname_stem, csv_dir)

            dt_key = str(result["delta_t"])
            delta_t_hist[dt_key] = delta_t_hist.get(dt_key, 0) + 1
            success += 1

        except Exception as e:
            failed_list.append({"file": fname, "error": str(e)})
            print(f"  [FAIL] {fname}: {e}")

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{total}]  elapsed={time.time()-start_t:.1f}s")

    elapsed = time.time() - start_t

    log_payload = {
        "execution_timestamp": datetime.now().isoformat(),
        "pipeline_stage": "feature_extraction_tddfs",
        "params": {
            "delta_t_min": delta_t_min,
            "delta_t_max": delta_t_max,
            "fc_hz":       fc_hz,
            "lambda_m":    lam_m,
            "fs_hz":       FS,
        },
        "summary": {
            "total_files": total,
            "success":     success,
            "failed":      len(failed_list),
            "elapsed_sec": round(elapsed, 2),
        },
        "delta_t_distribution": delta_t_hist,
        "failures": failed_list,
    }
    log_path = os.path.join(log_subdir, f"{run_id}_tddfs_extraction_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_payload, f, indent=4, ensure_ascii=False)

    print(
        f"\n[TD-DFS] Done. Success={success}, Failed={len(failed_list)}, "
        f"Elapsed={elapsed:.1f}s\n"
        f"  NPZ → {out_dir}\n"
        f"  Log → {log_path}\n"
        f"  Δt distribution: {delta_t_hist}"
    )
    if save_npy: print(f"  NPY → {npy_dir}")
    if save_csv: print(f"  CSV → {csv_dir}")


# ══════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════

def _parse_args():
    p = argparse.ArgumentParser(
        description=(
            "TD-DFS Feature Extractor\n"
            "시간 영역 차분 기반 순간 도플러 속도 추출 (진폭+위상 사용)\n"
            "TD(t) = H(t+Δt) − H(t)  →  fd 역산  →  v [cm/s]"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--delta_t_min", type=int,   default=1,            help="Δt 탐색 최솟값 [샘플] (기본: 1)")
    p.add_argument("--delta_t_max", type=int,   default=10,           help="Δt 탐색 최댓값 [샘플] (기본: 10)")
    p.add_argument("--fc_hz",       type=float, default=5.18e9,       help="WiFi 중심 주파수 [Hz] (기본: 5.18e9)")
    p.add_argument("--sanit_dir",   default=DEFAULT_SANIT_DIR,        help="입력 NPZ 디렉터리")
    p.add_argument("--out_dir",     default=DEFAULT_OUT_DIR,          help="출력 NPZ 디렉터리")
    p.add_argument("--log_dir",     default=DEFAULT_LOG_DIR,          help="JSON 로그 디렉터리")
    p.add_argument("--save_npy",    action="store_true",              help=".npy 1D 배열 별도 저장")
    p.add_argument("--npy_dir",     default=DEFAULT_NPY_DIR,          help="NPY 저장 경로 (--save_npy 시)")
    p.add_argument("--save_csv",    action="store_true",              help=".csv 시계열 별도 저장")
    p.add_argument("--csv_dir",     default=DEFAULT_CSV_DIR,          help="CSV 저장 경로 (--save_csv 시)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_tddfs_extraction(
        sanit_dir   = args.sanit_dir,
        out_dir     = args.out_dir,
        log_dir     = args.log_dir,
        delta_t_min = args.delta_t_min,
        delta_t_max = args.delta_t_max,
        fc_hz       = args.fc_hz,
        save_npy    = args.save_npy,
        npy_dir     = args.npy_dir,
        save_csv    = args.save_csv,
        csv_dir     = args.csv_dir,
    )
