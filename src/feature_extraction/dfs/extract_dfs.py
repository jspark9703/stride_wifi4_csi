"""
DFS (Doppler Frequency Spectrum) Feature Extractor
====================================================
입력  : data/sanitization/sanitization/*.npz  (keys: 'time', 'csi')
        csi : (N_packets, 108) complex64,  fs = 100 Hz
출력  : data/feature_extraction/dfs/*.npz

알고리즘 개요
--------------
STFT(Short-Time Fourier Transform) 기반 전력 스펙트로그램을 추출합니다.
진폭(Amplitude)과 위상(Phase) 정보를 모두 담은 복소 CSI를 직접 변환하여
시간의 흐름에 따른 전반적인 도플러 주파수 변화량을 2D 매트릭스로 표현합니다.

파이프라인
-----------
STEP 1. 복소 CSI 로드  →  (N_packets, N_subcarriers=108) complex64
STEP 2. 위상 정정 (Phase Conjugate Multiplication)
        Δ(t, s) = H(t, s+1) × conj(H(t, s))   for s = 0 .. NS-2
        → CFO / SFO 기인 선형 위상 경사 상쇄
STEP 3. 서브캐리어 평균  →  단일 복소 시계열 z(t)  (N_packets,)
        → 주파수 선택적 페이딩 평활화 + 노이즈 억제
STEP 4. Hanning 윈도우 STFT 수행
        → (N_fft, N_frames) complex64  (fftshift 적용, DC 중심)
STEP 5. 전력 스펙트로그램 산출  P(f,t) = |STFT(f,t)|²
        → dB 스케일 변환 후 [0, 1] Min-Max 정규화
STEP 6. 유효 도플러 대역 필터링
        fs = 100Hz → 최대 ±50Hz (Nyquist 한계)
        ±doppler_hz Hz 범위만 보존

출력 NPZ 키
-----------
  dfs_power   : (N_valid_freq, N_frames)  float32  — 정규화된 전력 스펙트로그램
  dfs_freq_hz : (N_valid_freq,)           float32  — 주파수 눈금 [Hz]
  label       : scalar str                         — 'big' / 'small'
  subject     : scalar str
  filename    : scalar str

추가 선택 저장 (--save_npy / --save_img):
  <stem>_dfs.npy  — 전력 스펙트로그램 2D 배열
  <stem>_dfs.png  — 컬러맵 스펙트로그램 이미지

파라미터
---------
  --n_fft       : STFT FFT 크기           (기본: 64)
  --hop         : STFT hop 크기 [샘플]   (기본: 4)
  --doppler_hz  : 유효 도플러 대역 ±Hz    (기본: 50)
  --sanit_dir   : 입력 NPZ 디렉터리
  --out_dir     : 출력 NPZ 디렉터리
  --log_dir     : JSON 로그 디렉터리
  --save_npy    : .npy 파일 별도 저장
  --save_img    : .png 이미지 저장

실행 예시
---------
  python extract_dfs.py
  python extract_dfs.py --n_fft 128 --hop 8 --doppler_hz 50
  python extract_dfs.py --save_npy --save_img
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
DEFAULT_OUT_DIR   = os.path.join(BASE_DIR, "data", "feature_extraction", "dfs")
DEFAULT_LOG_DIR   = os.path.join(BASE_DIR, "data", "result", "feature_extraction", "dfs")
DEFAULT_NPY_DIR   = os.path.join(BASE_DIR, "data", "feature_extraction", "dfs_npy")
DEFAULT_IMG_DIR   = os.path.join(BASE_DIR, "data", "result", "feature_extraction", "dfs_img")

FS = 100  # 샘플링 주파수 [Hz]  (전처리 후 고정)


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
#  STEP 2: 위상 정정 (Phase Conjugate Multiplication)
# ══════════════════════════════════════════════════════════════

def phase_conjugate_multiply(csi: np.ndarray) -> np.ndarray:
    """
    인접 서브캐리어 위상차로 CFO/SFO 선형 위상 경사 제거.

      Δ(t, s) = H(t, s+1) × conj(H(t, s))   s = 0 .. NS-2

    CFO는 모든 서브캐리어에 공통 위상 오프셋을 추가하고,
    SFO는 서브캐리어 인덱스에 비례하는 선형 경사를 추가한다.
    인접 차분 연산에서 이 공통/선형 항이 상쇄되고 신체 움직임
    도플러 성분만 남는다.

    Returns
    -------
    delta_csi : (N_packets, N_subcarriers-1) complex64
    """
    return (csi[:, 1:] * np.conj(csi[:, :-1])).astype(np.complex64)


# ══════════════════════════════════════════════════════════════
#  STEP 3: 서브캐리어 평균
# ══════════════════════════════════════════════════════════════

def subcarrier_mean(delta_csi: np.ndarray) -> np.ndarray:
    """
    서브캐리어 축 평균 → 단일 복소 시계열.

    Returns
    -------
    z : (N_packets,) complex64
    """
    return delta_csi.mean(axis=1).astype(np.complex64)


# ══════════════════════════════════════════════════════════════
#  STEP 4: Hanning 윈도우 STFT
# ══════════════════════════════════════════════════════════════

def compute_stft(
    z: np.ndarray,
    n_fft: int,
    hop: int,
) -> tuple:
    """
    복소 입력에 Hanning 윈도우 STFT를 수행.
    복소 입력 → 양방향 스펙트럼(±주파수) 획득 → 도플러 방향(접근/후퇴) 구분.

    Parameters
    ----------
    z     : (N,) complex64
    n_fft : FFT 크기  (주파수 해상도 = fs/n_fft  [Hz/bin])
    hop   : 프레임 이동 간격 [샘플]

    Returns
    -------
    stft    : (n_fft, n_frames) complex64  — fftshift 적용 (DC 중심)
    freq_hz : (n_fft,)         float32    — 주파수 눈금 [Hz]
    """
    N   = len(z)
    win = np.hanning(n_fft).astype(np.float32)

    n_frames = max(1, 1 + (N - n_fft) // hop)
    stft = np.zeros((n_fft, n_frames), dtype=np.complex64)

    for i in range(n_frames):
        s   = i * hop
        e   = s + n_fft
        seg = np.zeros(n_fft, dtype=np.complex64)
        if e <= N:
            seg = z[s:e].copy()
        else:
            seg[:N - s] = z[s:N]
        stft[:, i] = np.fft.fft(seg * win)

    # fftshift: 음의 주파수를 왼쪽 → DC 중심
    stft    = np.fft.fftshift(stft, axes=0)
    freq_hz = np.fft.fftshift(np.fft.fftfreq(n_fft, d=1.0 / FS)).astype(np.float32)

    return stft.astype(np.complex64), freq_hz


# ══════════════════════════════════════════════════════════════
#  STEP 5: 전력 스펙트로그램
# ══════════════════════════════════════════════════════════════

def power_spectrogram(stft: np.ndarray) -> np.ndarray:
    """
    전력 스펙트로그램 산출:   P(f, t) = |STFT(f, t)|²

    dB 스케일 변환  →  [0, 1] Min-Max 정규화.

    Returns
    -------
    power : (n_fft, n_frames) float32
    """
    power_db = 10.0 * np.log10(np.abs(stft) ** 2 + 1e-12)  # dB
    p_min, p_max = power_db.min(), power_db.max()
    if p_max - p_min > 1e-8:
        power_db = (power_db - p_min) / (p_max - p_min)
    return power_db.astype(np.float32)


# ══════════════════════════════════════════════════════════════
#  STEP 6: 유효 도플러 대역 필터링
# ══════════════════════════════════════════════════════════════

def filter_doppler_band(
    power: np.ndarray,
    freq_hz: np.ndarray,
    doppler_hz: float,
) -> tuple:
    """
    ±doppler_hz Hz 범위만 보존 (Nyquist = ±50Hz 클리핑).

    인간 보행 도플러: ±0.5 ~ ±5 Hz 주 성분.
    여유 마진으로 ±50Hz (전체 대역) 사용을 기본으로 함.

    Returns
    -------
    filtered_power : (n_valid_freq, n_frames) float32
    valid_freq     : (n_valid_freq,)          float32
    """
    limit = min(doppler_hz, FS / 2.0)
    mask  = np.abs(freq_hz) <= limit
    return power[mask].astype(np.float32), freq_hz[mask].astype(np.float32)


# ══════════════════════════════════════════════════════════════
#  파일 하나 처리
# ══════════════════════════════════════════════════════════════

def extract_features_single(
    npz_path: str,
    n_fft: int        = 64,
    hop: int          = 4,
    doppler_hz: float = 50.0,
) -> dict:
    """
    단일 NPZ → DFS 특징 추출.

    Returns
    -------
    {
      'dfs_power'   : (n_valid_freq, n_frames) float32
      'dfs_freq_hz' : (n_valid_freq,)          float32
    }
    """
    csi   = load_csi(npz_path)                             # (N, NS)
    delta = phase_conjugate_multiply(csi)                  # (N, NS-1)
    z     = subcarrier_mean(delta)                         # (N,)

    stft, freq_hz = compute_stft(z, n_fft=n_fft, hop=hop)
    power         = power_spectrogram(stft)                # (n_fft, n_frames)
    dfs_power, valid_freq = filter_doppler_band(power, freq_hz, doppler_hz)

    return {
        "dfs_power":   dfs_power,   # (n_valid_freq, n_frames)
        "dfs_freq_hz": valid_freq,  # (n_valid_freq,)
    }


# ══════════════════════════════════════════════════════════════
#  저장 헬퍼 (선택)
# ══════════════════════════════════════════════════════════════

def _save_npy(dfs_power: np.ndarray, fname_stem: str, npy_dir: str):
    """DFS 2D 배열을 .npy 파일로 별도 저장 (Stage 4: 포맷 분리)."""
    ensure_dir(npy_dir)
    np.save(os.path.join(npy_dir, f"{fname_stem}_dfs.npy"), dfs_power)


def _save_img(
    dfs_power: np.ndarray,
    freq_hz: np.ndarray,
    fname_stem: str,
    img_dir: str,
    n_frames: int,
):
    """DFS 전력 스펙트로그램 → PNG 이미지 저장."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ensure_dir(img_dir)
        BG, PANEL, TEXT, GRID = "#0D1117", "#161B22", "#E6EDF3", "#30363D"

        time_axis = np.linspace(0, n_frames / FS, n_frames)

        fig, ax = plt.subplots(figsize=(8, 4), facecolor=BG)
        ax.set_facecolor(PANEL)
        ax.spines[:].set_color(GRID)
        ax.tick_params(colors=TEXT)

        im = ax.pcolormesh(
            time_axis, freq_hz, dfs_power,
            shading="auto", cmap="inferno", vmin=0, vmax=1,
        )
        ax.set_xlabel("Time [s]", color=TEXT)
        ax.set_ylabel("Doppler Frequency [Hz]", color=TEXT)
        ax.set_title(f"DFS Power Spectrogram — {fname_stem}", color=TEXT, fontsize=10)
        ax.axhline(0, color="white", lw=0.5, ls="--", alpha=0.4)

        cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cbar.ax.tick_params(colors=TEXT, labelsize=7)
        cbar.set_label("Power (norm. dB)", color=TEXT, fontsize=8)

        plt.tight_layout()
        out_path = os.path.join(img_dir, f"{fname_stem}_dfs.png")
        fig.savefig(out_path, dpi=100, bbox_inches="tight", facecolor=BG)
        plt.close(fig)
    except ImportError:
        print("  [WARN] matplotlib 없음 — 이미지 저장 건너뜀")


# ══════════════════════════════════════════════════════════════
#  배치 처리
# ══════════════════════════════════════════════════════════════

def run_dfs_extraction(
    sanit_dir:  str   = DEFAULT_SANIT_DIR,
    out_dir:    str   = DEFAULT_OUT_DIR,
    log_dir:    str   = DEFAULT_LOG_DIR,
    n_fft:      int   = 64,
    hop:        int   = 4,
    doppler_hz: float = 50.0,
    save_npy:   bool  = False,
    npy_dir:    str   = DEFAULT_NPY_DIR,
    save_img:   bool  = False,
    img_dir:    str   = DEFAULT_IMG_DIR,
):
    ensure_dir(out_dir)
    ensure_dir(log_dir)

    run_id     = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_subdir = os.path.join(log_dir, run_id.split("_")[0])
    ensure_dir(log_subdir)

    all_files = sorted(glob.glob(os.path.join(sanit_dir, "*.npz")))
    total     = len(all_files)
    if total == 0:
        print(f"[DFS] No NPZ files found in {sanit_dir}")
        return

    # 예상 출력 차원
    n_frames_est = max(1, 1 + (256 - n_fft) // hop)
    n_valid_freq = int(sum(1 for f in
                           np.fft.fftshift(np.fft.fftfreq(n_fft, 1.0 / FS))
                           if abs(f) <= min(doppler_hz, FS / 2)))

    print(
        f"[DFS] Processing {total} files\n"
        f"      n_fft={n_fft}, hop={hop}, doppler_hz=±{doppler_hz}Hz\n"
        f"      Output shape ≈ ({n_valid_freq}, ~{n_frames_est})"
    )

    start_t     = time.time()
    success     = 0
    failed_list = []

    for i, fp in enumerate(all_files):
        fname      = os.path.basename(fp)
        fname_stem = os.path.splitext(fname)[0]
        try:
            result = extract_features_single(
                fp, n_fft=n_fft, hop=hop, doppler_hz=doppler_hz
            )
            label, subject = _parse_label_subject(fp)

            # ── 메인 NPZ 저장
            out_path = os.path.join(out_dir, fname)
            np.savez_compressed(
                out_path,
                dfs_power   = result["dfs_power"],
                dfs_freq_hz = result["dfs_freq_hz"],
                label       = np.array(label),
                subject     = np.array(subject),
                filename    = np.array(fname),
            )

            # ── 선택 저장
            if save_npy:
                _save_npy(result["dfs_power"], fname_stem, npy_dir)
            if save_img:
                _save_img(
                    result["dfs_power"], result["dfs_freq_hz"],
                    fname_stem, img_dir, result["dfs_power"].shape[1],
                )

            success += 1

        except Exception as e:
            failed_list.append({"file": fname, "error": str(e)})
            print(f"  [FAIL] {fname}: {e}")

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{total}]  elapsed={time.time()-start_t:.1f}s")

    elapsed = time.time() - start_t

    log_payload = {
        "execution_timestamp": datetime.now().isoformat(),
        "pipeline_stage": "feature_extraction_dfs",
        "params": {
            "n_fft":      n_fft,
            "hop":        hop,
            "doppler_hz": doppler_hz,
            "window":     "hanning",
            "fs_hz":      FS,
            "n_valid_freq_approx": n_valid_freq,
        },
        "summary": {
            "total_files": total,
            "success":     success,
            "failed":      len(failed_list),
            "elapsed_sec": round(elapsed, 2),
        },
        "failures": failed_list,
    }
    log_path = os.path.join(log_subdir, f"{run_id}_dfs_extraction_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_payload, f, indent=4, ensure_ascii=False)

    print(
        f"\n[DFS] Done. Success={success}, Failed={len(failed_list)}, "
        f"Elapsed={elapsed:.1f}s\n"
        f"  NPZ → {out_dir}\n"
        f"  Log → {log_path}"
    )
    if save_npy: print(f"  NPY → {npy_dir}")
    if save_img: print(f"  IMG → {img_dir}")


# ══════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════

def _parse_args():
    p = argparse.ArgumentParser(
        description="DFS Feature Extractor — STFT 기반 전력 스펙트로그램 (진폭+위상 사용)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--n_fft",      type=int,   default=64,              help="STFT FFT 크기 (기본: 64)")
    p.add_argument("--hop",        type=int,   default=4,               help="STFT hop 크기 [샘플] (기본: 4)")
    p.add_argument("--doppler_hz", type=float, default=50.0,            help="유효 도플러 대역 ±Hz (기본: 50)")
    p.add_argument("--sanit_dir",  default=DEFAULT_SANIT_DIR,           help="입력 NPZ 디렉터리")
    p.add_argument("--out_dir",    default=DEFAULT_OUT_DIR,             help="출력 NPZ 디렉터리")
    p.add_argument("--log_dir",    default=DEFAULT_LOG_DIR,             help="JSON 로그 디렉터리")
    p.add_argument("--save_npy",   action="store_true",                 help=".npy 2D 배열 별도 저장")
    p.add_argument("--npy_dir",    default=DEFAULT_NPY_DIR,             help="NPY 저장 경로 (--save_npy 시)")
    p.add_argument("--save_img",   action="store_true",                 help=".png 스펙트로그램 이미지 저장")
    p.add_argument("--img_dir",    default=DEFAULT_IMG_DIR,             help="PNG 저장 경로 (--save_img 시)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_dfs_extraction(
        sanit_dir  = args.sanit_dir,
        out_dir    = args.out_dir,
        log_dir    = args.log_dir,
        n_fft      = args.n_fft,
        hop        = args.hop,
        doppler_hz = args.doppler_hz,
        save_npy   = args.save_npy,
        npy_dir    = args.npy_dir,
        save_img   = args.save_img,
        img_dir    = args.img_dir,
    )
