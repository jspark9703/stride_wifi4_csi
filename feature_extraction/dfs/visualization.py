"""
DFS Feature Visualization
==========================
sanitization NPZ로부터 DFS 전력 스펙트로그램을 계산하여 시각화합니다.
(feature NPZ가 있으면 우선 사용, 없으면 on-the-fly 계산)

저장 이미지
-----------
  dfs_spectrogram_grid.png  : big 10 / small 10 각각의 STFT 스펙트로그램 그리드
  dfs_mean_compare.png      : big vs small 평균 스펙트로그램 + 차분 맵
  dfs_doppler_profile.png   : 주파수 축 평균 도플러 세기 프로파일 비교 (big vs small)
  dfs_feature_heatmap.png   : 10+10 샘플의 평탄화된 스펙트로그램 히트맵

실행
----
  python visualization.py
  python visualization.py --n_fft 128 --hop 8 --doppler_hz 50
"""

import os
import sys
import glob
import random
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ─── 경로 설정 ────────────────────────────────────────────────
HERE      = os.path.dirname(os.path.abspath(__file__))
BASE_DIR  = os.path.abspath(os.path.join(HERE, "..", ".."))
SANIT_DIR = os.path.join(BASE_DIR, "data", "sanitization", "sanitization")
FEAT_DIR  = os.path.join(BASE_DIR, "data", "feature_extraction", "dfs")
OUT_DIR   = HERE

# DFS 기본 파라미터
N_FFT      = 64
HOP        = 4
DOPPLER_HZ = 50.0
FS         = 100
N_SAMPLES  = 10
SEED       = 42

# ─── 스타일 ───────────────────────────────────────────────────
BIG_COLOR   = "#EF5350"
SMALL_COLOR = "#42A5F5"
BG_COLOR    = "#0D1117"
PANEL_COLOR = "#161B22"
TEXT_COLOR  = "#E6EDF3"
GRID_COLOR  = "#30363D"

plt.rcParams.update({
    "figure.facecolor": BG_COLOR,  "axes.facecolor":  PANEL_COLOR,
    "axes.edgecolor":   GRID_COLOR, "axes.labelcolor": TEXT_COLOR,
    "xtick.color": TEXT_COLOR, "ytick.color": TEXT_COLOR,
    "text.color":  TEXT_COLOR, "grid.color":  GRID_COLOR,
    "grid.linewidth": 0.5,
    "font.family": "DejaVu Sans", "font.size": 9,
})


# ══════════════════════════════════════════════════════════════
#  DFS 계산 함수 (extract_dfs.py 와 동일 로직, 독립 실행 가능)
# ══════════════════════════════════════════════════════════════

def _phase_conjugate_multiply(csi: np.ndarray) -> np.ndarray:
    return (csi[:, 1:] * np.conj(csi[:, :-1])).astype(np.complex64)


def _compute_dfs(npz_path: str, n_fft: int, hop: int, doppler_hz: float):
    """
    sanitization NPZ → DFS 전력 스펙트로그램 + 주파수 눈금.

    Returns
    -------
    power    : (n_valid_freq, n_frames) float32
    freq_hz  : (n_valid_freq,)          float32
    """
    data  = np.load(npz_path)
    csi   = data["csi"].astype(np.complex64)           # (N, NS)
    delta = _phase_conjugate_multiply(csi)              # (N, NS-1)
    z     = delta.mean(axis=1)                          # (N,) complex

    N   = len(z)
    win = np.hanning(n_fft).astype(np.float32)
    n_frames = max(1, 1 + (N - n_fft) // hop)

    stft = np.zeros((n_fft, n_frames), dtype=np.complex64)
    for i in range(n_frames):
        s = i * hop; e = s + n_fft
        seg = np.zeros(n_fft, dtype=np.complex64)
        if e <= N:
            seg = z[s:e].copy()
        else:
            seg[:N - s] = z[s:N]
        stft[:, i] = np.fft.fft(seg * win)

    stft    = np.fft.fftshift(stft, axes=0)
    freq_hz = np.fft.fftshift(np.fft.fftfreq(n_fft, 1.0 / FS)).astype(np.float32)

    power_db = 10.0 * np.log10(np.abs(stft) ** 2 + 1e-12)
    p_min, p_max = power_db.min(), power_db.max()
    if p_max - p_min > 1e-8:
        power_db = (power_db - p_min) / (p_max - p_min)
    power_db = power_db.astype(np.float32)

    limit = min(doppler_hz, FS / 2.0)
    mask  = np.abs(freq_hz) <= limit
    return power_db[mask], freq_hz[mask]


def load_dfs(npz_path: str, n_fft: int, hop: int, doppler_hz: float):
    """
    feature NPZ 우선 로드, 없으면 sanitization NPZ에서 on-the-fly 계산.
    """
    fname = os.path.basename(npz_path)
    feat_path = os.path.join(FEAT_DIR, fname)
    if os.path.exists(feat_path):
        d = np.load(feat_path)
        return d["dfs_power"].astype(np.float32), d["dfs_freq_hz"].astype(np.float32)
    return _compute_dfs(npz_path, n_fft, hop, doppler_hz)


# ══════════════════════════════════════════════════════════════
#  파일 수집
# ══════════════════════════════════════════════════════════════

def collect_files():
    all_files = sorted(glob.glob(os.path.join(SANIT_DIR, "*.npz")))
    big, small = [], []
    for fp in all_files:
        stem = os.path.basename(fp).lower()
        if "_big" in stem:
            big.append(fp)
        elif "_small" in stem or "_smal" in stem:
            small.append(fp)
    rng = random.Random(SEED)
    return (rng.sample(big,   min(N_SAMPLES, len(big))),
            rng.sample(small, min(N_SAMPLES, len(small))))


# ══════════════════════════════════════════════════════════════
#  PLOT 1: 스펙트로그램 그리드  (big 10 상단 / small 10 하단)
# ══════════════════════════════════════════════════════════════

def plot_spectrogram_grid(big_files, small_files, n_fft, hop, doppler_hz):
    ncols = max(len(big_files), len(small_files))
    fig   = plt.figure(figsize=(ncols * 2.5, 10), facecolor=BG_COLOR)
    fig.suptitle(
        f"DFS Power Spectrogram Grid  ─  Big (top) vs Small (bottom)  "
        f"|  n_fft={n_fft}, hop={hop}, ±{doppler_hz}Hz",
        fontsize=13, fontweight="bold", color=TEXT_COLOR, y=0.99,
    )
    gs = gridspec.GridSpec(
        2, ncols, figure=fig,
        hspace=0.5, wspace=0.15,
        top=0.92, bottom=0.06, left=0.04, right=0.98,
    )

    for row, (files, color, name) in enumerate([
        (big_files,   BIG_COLOR,   "Big"),
        (small_files, SMALL_COLOR, "Small"),
    ]):
        for col, fp in enumerate(files):
            ax = fig.add_subplot(gs[row, col])
            try:
                power, freq_hz = load_dfs(fp, n_fft, hop, doppler_hz)
                # (n_valid_freq, n_frames)
                n_frames  = power.shape[1]
                time_axis = np.linspace(0, n_frames * hop / FS, n_frames)

                im = ax.pcolormesh(
                    time_axis, freq_hz, power,
                    shading="auto", cmap="inferno", vmin=0, vmax=1,
                )
                ax.axhline(0, color="white", lw=0.5, ls="--", alpha=0.3)
                ax.set_title(
                    os.path.basename(fp).split("_")[3][:8],
                    fontsize=7, color=color, pad=2,
                )
                ax.set_xticks([])
                ax.set_yticks([freq_hz[0], 0, freq_hz[-1]])
                ax.set_yticklabels(
                    [f"{freq_hz[0]:.0f}", "0", f"{freq_hz[-1]:.0f}"],
                    fontsize=6, color=TEXT_COLOR,
                )
                ax.tick_params(length=0)
                for sp in ax.spines.values():
                    sp.set_edgecolor(color); sp.set_linewidth(1.5)
            except Exception as e:
                ax.text(0.5, 0.5, "ERR", ha="center", va="center",
                        transform=ax.transAxes, color="red", fontsize=8)

        y_pos = 0.78 if row == 0 else 0.32
        fig.text(0.005, y_pos, name, fontsize=12, color=color,
                 fontweight="bold", va="center")

    fig.text(0.5, 0.02, "Time [s]",         ha="center", color=TEXT_COLOR, fontsize=9)
    fig.text(0.0, 0.5,  "Doppler Freq [Hz]", va="center", color=TEXT_COLOR,
             fontsize=9, rotation=90)

    out = os.path.join(OUT_DIR, "dfs_spectrogram_grid.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  [saved] {out}")


# ══════════════════════════════════════════════════════════════
#  PLOT 2: 평균 스펙트로그램 비교 + 차분 맵
# ══════════════════════════════════════════════════════════════

def plot_mean_compare(big_files, small_files, n_fft, hop, doppler_hz):
    def load_all(files):
        """각 파일의 DFS 로드. 리스트로 반환 (길이 불균일)"""
        mats, f_ref = [], None
        for fp in files:
            try:
                power, freq_hz = load_dfs(fp, n_fft, hop, doppler_hz)
                mats.append(power)
                f_ref = freq_hz
            except Exception:
                pass
        return mats, f_ref

    big_mats,   freq_hz = load_all(big_files)
    small_mats, _       = load_all(small_files)

    if not big_mats or not small_mats:
        print("  [SKIP] plot_mean_compare"); return

    # ── 전체(big+small) 통틀어 최소 시간 프레임으로 통일
    global_min_t = min(m.shape[1] for m in big_mats + small_mats)

    big_stack   = np.stack([m[:, :global_min_t] for m in big_mats],   axis=0)  # (M, n_freq, T)
    small_stack = np.stack([m[:, :global_min_t] for m in small_mats], axis=0)

    big_mean   = big_stack.mean(axis=0)    # (n_freq, T)
    small_mean = small_stack.mean(axis=0)
    diff       = big_mean - small_mean     # shape 보장

    vmax    = max(big_mean.max(), small_mean.max())
    vabs    = np.abs(diff).max()
    n_frames  = global_min_t
    time_axis = np.linspace(0, n_frames * hop / FS, n_frames)

    fig, axes = plt.subplots(1, 3, figsize=(21, 5), facecolor=BG_COLOR)
    fig.suptitle(
        "DFS Mean Power Spectrogram  ─  Big vs Small vs Difference",
        fontsize=13, fontweight="bold", color=TEXT_COLOR, y=1.01,
    )

    configs = [
        (big_mean,   "inferno", f"Big Mean  (n={len(big_mats)})",    0,     vmax),
        (small_mean, "inferno", f"Small Mean  (n={len(small_mats)})", 0,     vmax),
        (diff,       "seismic", "Difference  (Big − Small)",          -vabs, vabs),
    ]

    for ax, (mat, cmap, title, vmin, vmax_cur) in zip(axes, configs):
        ax.set_facecolor(PANEL_COLOR)
        ax.spines[:].set_color(GRID_COLOR)
        ax.tick_params(colors=TEXT_COLOR, labelsize=8)

        im = ax.pcolormesh(
            time_axis, freq_hz, mat,
            shading="auto", cmap=cmap, vmin=vmin, vmax=vmax_cur,
        )
        ax.axhline(0, color="white", lw=0.6, ls="--", alpha=0.4)
        ax.set_title(title, fontsize=10, color=TEXT_COLOR, pad=5)
        ax.set_xlabel("Time [s]",          color=TEXT_COLOR, fontsize=9)
        ax.set_ylabel("Doppler Freq [Hz]",  color=TEXT_COLOR, fontsize=9)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.05)
        cb  = fig.colorbar(im, cax=cax)
        cb.ax.tick_params(colors=TEXT_COLOR, labelsize=7)
        cb.set_label("Power (norm. dB)" if cmap == "inferno" else "Difference",
                     color=TEXT_COLOR, fontsize=7)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "dfs_mean_compare.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  [saved] {out}")


# ══════════════════════════════════════════════════════════════
#  PLOT 3: 주파수 축 평균 도플러 세기 프로파일
# ══════════════════════════════════════════════════════════════

def plot_doppler_profile(big_files, small_files, n_fft, hop, doppler_hz):
    """
    각 파일의 DFS를 시간 축으로 평균 → 주파수 vs 도플러 세기 1D 프로파일.
    big vs small 오버레이로 도플러 주파수 분포 차이를 시각화.
    """
    def get_profiles(files):
        profiles, f_ref = [], None
        for fp in files:
            try:
                power, freq_hz = load_dfs(fp, n_fft, hop, doppler_hz)
                profiles.append(power.mean(axis=1))  # (n_freq,)
                f_ref = freq_hz
            except Exception:
                pass
        return np.array(profiles) if profiles else None, f_ref

    big_p,   freq_hz = get_profiles(big_files)
    small_p, _       = get_profiles(small_files)

    if big_p is None or small_p is None:
        print("  [SKIP] plot_doppler_profile"); return

    fig, ax = plt.subplots(figsize=(12, 5), facecolor=BG_COLOR)
    ax.set_facecolor(PANEL_COLOR)
    ax.spines[:].set_color(GRID_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    ax.grid(alpha=0.25)

    # 개별 샘플 (투명)
    for p in big_p:
        ax.plot(freq_hz, p, color=BIG_COLOR,   alpha=0.18, lw=0.8)
    for p in small_p:
        ax.plot(freq_hz, p, color=SMALL_COLOR, alpha=0.18, lw=0.8)

    # 평균선
    ax.plot(freq_hz, big_p.mean(axis=0),   color=BIG_COLOR,   lw=2.5,
            label=f"Big mean (n={len(big_p)})")
    ax.plot(freq_hz, small_p.mean(axis=0), color=SMALL_COLOR, lw=2.5,
            label=f"Small mean (n={len(small_p)})")

    # ±1σ 밴드
    for arr, color in [(big_p, BIG_COLOR), (small_p, SMALL_COLOR)]:
        mu, sigma = arr.mean(axis=0), arr.std(axis=0)
        ax.fill_between(freq_hz, mu - sigma, mu + sigma, color=color, alpha=0.12)

    ax.axvline(0, color="white", lw=0.8, ls="--", alpha=0.5)
    ax.set_title(
        "DFS Doppler Power Profile (Time-averaged)  ─  Big vs Small",
        fontsize=12, color=TEXT_COLOR, pad=8,
    )
    ax.set_xlabel("Doppler Frequency [Hz]", color=TEXT_COLOR, fontsize=10)
    ax.set_ylabel("Normalized Power (avg)",  color=TEXT_COLOR, fontsize=10)
    ax.set_xlim(freq_hz[0], freq_hz[-1])
    ax.legend(facecolor=PANEL_COLOR, edgecolor=GRID_COLOR,
              labelcolor=TEXT_COLOR, fontsize=10)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "dfs_doppler_profile.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  [saved] {out}")


# ══════════════════════════════════════════════════════════════
#  PLOT 4: 특징 벡터 히트맵 (평탄화된 DFS 스펙트로그램)
# ══════════════════════════════════════════════════════════════

def plot_feature_heatmap(big_files, small_files, n_fft, hop, doppler_hz):
    ravels, row_labels, row_colors = [], [], []

    for fp, color in [(f, BIG_COLOR) for f in big_files] + \
                     [(f, SMALL_COLOR) for f in small_files]:
        try:
            power, _ = load_dfs(fp, n_fft, hop, doppler_hz)
            ravels.append(power.ravel())
            row_labels.append(os.path.basename(fp).split("_")[3][:8])
            row_colors.append(color)
        except Exception:
            pass

    if not ravels:
        print("  [SKIP] plot_feature_heatmap"); return

    # 각 행의 길이가 다를 수 있으므로 (시간 축 길이 차이) 최소 길이로 통일
    min_len  = min(len(r) for r in ravels)
    mat      = np.array([r[:min_len] for r in ravels], dtype=np.float32)  # (M, D)

    row_min  = mat.min(axis=1, keepdims=True)
    row_rng  = mat.max(axis=1, keepdims=True) - row_min + 1e-10
    mat_norm = ((mat - row_min) / row_rng).astype(np.float32)
    n_rows, n_cols = mat_norm.shape

    fig, ax = plt.subplots(
        figsize=(max(16, n_cols // 10), max(6, n_rows * 0.6)),
        facecolor=BG_COLOR,
    )
    fig.suptitle(
        f"DFS Feature Heatmap  ─  {len(big_files)} Big (top) + {len(small_files)} Small (bottom)  "
        f"|  dim={n_cols}",
        fontsize=12, fontweight="bold", color=TEXT_COLOR, y=1.01,
    )
    ax.set_facecolor(PANEL_COLOR)
    ax.spines[:].set_color(GRID_COLOR)
    ax.tick_params(colors=TEXT_COLOR, length=0)

    im = ax.imshow(mat_norm, aspect="auto", cmap="inferno",
                   interpolation="nearest", vmin=0, vmax=1)

    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=8)
    for lbl, rc in zip(ax.get_yticklabels(), row_colors):
        lbl.set_color(rc)

    # x축: 주파수 빈 경계 (n_fft 기준)
    n_freq = sum(1 for f in np.fft.fftshift(np.fft.fftfreq(n_fft, 1.0 / FS))
                 if abs(f) <= min(doppler_hz, FS / 2))
    n_time = n_cols // n_freq
    for fi in range(1, n_freq, max(1, n_freq // 8)):
        ax.axvline(fi * n_time - 0.5, color=GRID_COLOR, lw=0.3, ls="--", alpha=0.4)

    # big/small 경계
    ax.axhline(len(big_files) - 0.5, color="white", lw=1.5)

    cbar = fig.colorbar(im, ax=ax, fraction=0.012, pad=0.01)
    cbar.ax.tick_params(colors=TEXT_COLOR, labelsize=7)
    cbar.set_label("Normalized Power", color=TEXT_COLOR, fontsize=8)

    patches = [
        mpatches.Patch(color=BIG_COLOR,   label="Big"),
        mpatches.Patch(color=SMALL_COLOR, label="Small"),
    ]
    ax.legend(handles=patches, loc="upper right",
              facecolor=PANEL_COLOR, edgecolor=GRID_COLOR,
              labelcolor=TEXT_COLOR, fontsize=9)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "dfs_feature_heatmap.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  [saved] {out}")


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def _parse_args():
    p = argparse.ArgumentParser(description="DFS Feature Visualization")
    p.add_argument("--n_fft",      type=int,   default=N_FFT,       help=f"STFT FFT 크기 (기본: {N_FFT})")
    p.add_argument("--hop",        type=int,   default=HOP,         help=f"STFT hop (기본: {HOP})")
    p.add_argument("--doppler_hz", type=float, default=DOPPLER_HZ,  help=f"도플러 대역 ±Hz (기본: {DOPPLER_HZ})")
    p.add_argument("--n_samples",  type=int,   default=N_SAMPLES,   help=f"클래스당 샘플 수 (기본: {N_SAMPLES})")
    p.add_argument("--sanit_dir",  default=SANIT_DIR)
    p.add_argument("--feat_dir",   default=FEAT_DIR)
    p.add_argument("--out_dir",    default=OUT_DIR)
    return p.parse_args()


def main():
    args = _parse_args()
    global SANIT_DIR, FEAT_DIR, OUT_DIR, N_SAMPLES
    SANIT_DIR, FEAT_DIR, OUT_DIR, N_SAMPLES = \
        args.sanit_dir, args.feat_dir, args.out_dir, args.n_samples
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"[DFS Visualization] Loading from:\n  {SANIT_DIR}")
    big_files, small_files = collect_files()
    print(f"  Big={len(big_files)}, Small={len(small_files)}")

    if not big_files or not small_files:
        print("[ERROR] 파일 부족. SANIT_DIR 경로를 확인하세요.")
        sys.exit(1)

    print("\n[1/4] Spectrogram Grid...")
    plot_spectrogram_grid(big_files, small_files, args.n_fft, args.hop, args.doppler_hz)

    print("[2/4] Mean Spectrogram Compare...")
    plot_mean_compare(big_files, small_files, args.n_fft, args.hop, args.doppler_hz)

    print("[3/4] Doppler Power Profile...")
    plot_doppler_profile(big_files, small_files, args.n_fft, args.hop, args.doppler_hz)

    print("[4/4] Feature Heatmap...")
    plot_feature_heatmap(big_files, small_files, args.n_fft, args.hop, args.doppler_hz)

    print(f"\n[DFS Visualization] All done → {OUT_DIR}")


if __name__ == "__main__":
    main()
