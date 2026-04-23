"""
SDP Feature Visualization
==========================
정제된 CSI 데이터에서 SDP 2D 매트릭스를 계산하여 시각화합니다.

저장 이미지
-----------
  sdp_matrix_grid.png     : big 10 / small 10 각각의 SDP 히트맵 그리드 (2 × 10 패널)
  sdp_mean_compare.png    : big vs small 평균 SDP 매트릭스 나란히 비교 + 차분 맵
  sdp_acf_profile.png     : lag별 ACF 평균 프로파일 big vs small 오버레이
  sdp_feature_heatmap.png : 10+10 샘플 특징 벡터 히트맵
"""

import os
import sys
import glob
import random

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ─── 경로 설정 ────────────────────────────────────────────────
HERE        = os.path.dirname(os.path.abspath(__file__))
BASE_DIR    = os.path.abspath(os.path.join(HERE, "..", ".."))
SANIT_DIR   = os.path.join(BASE_DIR, "data", "sanitization", "sanitization")
OUT_DIR     = HERE

N_LAG       = 20
WT          = 100
HOP         = 50
N_SAMPLES   = 10
SEED        = 42

# ─── 스타일 ───────────────────────────────────────────────────
BIG_COLOR   = "#EF5350"
SMALL_COLOR = "#42A5F5"
BG_COLOR    = "#0D1117"
PANEL_COLOR = "#161B22"
TEXT_COLOR  = "#E6EDF3"
GRID_COLOR  = "#30363D"

plt.rcParams.update({
    "figure.facecolor" : BG_COLOR,
    "axes.facecolor"   : PANEL_COLOR,
    "axes.edgecolor"   : GRID_COLOR,
    "axes.labelcolor"  : TEXT_COLOR,
    "xtick.color"      : TEXT_COLOR,
    "ytick.color"      : TEXT_COLOR,
    "text.color"       : TEXT_COLOR,
    "grid.color"       : GRID_COLOR,
    "grid.linewidth"   : 0.5,
    "font.family"      : "DejaVu Sans",
    "font.size"        : 9,
})


# ══════════════════════════════════════════════════════════════
# SDP 계산 함수 (extract_sdp.py 와 동일 로직, 독립 실행 가능)
# ══════════════════════════════════════════════════════════════
def _compute_sdp(window: np.ndarray, n_lag: int) -> np.ndarray:
    """window (WT, NS) → sdp_matrix (n_lag, WT) float32"""
    WT, NS = window.shape
    w = window - window.mean(axis=0, keepdims=True)
    acf_tensor = np.zeros((n_lag, WT, NS), dtype=np.float32)
    for tau in range(1, n_lag + 1):
        valid = WT - tau
        if valid <= 0:
            break
        acf_tensor[tau - 1, :valid, :] = w[:valid, :] * w[tau: tau + valid, :]
    sdp_2d  = acf_tensor.mean(axis=2)
    sdp_2d  = np.abs(sdp_2d)
    col_sum = sdp_2d.sum(axis=0, keepdims=True) + 1e-10
    return (sdp_2d / col_sum).astype(np.float32)


def load_sdp(npz_path, n_lag=N_LAG, wt=WT) -> np.ndarray:
    """sanitization NPZ → SDP 2D matrix (n_lag, wt)"""
    data      = np.load(npz_path)
    amplitude = np.abs(data["csi"]).astype(np.float32)   # (N, 108)
    N, NS     = amplitude.shape

    if N < wt:
        pad = np.zeros((wt - N, NS), dtype=np.float32)
        amplitude = np.vstack([amplitude, pad])
        N = wt

    # 중앙 윈도우 선택
    start   = max(0, (N - wt) // 2)
    window  = amplitude[start: start + wt]
    return _compute_sdp(window, n_lag)


# ══════════════════════════════════════════════════════════════
# 파일 수집
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
    sel_big   = rng.sample(big,   min(N_SAMPLES, len(big)))
    sel_small = rng.sample(small, min(N_SAMPLES, len(small)))
    return sel_big, sel_small


# ══════════════════════════════════════════════════════════════
# PLOT 1: SDP 히트맵 그리드  (big 10 상단 / small 10 하단)
# ══════════════════════════════════════════════════════════════
def plot_sdp_grid(big_files, small_files):
    n_big   = len(big_files)
    n_small = len(small_files)
    ncols   = max(n_big, n_small)

    fig = plt.figure(figsize=(ncols * 2.4, 10), facecolor=BG_COLOR)
    fig.suptitle(
        f"SDP Matrix Grid  ─  Big (top) vs Small (bottom)  |  n_lag={N_LAG}, wt={WT}",
        fontsize=13, fontweight="bold", color=TEXT_COLOR, y=0.99,
    )

    gs = gridspec.GridSpec(
        2, ncols, figure=fig,
        hspace=0.45, wspace=0.15,
        top=0.92, bottom=0.06, left=0.04, right=0.98,
    )

    groups = [(big_files, "Big", BIG_COLOR, 0), (small_files, "Small", SMALL_COLOR, 1)]

    for files, group_name, color, row in groups:
        for col, fp in enumerate(files):
            ax = fig.add_subplot(gs[row, col])
            try:
                sdp = load_sdp(fp)       # (N_lag, WT)

                im = ax.imshow(
                    sdp, aspect="auto", cmap="plasma",
                    origin="lower", interpolation="nearest",
                    vmin=0,
                )
                ax.set_title(
                    os.path.basename(fp).split("_")[3][:8],
                    fontsize=7, color=color, pad=2,
                )
                ax.set_xticks([])
                ax.set_yticks([0, N_LAG - 1])
                ax.set_yticklabels(["τ=1", f"τ={N_LAG}"], fontsize=6, color=TEXT_COLOR)
                ax.tick_params(length=0)
                # 테두리 색상
                for spine in ax.spines.values():
                    spine.set_edgecolor(color)
                    spine.set_linewidth(1.5)
            except Exception as e:
                ax.text(0.5, 0.5, f"ERR", ha="center", va="center",
                        transform=ax.transAxes, color="red", fontsize=8)

        # 그룹 라벨
        y_pos = 0.78 if row == 0 else 0.34
        fig.text(0.005, y_pos, group_name, fontsize=12, color=color,
                 fontweight="bold", va="center")

    # x축 공통 라벨
    fig.text(0.5, 0.02, "Time (packets)", ha="center", color=TEXT_COLOR, fontsize=9)
    fig.text(0.0, 0.5, "Lag τ (samples)", va="center", color=TEXT_COLOR,
             fontsize=9, rotation=90)

    out = os.path.join(OUT_DIR, "sdp_matrix_grid.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  [saved] {out}")


# ══════════════════════════════════════════════════════════════
# PLOT 2: 평균 SDP 비교 + 차분 맵
# ══════════════════════════════════════════════════════════════
def plot_mean_compare(big_files, small_files):
    def stack_sdp(files):
        mats = []
        for fp in files:
            try:
                mats.append(load_sdp(fp))
            except Exception:
                pass
        return np.stack(mats, axis=0) if mats else None   # (M, n_lag, WT)

    big_stack   = stack_sdp(big_files)
    small_stack = stack_sdp(small_files)
    if big_stack is None or small_stack is None:
        print("  [SKIP] plot_mean_compare")
        return

    big_mean   = big_stack.mean(axis=0)
    small_mean = small_stack.mean(axis=0)
    diff       = big_mean - small_mean

    vmax_main = max(big_mean.max(), small_mean.max())
    vabs_diff = np.abs(diff).max()

    fig, axes = plt.subplots(1, 3, figsize=(20, 5), facecolor=BG_COLOR)
    fig.suptitle(
        "Mean SDP Matrix  ─  Big vs Small vs Difference",
        fontsize=13, fontweight="bold", color=TEXT_COLOR, y=1.01,
    )

    triples = [
        (big_mean,   "plasma",  f"Big Mean  (n={len(big_files)})",   vmax_main, None),
        (small_mean, "plasma",  f"Small Mean (n={len(small_files)})", vmax_main, None),
        (diff,       "seismic", "Difference  (Big − Small)",          vabs_diff, vabs_diff),
    ]

    for ax, (mat, cmap, title, vmax, vabs) in zip(axes, triples):
        ax.set_facecolor(PANEL_COLOR)
        ax.spines[:].set_color(GRID_COLOR)
        ax.tick_params(colors=TEXT_COLOR, labelsize=8)

        kw = dict(aspect="auto", cmap=cmap, origin="lower", interpolation="bilinear")
        if vabs is not None:
            im = ax.imshow(mat, vmin=-vabs, vmax=vabs, **kw)
        else:
            im = ax.imshow(mat, vmin=0, vmax=vmax, **kw)

        ax.set_title(title, fontsize=10, color=TEXT_COLOR, pad=5)
        ax.set_xlabel("Time (packets)", color=TEXT_COLOR, fontsize=8)
        ax.set_ylabel("Lag τ (samples)", color=TEXT_COLOR, fontsize=8)
        ax.set_yticks(np.linspace(0, N_LAG - 1, 5).astype(int))
        ax.set_yticklabels([f"τ={v+1}" for v in np.linspace(0, N_LAG - 1, 5).astype(int)],
                           fontsize=7)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.05)
        cb  = fig.colorbar(im, cax=cax)
        cb.ax.tick_params(colors=TEXT_COLOR, labelsize=7)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "sdp_mean_compare.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  [saved] {out}")


# ══════════════════════════════════════════════════════════════
# PLOT 3: lag별 ACF 평균 프로파일 오버레이
# ══════════════════════════════════════════════════════════════
def plot_acf_profile(big_files, small_files):
    def mean_lag_profile(files):
        """각 파일의 SDP를 시간 축으로 평균 → lag 프로파일 (n_lag,)"""
        profiles = []
        for fp in files:
            try:
                sdp = load_sdp(fp)               # (n_lag, WT)
                profiles.append(sdp.mean(axis=1))  # (n_lag,)
            except Exception:
                pass
        return np.array(profiles) if profiles else None

    big_p   = mean_lag_profile(big_files)   # (M, n_lag)
    small_p = mean_lag_profile(small_files)

    if big_p is None or small_p is None:
        print("  [SKIP] plot_acf_profile")
        return

    lags = np.arange(1, N_LAG + 1)

    fig, ax = plt.subplots(figsize=(12, 5), facecolor=BG_COLOR)
    ax.set_facecolor(PANEL_COLOR)
    ax.spines[:].set_color(GRID_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    ax.grid(alpha=0.3)

    # 개별 샘플 (투명)
    for prof in big_p:
        ax.plot(lags, prof, color=BIG_COLOR, alpha=0.2, lw=0.8)
    for prof in small_p:
        ax.plot(lags, prof, color=SMALL_COLOR, alpha=0.2, lw=0.8)

    # 평균 굵게
    ax.plot(lags, big_p.mean(axis=0),   color=BIG_COLOR,   lw=2.5,
            label=f"Big mean (n={len(big_p)})")
    ax.plot(lags, small_p.mean(axis=0), color=SMALL_COLOR, lw=2.5,
            label=f"Small mean (n={len(small_p)})")

    # ±1σ 밴드
    ax.fill_between(
        lags,
        big_p.mean(axis=0) - big_p.std(axis=0),
        big_p.mean(axis=0) + big_p.std(axis=0),
        color=BIG_COLOR, alpha=0.15,
    )
    ax.fill_between(
        lags,
        small_p.mean(axis=0) - small_p.std(axis=0),
        small_p.mean(axis=0) + small_p.std(axis=0),
        color=SMALL_COLOR, alpha=0.15,
    )

    ax.set_title("SDP Lag Profile (Time-averaged ACF)  ─  Big vs Small",
                 fontsize=12, color=TEXT_COLOR, pad=8)
    ax.set_xlabel("Lag τ (samples)", color=TEXT_COLOR, fontsize=10)
    ax.set_ylabel("Normalized ACF (prob. avg)", color=TEXT_COLOR, fontsize=10)
    ax.set_xlim(lags[0], lags[-1])
    ax.legend(facecolor=PANEL_COLOR, edgecolor=GRID_COLOR,
              labelcolor=TEXT_COLOR, fontsize=10)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "sdp_acf_profile.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  [saved] {out}")


# ══════════════════════════════════════════════════════════════
# PLOT 4: 특징 벡터 히트맵 (평탄화된 SDP)
# ══════════════════════════════════════════════════════════════
def plot_feature_heatmap(big_files, small_files):
    feat_matrix = []
    row_labels  = []
    row_colors  = []

    for fp in big_files:
        try:
            sdp = load_sdp(fp)
            feat_matrix.append(sdp.ravel())
            row_labels.append(os.path.basename(fp).split("_")[3][:8])
            row_colors.append(BIG_COLOR)
        except Exception:
            pass

    for fp in small_files:
        try:
            sdp = load_sdp(fp)
            feat_matrix.append(sdp.ravel())
            row_labels.append(os.path.basename(fp).split("_")[3][:8])
            row_colors.append(SMALL_COLOR)
        except Exception:
            pass

    if not feat_matrix:
        print("  [SKIP] plot_feature_heatmap")
        return

    mat      = np.array(feat_matrix)   # (M, n_lag*WT)
    # 각 행 min-max 정규화
    row_min  = mat.min(axis=1, keepdims=True)
    row_rng  = mat.max(axis=1, keepdims=True) - row_min + 1e-10
    mat_norm = (mat - row_min) / row_rng

    n_rows, n_cols = mat_norm.shape
    fig, ax = plt.subplots(
        figsize=(max(16, n_cols // 8), max(6, n_rows * 0.6)),
        facecolor=BG_COLOR,
    )
    fig.suptitle(
        f"SDP Feature Heatmap  ─  {len(big_files)} Big (top) + {len(small_files)} Small (bottom)  "
        f"|  dim={n_cols} (lag×time)",
        fontsize=12, fontweight="bold", color=TEXT_COLOR, y=1.01,
    )

    ax.set_facecolor(PANEL_COLOR)
    ax.spines[:].set_color(GRID_COLOR)
    ax.tick_params(colors=TEXT_COLOR, length=0)

    im = ax.imshow(mat_norm, aspect="auto", cmap="inferno",
                   interpolation="nearest", vmin=0, vmax=1)

    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=8)
    for tick_lbl, rc in zip(ax.get_yticklabels(), row_colors):
        tick_lbl.set_color(rc)

    # x축: lag 경계 표시
    for lag_i in range(1, N_LAG):
        ax.axvline(lag_i * WT - 0.5, color=GRID_COLOR, lw=0.4, ls="--", alpha=0.5)
    lag_centers = [int((i + 0.5) * WT) for i in range(N_LAG)]
    ax.set_xticks(lag_centers[::4])
    ax.set_xticklabels([f"τ={i*4+1}" for i in range(len(lag_centers[::4]))],
                       fontsize=7, rotation=45, ha="right")

    # big/small 경계
    separator  = len(big_files) - 0.5
    ax.axhline(separator, color="white", lw=1.5)

    cbar = fig.colorbar(im, ax=ax, fraction=0.012, pad=0.01)
    cbar.ax.tick_params(colors=TEXT_COLOR, labelsize=7)
    cbar.set_label("Normalized SDP value", color=TEXT_COLOR, fontsize=8)

    import matplotlib.patches as mpatches
    patches = [
        mpatches.Patch(color=BIG_COLOR,   label="Big"),
        mpatches.Patch(color=SMALL_COLOR, label="Small"),
    ]
    ax.legend(handles=patches, loc="upper right",
              facecolor=PANEL_COLOR, edgecolor=GRID_COLOR,
              labelcolor=TEXT_COLOR, fontsize=9)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "sdp_feature_heatmap.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  [saved] {out}")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main():
    print(f"[SDP Visualization] Loading samples from:\n  {SANIT_DIR}")
    big_files, small_files = collect_files()
    print(f"  Big={len(big_files)}, Small={len(small_files)}")

    if not big_files or not small_files:
        print("[ERROR] Not enough files. Check SANIT_DIR.")
        sys.exit(1)

    print("\n[1/4] SDP Matrix Grid...")
    plot_sdp_grid(big_files, small_files)

    print("[2/4] Mean SDP Compare...")
    plot_mean_compare(big_files, small_files)

    print("[3/4] ACF Lag Profile...")
    plot_acf_profile(big_files, small_files)

    print("[4/4] Feature Heatmap...")
    plot_feature_heatmap(big_files, small_files)

    print(f"\n[SDP Visualization] All done → {OUT_DIR}")


if __name__ == "__main__":
    main()
