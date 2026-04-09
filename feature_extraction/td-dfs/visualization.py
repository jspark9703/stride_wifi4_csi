"""
TD-DFS Feature Visualization
==============================
sanitization NPZ로부터 TD-DFS 순간 도플러 속도 시계열을 계산하여 시각화합니다.
(feature NPZ가 있으면 우선 사용, 없으면 on-the-fly 계산)

저장 이미지
-----------
  tddfs_velocity_grid.png   : big 10 / small 10 속도 시계열 그리드 (2 × 10 패널)
  tddfs_velocity_compare.png: big vs small 평균 속도 시계열 + ±1σ 밴드 오버레이
  tddfs_speed_dist.png      : big vs small 속도 절댓값 분포 (히스토그램 + KDE)
  tddfs_delta_t_stats.png   : 최적 Δt 분포 + 클래스별 속도 통계 요약

실행
----
  python visualization.py
  python visualization.py --delta_t_max 15 --fc_hz 5.18e9
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

# ─── 경로 설정 ────────────────────────────────────────────────
HERE      = os.path.dirname(os.path.abspath(__file__))
BASE_DIR  = os.path.abspath(os.path.join(HERE, "..", ".."))
SANIT_DIR = os.path.join(BASE_DIR, "data", "sanitization", "sanitization")
FEAT_DIR  = os.path.join(BASE_DIR, "data", "feature_extraction", "td-dfs")
OUT_DIR   = HERE

# TD-DFS 기본 파라미터
DELTA_T_MIN = 1
DELTA_T_MAX = 10
FC_HZ       = 5.18e9
C_MPS       = 3e8
FS          = 100
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
    "figure.facecolor": BG_COLOR,  "axes.facecolor":  PANEL_COLOR,
    "axes.edgecolor":   GRID_COLOR, "axes.labelcolor": TEXT_COLOR,
    "xtick.color": TEXT_COLOR, "ytick.color": TEXT_COLOR,
    "text.color":  TEXT_COLOR, "grid.color":  GRID_COLOR,
    "grid.linewidth": 0.5,
    "font.family": "DejaVu Sans", "font.size": 9,
})


# ══════════════════════════════════════════════════════════════
#  TD-DFS 계산 함수 (extract_tddfs.py 와 동일 로직, 독립 실행 가능)
# ══════════════════════════════════════════════════════════════

def _compute_td_csi(csi: np.ndarray, delta_t: int) -> np.ndarray:
    return (csi[delta_t:] - csi[:-delta_t]).astype(np.complex64)


def _find_optimal_delta_t(csi: np.ndarray, dt_min: int, dt_max: int) -> int:
    best_dt, best_var = dt_min, np.inf
    for dt in range(dt_min, dt_max + 1):
        td = _compute_td_csi(csi, dt)
        z  = td.mean(axis=1)
        variance = float(np.var(np.angle(z)))
        if variance < best_var:
            best_var, best_dt = variance, dt
    return best_dt


def _estimate_velocity(csi: np.ndarray, delta_t: int, fc_hz: float):
    lam_m   = C_MPS / fc_hz
    td      = _compute_td_csi(csi, delta_t)
    M       = td.shape[0]
    A_ref   = np.abs(csi[:M]).mean(axis=1)
    TD_mean = td.mean(axis=1)
    TD_amp  = np.abs(TD_mean)
    direction = np.sign(np.sin(np.angle(TD_mean)))
    direction[direction == 0] = 1.0
    arg    = np.clip(TD_amp / (2.0 * A_ref + 1e-10), 0.0, 1.0)
    fd_abs = np.arcsin(arg) * FS / (np.pi * delta_t)
    fdop   = (fd_abs * direction).astype(np.float32)
    velocity = (fdop * lam_m / 2.0 * 100.0).astype(np.float32)
    return velocity, fdop


def _compute_tddfs(npz_path: str, dt_min: int, dt_max: int, fc_hz: float):
    """sanitization NPZ → (velocity [cm/s], fdop [Hz], best_dt)"""
    data    = np.load(npz_path)
    csi     = data["csi"].astype(np.complex64)
    best_dt = _find_optimal_delta_t(csi, dt_min, dt_max)
    velocity, fdop = _estimate_velocity(csi, best_dt, fc_hz)
    return velocity, fdop, best_dt


def load_tddfs(npz_path: str, dt_min: int, dt_max: int, fc_hz: float):
    """feature NPZ 우선 로드, 없으면 on-the-fly 계산."""
    fname     = os.path.basename(npz_path)
    feat_path = os.path.join(FEAT_DIR, fname)
    if os.path.exists(feat_path):
        d = np.load(feat_path)
        return (d["velocity"].astype(np.float32),
                d["fdop"].astype(np.float32),
                int(d["delta_t"]))
    return _compute_tddfs(npz_path, dt_min, dt_max, fc_hz)


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
#  PLOT 1: 속도 시계열 그리드  (big 10 상단 / small 10 하단)
# ══════════════════════════════════════════════════════════════

def plot_velocity_grid(big_files, small_files, dt_min, dt_max, fc_hz):
    ncols = max(len(big_files), len(small_files))
    fig   = plt.figure(figsize=(ncols * 2.5, 10), facecolor=BG_COLOR)
    fig.suptitle(
        f"TD-DFS Instantaneous Doppler Velocity  ─  Big (top) vs Small (bottom)  "
        f"|  Δt=[{dt_min},{dt_max}]",
        fontsize=13, fontweight="bold", color=TEXT_COLOR, y=0.99,
    )
    gs = gridspec.GridSpec(
        2, ncols, figure=fig,
        hspace=0.6, wspace=0.2,
        top=0.92, bottom=0.06, left=0.04, right=0.98,
    )

    for row, (files, color, name) in enumerate([
        (big_files,   BIG_COLOR,   "Big"),
        (small_files, SMALL_COLOR, "Small"),
    ]):
        for col, fp in enumerate(files):
            ax = fig.add_subplot(gs[row, col])
            try:
                vel, _, best_dt = load_tddfs(fp, dt_min, dt_max, fc_hz)
                t = np.arange(len(vel)) / FS

                # 속도 0 기준선
                ax.axhline(0, color=GRID_COLOR, lw=0.6, ls="--")

                # 양수(접근) / 음수(후퇴) 색상 분리
                pos_mask = vel >= 0
                ax.fill_between(t, 0, vel,  where=pos_mask,
                                color="#EF5350", alpha=0.7, lw=0)
                ax.fill_between(t, vel, 0, where=~pos_mask,
                                color="#42A5F5", alpha=0.7, lw=0)
                ax.plot(t, vel, color=color, lw=0.5, alpha=0.9)

                ax.set_title(
                    f"{os.path.basename(fp).split('_')[3][:6]}  Δt={best_dt}",
                    fontsize=6.5, color=color, pad=2,
                )
                ax.set_xticks([]); ax.set_yticks([])
                ax.tick_params(length=0)
                for sp in ax.spines.values():
                    sp.set_edgecolor(color); sp.set_linewidth(1.2)

            except Exception:
                ax.text(0.5, 0.5, "ERR", ha="center", va="center",
                        transform=ax.transAxes, color="red", fontsize=8)

        y_pos = 0.78 if row == 0 else 0.32
        fig.text(0.005, y_pos, name, fontsize=12, color=color,
                 fontweight="bold", va="center")

    # 공통 범례
    fig.text(0.5, 0.02, "Time [s]", ha="center", color=TEXT_COLOR, fontsize=9)
    fig.text(0.0, 0.5,  "Velocity [cm/s]", va="center", color=TEXT_COLOR,
             fontsize=9, rotation=90)
    patches = [
        mpatches.Patch(color="#EF5350", label="Approaching (+)"),
        mpatches.Patch(color="#42A5F5", label="Receding (−)"),
    ]
    fig.legend(handles=patches, loc="lower right", ncol=2,
               facecolor=PANEL_COLOR, edgecolor=GRID_COLOR,
               labelcolor=TEXT_COLOR, fontsize=8, bbox_to_anchor=(0.98, 0.01))

    out = os.path.join(OUT_DIR, "tddfs_velocity_grid.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  [saved] {out}")


# ══════════════════════════════════════════════════════════════
#  PLOT 2: 평균 속도 시계열 + ±1σ 밴드 오버레이
# ══════════════════════════════════════════════════════════════

def plot_velocity_compare(big_files, small_files, dt_min, dt_max, fc_hz):
    def stack_vel(files, target_len=None):
        vels = []
        for fp in files:
            try:
                vel, _, _ = load_tddfs(fp, dt_min, dt_max, fc_hz)
                vels.append(vel)
            except Exception:
                pass
        if not vels:
            return None
        min_len = min(len(v) for v in vels) if target_len is None else target_len
        return np.array([v[:min_len] for v in vels], dtype=np.float32)  # (M, T)

    big_arr   = stack_vel(big_files)
    small_arr = stack_vel(small_files)

    if big_arr is None or small_arr is None:
        print("  [SKIP] plot_velocity_compare"); return

    min_t = min(big_arr.shape[1], small_arr.shape[1])
    big_arr   = big_arr[:, :min_t]
    small_arr = small_arr[:, :min_t]
    t = np.arange(min_t) / FS

    fig, axes = plt.subplots(3, 1, figsize=(16, 12), facecolor=BG_COLOR,
                              gridspec_kw={"height_ratios": [3, 3, 2]})
    fig.suptitle(
        "TD-DFS Mean Velocity Comparison  ─  Big vs Small",
        fontsize=13, fontweight="bold", color=TEXT_COLOR, y=1.01,
    )

    # 패널 1: big 평균 속도
    ax = axes[0]
    ax.set_facecolor(PANEL_COLOR); ax.spines[:].set_color(GRID_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8); ax.grid(alpha=0.2)
    ax.axhline(0, color="white", lw=0.6, ls="--", alpha=0.4)
    for v in big_arr:
        ax.plot(t, v, color=BIG_COLOR, alpha=0.15, lw=0.6)
    mu, sigma = big_arr.mean(0), big_arr.std(0)
    ax.plot(t, mu, color=BIG_COLOR, lw=2.5, label=f"Big mean (n={len(big_arr)})")
    ax.fill_between(t, mu - sigma, mu + sigma, color=BIG_COLOR, alpha=0.15)
    ax.set_ylabel("Velocity [cm/s]", color=TEXT_COLOR, fontsize=9)
    ax.legend(facecolor=PANEL_COLOR, edgecolor=GRID_COLOR,
              labelcolor=TEXT_COLOR, fontsize=9)
    ax.set_title("Big Stride — Instantaneous Velocity", color=TEXT_COLOR, fontsize=10)

    # 패널 2: small 평균 속도
    ax = axes[1]
    ax.set_facecolor(PANEL_COLOR); ax.spines[:].set_color(GRID_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8); ax.grid(alpha=0.2)
    ax.axhline(0, color="white", lw=0.6, ls="--", alpha=0.4)
    for v in small_arr:
        ax.plot(t, v, color=SMALL_COLOR, alpha=0.15, lw=0.6)
    mu_s, sigma_s = small_arr.mean(0), small_arr.std(0)
    ax.plot(t, mu_s, color=SMALL_COLOR, lw=2.5, label=f"Small mean (n={len(small_arr)})")
    ax.fill_between(t, mu_s - sigma_s, mu_s + sigma_s, color=SMALL_COLOR, alpha=0.15)
    ax.set_ylabel("Velocity [cm/s]", color=TEXT_COLOR, fontsize=9)
    ax.legend(facecolor=PANEL_COLOR, edgecolor=GRID_COLOR,
              labelcolor=TEXT_COLOR, fontsize=9)
    ax.set_title("Small Stride — Instantaneous Velocity", color=TEXT_COLOR, fontsize=10)

    # 패널 3: 차이 (big평균 - small평균)
    ax = axes[2]
    ax.set_facecolor(PANEL_COLOR); ax.spines[:].set_color(GRID_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8); ax.grid(alpha=0.2)
    diff = mu - mu_s
    ax.axhline(0, color="white", lw=0.6, ls="--", alpha=0.4)
    ax.fill_between(t, diff, 0,
                    where=diff >= 0, color=BIG_COLOR,   alpha=0.5, label="Big > Small")
    ax.fill_between(t, diff, 0,
                    where=diff <  0, color=SMALL_COLOR, alpha=0.5, label="Small > Big")
    ax.plot(t, diff, color=TEXT_COLOR, lw=1.2)
    ax.set_xlabel("Time [s]",           color=TEXT_COLOR, fontsize=9)
    ax.set_ylabel("Δvel [cm/s]",        color=TEXT_COLOR, fontsize=9)
    ax.set_title("Difference (Big − Small mean velocity)", color=TEXT_COLOR, fontsize=10)
    ax.legend(facecolor=PANEL_COLOR, edgecolor=GRID_COLOR,
              labelcolor=TEXT_COLOR, fontsize=9)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "tddfs_velocity_compare.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  [saved] {out}")


# ══════════════════════════════════════════════════════════════
#  PLOT 3: 속도 절댓값 분포 (히스토그램 + KDE)
# ══════════════════════════════════════════════════════════════

def plot_speed_distribution(big_files, small_files, dt_min, dt_max, fc_hz):
    """
    |velocity| 분포를 히스토그램으로 표시 + 가우시안 커널 밀도 추정 오버레이.
    big vs small 분포 차이를 통해 보폭 크기와 도플러 속도의 상관관계 시각화.
    """
    def collect_speeds(files):
        speeds = []
        for fp in files:
            try:
                vel, _, _ = load_tddfs(fp, dt_min, dt_max, fc_hz)
                speeds.extend(np.abs(vel).tolist())
            except Exception:
                pass
        return np.array(speeds, dtype=np.float32)

    big_spd   = collect_speeds(big_files)
    small_spd = collect_speeds(small_files)

    if len(big_spd) == 0 or len(small_spd) == 0:
        print("  [SKIP] plot_speed_distribution"); return

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), facecolor=BG_COLOR)
    fig.suptitle(
        "TD-DFS Instantaneous Speed |v| Distribution  ─  Big vs Small",
        fontsize=13, fontweight="bold", color=TEXT_COLOR, y=1.01,
    )

    # 공통 x 범위
    all_spd = np.concatenate([big_spd, small_spd])
    x_max   = np.percentile(all_spd, 99)
    bins    = np.linspace(0, x_max, 60)

    # 패널 1: 히스토그램 오버레이
    ax = axes[0]
    ax.set_facecolor(PANEL_COLOR); ax.spines[:].set_color(GRID_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8); ax.grid(alpha=0.2)

    ax.hist(big_spd,   bins=bins, density=True, alpha=0.5,
            color=BIG_COLOR,   label=f"Big (n={len(big_spd)})")
    ax.hist(small_spd, bins=bins, density=True, alpha=0.5,
            color=SMALL_COLOR, label=f"Small (n={len(small_spd)})")

    # KDE
    try:
        from scipy.stats import gaussian_kde
        x_kde = np.linspace(0, x_max, 300)
        ax.plot(x_kde, gaussian_kde(big_spd)(x_kde),
                color=BIG_COLOR,   lw=2.5)
        ax.plot(x_kde, gaussian_kde(small_spd)(x_kde),
                color=SMALL_COLOR, lw=2.5)
    except ImportError:
        pass

    ax.axvline(big_spd.mean(),   color=BIG_COLOR,   lw=1.5, ls="--",
               label=f"Big mean = {big_spd.mean():.1f}")
    ax.axvline(small_spd.mean(), color=SMALL_COLOR, lw=1.5, ls="--",
               label=f"Small mean = {small_spd.mean():.1f}")

    ax.set_title("Speed Distribution (Histogram + KDE)", color=TEXT_COLOR, fontsize=10)
    ax.set_xlabel("|velocity| [cm/s]", color=TEXT_COLOR, fontsize=9)
    ax.set_ylabel("Density",            color=TEXT_COLOR, fontsize=9)
    ax.legend(facecolor=PANEL_COLOR, edgecolor=GRID_COLOR,
              labelcolor=TEXT_COLOR, fontsize=9)

    # 패널 2: 박스플롯 비교
    ax2 = axes[1]
    ax2.set_facecolor(PANEL_COLOR); ax2.spines[:].set_color(GRID_COLOR)
    ax2.tick_params(colors=TEXT_COLOR, labelsize=8); ax2.grid(alpha=0.2, axis="y")

    bp = ax2.boxplot(
        [big_spd, small_spd],
        tick_labels=["Big", "Small"],
        patch_artist=True,
        medianprops=dict(color="white", lw=2),
        whiskerprops=dict(color=TEXT_COLOR),
        capprops=dict(color=TEXT_COLOR),
        flierprops=dict(marker=".", color=GRID_COLOR, markersize=2, alpha=0.4),
    )
    bp["boxes"][0].set_facecolor(BIG_COLOR);   bp["boxes"][0].set_alpha(0.7)
    bp["boxes"][1].set_facecolor(SMALL_COLOR); bp["boxes"][1].set_alpha(0.7)

    ax2.set_title("Speed Box Plot", color=TEXT_COLOR, fontsize=10)
    ax2.set_ylabel("|velocity| [cm/s]", color=TEXT_COLOR, fontsize=9)
    for tick, color in zip(ax2.get_xticklabels(), [BIG_COLOR, SMALL_COLOR]):
        tick.set_color(color)
        tick.set_fontsize(11)
        tick.set_fontweight("bold")

    # 통계 주석
    for i, (arr, color) in enumerate([(big_spd, BIG_COLOR), (small_spd, SMALL_COLOR)], 1):
        ax2.text(i, arr.max() * 0.98,
                 f"μ={arr.mean():.1f}\nσ={arr.std():.1f}",
                 ha="center", va="top", fontsize=8, color=color)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "tddfs_speed_dist.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  [saved] {out}")


# ══════════════════════════════════════════════════════════════
#  PLOT 4: 최적 Δt 분포 + 속도 통계 요약
# ══════════════════════════════════════════════════════════════

def plot_delta_t_stats(big_files, small_files, dt_min, dt_max, fc_hz):
    """
    최적 Δt 히스토그램 (big/small 비교) + 클래스별 속도 통계 요약 테이블.
    Δt 분포로 신호 SNR 수준을 간접 확인할 수 있음.
    """
    big_dts, small_dts   = [], []
    big_stats, small_stats = [], []

    for fp, dt_list, stat_list in [
        (big_files,   big_dts,   big_stats),
        (small_files, small_dts, small_stats),
    ]:
        for f in fp:
            try:
                vel, _, best_dt = load_tddfs(f, dt_min, dt_max, fc_hz)
                dt_list.append(best_dt)
                stat_list.append({
                    "mean": float(np.mean(vel)),
                    "std":  float(np.std(vel)),
                    "abs_mean": float(np.mean(np.abs(vel))),
                    "abs_max":  float(np.max(np.abs(vel))),
                })
            except Exception:
                pass

    if not big_dts:
        print("  [SKIP] plot_delta_t_stats"); return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor=BG_COLOR)
    fig.suptitle(
        "TD-DFS Δt Distribution & Velocity Statistics Summary",
        fontsize=13, fontweight="bold", color=TEXT_COLOR, y=1.01,
    )

    # ── 패널 1: Δt 히스토그램
    ax = axes[0]
    ax.set_facecolor(PANEL_COLOR); ax.spines[:].set_color(GRID_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8); ax.grid(alpha=0.2)

    bins = np.arange(dt_min - 0.5, dt_max + 1.5, 1)
    ax.hist(big_dts,   bins=bins, alpha=0.7, color=BIG_COLOR,
            label=f"Big (n={len(big_dts)})",   edgecolor="none")
    ax.hist(small_dts, bins=bins, alpha=0.7, color=SMALL_COLOR,
            label=f"Small (n={len(small_dts)})", edgecolor="none")
    ax.set_title("Optimal Δt Distribution", color=TEXT_COLOR, fontsize=10)
    ax.set_xlabel("Optimal Δt [samples]",   color=TEXT_COLOR, fontsize=9)
    ax.set_ylabel("Count",                  color=TEXT_COLOR, fontsize=9)
    ax.set_xticks(range(dt_min, dt_max + 1))
    ax.legend(facecolor=PANEL_COLOR, edgecolor=GRID_COLOR,
              labelcolor=TEXT_COLOR, fontsize=9)

    # ── 패널 2: 통계 속성 1 — mean |v| 별 샘플 분포
    ax = axes[1]
    ax.set_facecolor(PANEL_COLOR); ax.spines[:].set_color(GRID_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8); ax.grid(alpha=0.2)

    big_abs_means   = [s["abs_mean"] for s in big_stats]
    small_abs_means = [s["abs_mean"] for s in small_stats]
    ax.scatter(range(len(big_abs_means)),   big_abs_means,
               color=BIG_COLOR,   s=50, alpha=0.8, label="Big", zorder=3)
    ax.scatter(range(len(small_abs_means)), small_abs_means,
               color=SMALL_COLOR, s=50, alpha=0.8, label="Small", zorder=3)
    ax.axhline(np.mean(big_abs_means),   color=BIG_COLOR,   lw=1.5, ls="--",
               alpha=0.7, label=f"Big avg={np.mean(big_abs_means):.1f}")
    ax.axhline(np.mean(small_abs_means), color=SMALL_COLOR, lw=1.5, ls="--",
               alpha=0.7, label=f"Small avg={np.mean(small_abs_means):.1f}")
    ax.set_title("Mean |velocity| per Sample", color=TEXT_COLOR, fontsize=10)
    ax.set_xlabel("Sample index",              color=TEXT_COLOR, fontsize=9)
    ax.set_ylabel("Mean |vel| [cm/s]",         color=TEXT_COLOR, fontsize=9)
    ax.legend(facecolor=PANEL_COLOR, edgecolor=GRID_COLOR,
              labelcolor=TEXT_COLOR, fontsize=8)

    # ── 패널 3: 속도 표준편차 비교
    ax = axes[2]
    ax.set_facecolor(PANEL_COLOR); ax.spines[:].set_color(GRID_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8); ax.grid(alpha=0.2)

    big_stds   = [s["std"] for s in big_stats]
    small_stds = [s["std"] for s in small_stats]

    bp = ax.boxplot(
        [big_stds, small_stds],
        tick_labels=["Big", "Small"],
        patch_artist=True,
        medianprops=dict(color="white", lw=2),
        whiskerprops=dict(color=TEXT_COLOR),
        capprops=dict(color=TEXT_COLOR),
    )
    bp["boxes"][0].set_facecolor(BIG_COLOR);   bp["boxes"][0].set_alpha(0.7)
    bp["boxes"][1].set_facecolor(SMALL_COLOR); bp["boxes"][1].set_alpha(0.7)

    ax.set_title("Velocity Std Dev per Sample\n(Variability of Stride Motion)",
                 color=TEXT_COLOR, fontsize=10)
    ax.set_ylabel("Std [cm/s]", color=TEXT_COLOR, fontsize=9)
    for tick, color in zip(ax.get_xticklabels(), [BIG_COLOR, SMALL_COLOR]):
        tick.set_color(color)
        tick.set_fontsize(11)
        tick.set_fontweight("bold")

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "tddfs_delta_t_stats.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  [saved] {out}")


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def _parse_args():
    p = argparse.ArgumentParser(description="TD-DFS Feature Visualization")
    p.add_argument("--delta_t_min", type=int,   default=DELTA_T_MIN)
    p.add_argument("--delta_t_max", type=int,   default=DELTA_T_MAX)
    p.add_argument("--fc_hz",       type=float, default=FC_HZ)
    p.add_argument("--n_samples",   type=int,   default=N_SAMPLES)
    p.add_argument("--sanit_dir",   default=SANIT_DIR)
    p.add_argument("--feat_dir",    default=FEAT_DIR)
    p.add_argument("--out_dir",     default=OUT_DIR)
    return p.parse_args()


def main():
    args = _parse_args()
    global SANIT_DIR, FEAT_DIR, OUT_DIR, N_SAMPLES
    SANIT_DIR, FEAT_DIR, OUT_DIR, N_SAMPLES = \
        args.sanit_dir, args.feat_dir, args.out_dir, args.n_samples
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"[TD-DFS Visualization] Loading from:\n  {SANIT_DIR}")
    big_files, small_files = collect_files()
    print(f"  Big={len(big_files)}, Small={len(small_files)}")

    if not big_files or not small_files:
        print("[ERROR] 파일 부족. SANIT_DIR 경로를 확인하세요.")
        sys.exit(1)

    print("\n[1/4] Velocity Grid...")
    plot_velocity_grid(big_files, small_files, args.delta_t_min, args.delta_t_max, args.fc_hz)

    print("[2/4] Velocity Compare...")
    plot_velocity_compare(big_files, small_files, args.delta_t_min, args.delta_t_max, args.fc_hz)

    print("[3/4] Speed Distribution...")
    plot_speed_distribution(big_files, small_files, args.delta_t_min, args.delta_t_max, args.fc_hz)

    print("[4/4] Δt Stats...")
    plot_delta_t_stats(big_files, small_files, args.delta_t_min, args.delta_t_max, args.fc_hz)

    print(f"\n[TD-DFS Visualization] All done → {OUT_DIR}")


if __name__ == "__main__":
    main()
