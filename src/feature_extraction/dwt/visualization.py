"""
DWT Feature Visualization
==========================
정제된 CSI 데이터에서 DWT 에너지 특징을 계산하여 시각화합니다.

저장 이미지
-----------
  dwt_energy_grid.png      : big/small 각 10개 샘플의 레벨별 에너지 바 그리드
  dwt_energy_compare.png   : big vs small 평균 에너지 프로파일 비교 (레벨 × PCA PC)
  dwt_waveform_denoise.png : 대표 1개 샘플의 PC1 시계열 — 원본/노이즈제거/재구성 비교
  dwt_feature_heatmap.png  : 10+10 샘플의 특징 벡터 히트맵 (클래스별 정렬)
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
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import pywt
from sklearn.decomposition import PCA

# ─── 경로 설정 ────────────────────────────────────────────────
HERE        = os.path.dirname(os.path.abspath(__file__))
BASE_DIR    = os.path.abspath(os.path.join(HERE, "..", ".."))
SANIT_DIR   = os.path.join(BASE_DIR, "data", "sanitization", "sanitization")
OUT_DIR     = HERE          # 이미지를 이 폴더에 저장

WAVELET     = "sym3"
LEVEL       = 10
N_PCA       = 6
N_SAMPLES   = 10            # big / small 각각
SEED        = 42

# ─── 스타일 ───────────────────────────────────────────────────
BIG_COLOR   = "#EF5350"     # 빨강 계열
SMALL_COLOR = "#42A5F5"     # 파랑 계열
BG_COLOR    = "#0D1117"
PANEL_COLOR = "#161B22"
TEXT_COLOR  = "#E6EDF3"
GRID_COLOR  = "#30363D"

plt.rcParams.update({
    "figure.facecolor"  : BG_COLOR,
    "axes.facecolor"    : PANEL_COLOR,
    "axes.edgecolor"    : GRID_COLOR,
    "axes.labelcolor"   : TEXT_COLOR,
    "xtick.color"       : TEXT_COLOR,
    "ytick.color"       : TEXT_COLOR,
    "text.color"        : TEXT_COLOR,
    "grid.color"        : GRID_COLOR,
    "grid.linewidth"    : 0.5,
    "font.family"       : "DejaVu Sans",
    "font.size"         : 9,
})


# ══════════════════════════════════════════════════════════════
# 특징 계산 함수 (extract_dwt.py 와 동일 로직, 독립 실행 가능)
# ══════════════════════════════════════════════════════════════
def _soft_threshold(coeffs, thr):
    return np.sign(coeffs) * np.maximum(np.abs(coeffs) - thr, 0.0)


def _dwt_denoise(sig, wavelet, level):
    max_lv = pywt.dwt_max_level(len(sig), wavelet)
    lv = min(level, max_lv)
    coeffs = pywt.wavedec(sig, wavelet, level=lv)
    sigma  = np.median(np.abs(coeffs[-1])) / 0.6745
    thr    = sigma * np.sqrt(2 * np.log(len(sig)))
    thrd   = [coeffs[0]] + [_soft_threshold(c, thr) for c in coeffs[1:]]
    rec    = pywt.waverec(thrd, wavelet)
    return rec[: len(sig)]


def _energy_features(sig, wavelet, level):
    max_lv = pywt.dwt_max_level(len(sig), wavelet)
    lv = min(level, max_lv)
    coeffs = pywt.wavedec(sig, wavelet, level=lv)
    energy = np.array([np.sum(c ** 2) for c in coeffs], dtype=np.float32)
    nrm = np.linalg.norm(energy)
    return energy / nrm if nrm > 1e-10 else energy


def load_features(npz_path, wavelet=WAVELET, level=LEVEL, n_pca=N_PCA):
    """sanitization NPZ → (feature_vec, pca_streams, raw_amplitude)"""
    data      = np.load(npz_path)
    csi       = data["csi"]
    amplitude = np.abs(csi).astype(np.float32)          # (N, shape[1])

    min_len = 2 ** (level + 1)
    if len(amplitude) < min_len:
        pad = np.zeros((min_len - amplitude.shape[0], amplitude.shape[1]), dtype=np.float32)
        amplitude = np.vstack([amplitude, pad])

    k = min(n_pca, amplitude.shape[0])
    pca    = PCA(n_components=k)
    streams = pca.fit_transform(amplitude).astype(np.float32)   # (N, K)

    feat_parts = []
    for ki in range(streams.shape[1]):
        sig      = streams[:, ki].astype(np.float64)
        denoised = _dwt_denoise(sig, wavelet, level)
        energy   = _energy_features(denoised, wavelet, level)
        feat_parts.append(energy)

    feature_vec = np.concatenate(feat_parts).astype(np.float32)
    return feature_vec, streams, amplitude


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
# PLOT 1: 에너지 그리드  (big 10 + small 10, 각 막대그래프)
# ══════════════════════════════════════════════════════════════
def plot_energy_grid(big_files, small_files):
    fig = plt.figure(figsize=(24, 14), facecolor=BG_COLOR)
    fig.suptitle(
        "DWT Level-wise Energy Feature  ─  Big (top) vs Small (bottom)",
        fontsize=14, fontweight="bold", color=TEXT_COLOR, y=0.98,
    )

    nrows, ncols = 4, 10          # 2행 big / 2행 small → 총 4행 × 10열
    gs = gridspec.GridSpec(
        nrows, ncols, figure=fig,
        hspace=0.6, wspace=0.3,
        top=0.92, bottom=0.06, left=0.04, right=0.98,
    )

    level_actual = LEVEL    # 실제 레벨 (파일마다 다를 수 있으나 같다고 가정)
    x_labels     = ["cA"] + [f"D{i}" for i in range(level_actual, 0, -1)]

    # 한 행씩: big (0,1), small (2,3) → 첫 번째 행에 라벨
    groups = [
        (big_files,    BIG_COLOR,   "Big",   0),
        (small_files,  SMALL_COLOR, "Small", 2),
    ]

    for files, color, group_name, start_row in groups:
        for col_idx, fp in enumerate(files[:10]):
            row = start_row + (col_idx // 10)   # col_idx 0~9 → row start_row
            # 실제로 한 그룹 10개를 두 행에 나누지 않고 1행에 꽉 채움
            # start_row를 row1/row3으로 사용, span 2행
            ax = fig.add_subplot(gs[start_row: start_row + 2, col_idx])

            try:
                feat, _, _ = load_features(fp)
                # feat = (K*(L+1),) → reshape (K, L+1)
                per_pc = feat.reshape(N_PCA, -1)          # (K, L+1)
                energy_avg = per_pc.mean(axis=0)           # (L+1,)
                x = np.arange(len(energy_avg))

                bars = ax.bar(x, energy_avg, color=color, alpha=0.85,
                              edgecolor="none", width=0.75)

                # 최고 에너지 레벨 강조
                peak = np.argmax(energy_avg)
                bars[peak].set_edgecolor("white")
                bars[peak].set_linewidth(1.5)

                ax.set_xticks(x)
                ax.set_xticklabels(
                    x_labels[: len(energy_avg)],
                    fontsize=6, rotation=60, ha="right"
                )
                ax.set_yticks([])
                ax.set_title(
                    os.path.basename(fp).split("_")[3][:6],
                    fontsize=7, color=TEXT_COLOR, pad=2,
                )
                ax.tick_params(length=0)
                ax.spines[:].set_visible(False)
            except Exception as e:
                ax.text(0.5, 0.5, f"ERR\n{str(e)[:20]}", ha="center",
                        va="center", transform=ax.transAxes, fontsize=6)

        # 그룹 라벨 (왼쪽 첫 번째 셀 위에)
        first_ax = fig.add_subplot(gs[start_row: start_row + 2, 0])
        first_ax.set_visible(False)

    # 수동 그룹 텍스트
    fig.text(0.02, 0.75, "BIG",   fontsize=13, color=BIG_COLOR,   fontweight="bold", va="center")
    fig.text(0.02, 0.30, "SMALL", fontsize=13, color=SMALL_COLOR, fontweight="bold", va="center")

    out = os.path.join(OUT_DIR, "dwt_energy_grid.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  [saved] {out}")


# ══════════════════════════════════════════════════════════════
# PLOT 2: Big vs Small 평균 에너지 프로파일 비교
# ══════════════════════════════════════════════════════════════
def plot_energy_compare(big_files, small_files):
    def get_energies(files):
        evecs = []
        for fp in files:
            try:
                feat, _, _ = load_features(fp)
                per_pc = feat.reshape(N_PCA, -1)
                evecs.append(per_pc)
            except Exception:
                pass
        return np.array(evecs) if evecs else None   # (M, K, L+1)

    big_e   = get_energies(big_files)    # (N, K, L+1)
    small_e = get_energies(small_files)

    if big_e is None or small_e is None:
        print("  [SKIP] plot_energy_compare: not enough data")
        return

    n_levels = big_e.shape[2]
    level_labels = ["cA"] + [f"D{i}" for i in range(LEVEL, 0, -1)]
    level_labels = level_labels[:n_levels]
    x = np.arange(n_levels)

    fig, axes = plt.subplots(
        1, N_PCA, figsize=(5 * N_PCA, 5),
        facecolor=BG_COLOR, sharey=False,
    )
    fig.suptitle(
        "DWT Average Energy Profile per PCA Component  ─  Big vs Small",
        fontsize=13, fontweight="bold", color=TEXT_COLOR, y=1.02,
    )

    for k, ax in enumerate(axes):
        big_mean  = big_e[:, k, :].mean(axis=0)
        big_std   = big_e[:, k, :].std(axis=0)
        small_mean = small_e[:, k, :].mean(axis=0)
        small_std  = small_e[:, k, :].std(axis=0)

        ax.set_facecolor(PANEL_COLOR)
        ax.spines[:].set_color(GRID_COLOR)
        ax.tick_params(colors=TEXT_COLOR, length=0)
        ax.set_title(f"PC {k+1}", fontsize=10, color=TEXT_COLOR)

        ax.bar(x - 0.18, big_mean,   0.33, color=BIG_COLOR,   alpha=0.85,
               label="Big",   yerr=big_std,   ecolor="#FF8A80", capsize=2)
        ax.bar(x + 0.18, small_mean, 0.33, color=SMALL_COLOR, alpha=0.85,
               label="Small", yerr=small_std, ecolor="#82B1FF", capsize=2)

        ax.set_xticks(x)
        ax.set_xticklabels(level_labels, rotation=60, ha="right", fontsize=7)
        ax.set_xlabel("Wavelet Level", color=TEXT_COLOR, fontsize=8)
        ax.set_ylabel("Normalized Energy", color=TEXT_COLOR, fontsize=8) if k == 0 else None
        ax.yaxis.set_tick_params(labelsize=7)
        ax.grid(axis="y", alpha=0.3)

        if k == 0:
            ax.legend(fontsize=8, facecolor=PANEL_COLOR, labelcolor=TEXT_COLOR,
                      edgecolor=GRID_COLOR)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "dwt_energy_compare.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  [saved] {out}")


# ══════════════════════════════════════════════════════════════
# PLOT 3: 대표 샘플 파형 — 원본 / 노이즈제거 / 차이 (PC1)
# ══════════════════════════════════════════════════════════════
def plot_waveform_denoise(big_files, small_files):
    fig, axes = plt.subplots(
        2, 3, figsize=(18, 8), facecolor=BG_COLOR,
    )
    fig.suptitle(
        "DWT Denoising on PC1 Amplitude Stream  ─  Representative Samples",
        fontsize=13, fontweight="bold", color=TEXT_COLOR, y=1.01,
    )

    pairs = [
        (big_files[0],   BIG_COLOR,   "Big"),
        (small_files[0], SMALL_COLOR, "Small"),
    ]

    for row, (fp, color, label) in enumerate(pairs):
        _, streams, _ = load_features(fp)
        sig      = streams[:, 0].astype(np.float64)  # PC1 시계열
        denoised = _dwt_denoise(sig, WAVELET, LEVEL)
        t        = np.arange(len(sig))

        titles  = ["Original PC1", "Denoised PC1", "Residual Noise"]
        signals = [sig, denoised, sig - denoised]
        colors  = [color, "#FFEB3B", GRID_COLOR]
        alphas  = [0.6, 1.0, 0.7]

        for col, (title_str, s, c, a) in enumerate(
            zip(titles, signals, colors, alphas)
        ):
            ax = axes[row][col]
            ax.set_facecolor(PANEL_COLOR)
            ax.spines[:].set_color(GRID_COLOR)
            ax.tick_params(colors=TEXT_COLOR, length=2, labelsize=8)

            ax.plot(t, s, color=c, lw=0.8, alpha=a)
            ax.set_title(f"{label}  ─  {title_str}", fontsize=9,
                         color=TEXT_COLOR, pad=4)
            ax.set_xlabel("Packet index", color=TEXT_COLOR, fontsize=8)
            ax.set_ylabel("Amplitude (a.u.)", color=TEXT_COLOR, fontsize=8) \
                if col == 0 else None
            ax.grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "dwt_waveform_denoise.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  [saved] {out}")


# ══════════════════════════════════════════════════════════════
# PLOT 4: 특징 벡터 히트맵 (big 10 + small 10, 정렬)
# ══════════════════════════════════════════════════════════════
def plot_feature_heatmap(big_files, small_files):
    feat_matrix = []
    row_labels  = []
    row_colors  = []

    for fp in big_files:
        try:
            feat, _, _ = load_features(fp)
            feat_matrix.append(feat)
            stem = os.path.basename(fp)
            row_labels.append(stem.split("_")[3][:8])
            row_colors.append(BIG_COLOR)
        except Exception:
            pass

    for fp in small_files:
        try:
            feat, _, _ = load_features(fp)
            feat_matrix.append(feat)
            stem = os.path.basename(fp)
            row_labels.append(stem.split("_")[3][:8])
            row_colors.append(SMALL_COLOR)
        except Exception:
            pass

    if not feat_matrix:
        print("  [SKIP] plot_feature_heatmap: no data")
        return

    mat = np.array(feat_matrix)   # (M, D)

    # 각 행 min-max 정규화
    mat_norm = (mat - mat.min(axis=1, keepdims=True))
    row_range = mat.max(axis=1, keepdims=True) - mat.min(axis=1, keepdims=True) + 1e-10
    mat_norm  = mat_norm / row_range

    n_rows, n_cols = mat_norm.shape
    fig, ax = plt.subplots(
        figsize=(max(16, n_cols // 3), max(6, n_rows * 0.55)),
        facecolor=BG_COLOR,
    )
    fig.suptitle(
        f"DWT Feature Vector Heatmap  ─  {len(big_files)} Big (top) + {len(small_files)} Small (bottom)",
        fontsize=13, fontweight="bold", color=TEXT_COLOR, y=1.01,
    )

    ax.set_facecolor(PANEL_COLOR)
    ax.spines[:].set_color(GRID_COLOR)
    ax.tick_params(colors=TEXT_COLOR, length=0)

    im = ax.imshow(mat_norm, aspect="auto", cmap="viridis",
                   interpolation="nearest", vmin=0, vmax=1)

    # y축 레이블 + 클래스 색상 텍스트
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=8)
    for i, (tick_lbl, rc) in enumerate(
        zip(ax.get_yticklabels(), row_colors)
    ):
        tick_lbl.set_color(rc)

    # x축 레이블: PC 경계 표시
    feat_per_pc = n_cols // N_PCA
    major_ticks = [i * feat_per_pc for i in range(N_PCA)]
    ax.set_xticks(major_ticks)
    ax.set_xticklabels([f"PC{i+1}" for i in range(N_PCA)], fontsize=8)

    # PC 경계선
    for xt in major_ticks[1:]:
        ax.axvline(xt - 0.5, color="#555555", lw=0.8, ls="--")

    # big/small 경계선
    separator = len(big_files) - 0.5
    ax.axhline(separator, color="white", lw=1.5, ls="-")

    cbar = fig.colorbar(im, ax=ax, fraction=0.015, pad=0.01)
    cbar.ax.tick_params(colors=TEXT_COLOR, labelsize=7)
    cbar.set_label("Normalized Energy", color=TEXT_COLOR, fontsize=8)

    # 범례 패치
    import matplotlib.patches as mpatches
    patches = [
        mpatches.Patch(color=BIG_COLOR,   label="Big"),
        mpatches.Patch(color=SMALL_COLOR, label="Small"),
    ]
    ax.legend(handles=patches, loc="upper right",
              facecolor=PANEL_COLOR, edgecolor=GRID_COLOR,
              labelcolor=TEXT_COLOR, fontsize=9)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "dwt_feature_heatmap.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  [saved] {out}")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main():
    print(f"[DWT Visualization] Loading samples from:\n  {SANIT_DIR}")
    big_files, small_files = collect_files()
    print(f"  Big={len(big_files)}, Small={len(small_files)}")

    if not big_files or not small_files:
        print("[ERROR] Not enough files. Check SANIT_DIR.")
        sys.exit(1)

    print("\n[1/4] Energy Grid...")
    plot_energy_grid(big_files, small_files)

    print("[2/4] Energy Compare...")
    plot_energy_compare(big_files, small_files)

    print("[3/4] Waveform Denoise...")
    plot_waveform_denoise(big_files, small_files)

    print("[4/4] Feature Heatmap...")
    plot_feature_heatmap(big_files, small_files)

    print(f"\n[DWT Visualization] All done → {OUT_DIR}")


if __name__ == "__main__":
    main()
