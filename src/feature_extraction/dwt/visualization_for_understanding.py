# =============================================================================
# DWT Feature Extraction 파이프라인 시각화
#
# 이 스크립트는 `extract_dwt.py`에 구현된 파이프라인의 핵심 진행 단계와
# 각 단계의 데이터를 시각적으로 직관적으로 이해하기 위해 작성되었습니다.
#
# **전체 파이프라인 진행 및 시각화 단계:**
# 1. **진폭 단계**: NPZ 파일 데이터 로드 → `CSI` 복소수에서 **Amplitude(진폭)** 만 추출
# 2. **PCA 압축**: 108개의 서브캐리어 차원을 시계열 특성이 통합된 상위 K개의 PC로 축소
# 3. **DWT + Soft Thresholding**: big/small 레이블별 잡음 제거 비교
#    3-1. Highest Frequency Detail 계수 Soft Thresholding 전후 비교
#    3-2. Approximation + Detail Level 1~7 웨이블릿 분해 계수 시각화
#    3-3. PC 1 Original vs Denoised 비교 (Big vs Small)
#    3-4. PC 1 Denoised Overlay (Big vs Small)
#    3-5. PC 2~6 Original vs Denoised 비교
# 4. **Energy Extraction**: big/small 레이블별 에너지 분포 비교
#    4-1. PC 1 에너지 분포
#    4-2. PC 1 에너지 비교 (겹침 바 그래프)
#    4-3. PC 2~6 에너지 분포
# 5. **통합 히트맵**: big/small 레이블별 전체 주성분 에너지 히트맵 비교
# 6. **다수 샘플 평균 에너지**: big/small 전체 파일 평균 에너지 분포
# =============================================================================

import os
import glob
import argparse
import warnings
import numpy as np
import matplotlib.pyplot as plt
import pywt
from sklearn.decomposition import PCA
from datetime import datetime

# 전역 폰트 및 시각화 기본 설정 (Windows 한글 호환)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (10, 5)
plt.rcParams['axes.grid'] = True

# ──────────────────────────────────────────
# 경고 필터링
# ──────────────────────────────────────────
warnings.filterwarnings('ignore', message='.*Mean of empty slice.*')
warnings.filterwarnings('ignore', category=RuntimeWarning)

# =============================================================================
# 설정
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
SANIT_ROOT = os.path.join(PROJECT_ROOT, 'data', 'sanitization', 'sanitization')

# 결과 저장 경로: feature_extraction/result/dwt/yyyymmdd/
TODAY = datetime.now().strftime('%Y%m%d')
RESULT_DIR = os.path.join(PROJECT_ROOT, 'feature_extraction', 'result', 'dwt', TODAY)

# DWT 파라미터
WAVELET = 'sym3'
LEVEL = 10
N_COMPONENTS = 6

# 전역 변수로 선언 (명령줄 인자로 재설정 가능)
_SANIT_ROOT = SANIT_ROOT
_RESULT_DIR = RESULT_DIR

fig_counter = 0  # 그래프 저장 순서 카운터


def save_fig(fig, name):
    """그래프를 _RESULT_DIR에 저장하고 화면에도 표시"""
    global fig_counter
    fig_counter += 1
    filepath = os.path.join(_RESULT_DIR, f'{fig_counter:02d}_{name}.png')
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f'  → 저장: {filepath}')
    plt.show()


def soft_threshold(coeffs, threshold):
    """Soft Thresholding 함수"""
    return np.sign(coeffs) * np.maximum(np.abs(coeffs) - threshold, 0.0)


def extract_label(filename):
    """파일명에서 big/small 레이블 추출 (예: csi_..._small.npz → small)"""
    basename = os.path.splitext(os.path.basename(filename))[0]
    if basename.endswith('_big'):
        return 'big'
    elif basename.endswith('_small'):
        return 'small'
    return 'unknown'


def load_and_extract_amplitude(filepath):
    """NPZ 파일에서 CSI 복소수 → 진폭(Amplitude) 추출"""
    data = np.load(filepath)
    csi_complex = data['csi']
    return np.abs(csi_complex).astype(np.float32)


def apply_pca(amplitude, n_components):
    """PCA 적용하여 차원 축소"""
    pca = PCA(n_components=min(n_components, amplitude.shape[0]))
    pca_streams = pca.fit_transform(amplitude)
    return pca, pca_streams


def denoise_signal(signal_1d, wavelet, actual_level):
    """DWT + Soft Thresholding으로 신호 잡음 제거"""
    coeffs = pywt.wavedec(signal_1d, wavelet, level=actual_level)
    highest_detail = coeffs[-1]
    sigma = np.median(np.abs(highest_detail)) / 0.6745
    thr = sigma * np.sqrt(2 * np.log(len(signal_1d)))
    thresholded_coeffs = [coeffs[0]] + [soft_threshold(c, thr) for c in coeffs[1:]]
    denoised = pywt.waverec(thresholded_coeffs, wavelet)[:len(signal_1d)]
    return coeffs, thresholded_coeffs, denoised, thr


def compute_energy_vector(denoised_signal, wavelet, actual_level):
    """잡음 제거된 신호에서 에너지 벡터 추출"""
    final_coeffs = pywt.wavedec(denoised_signal, wavelet, level=actual_level)
    energy_vec = np.array([np.sum(c ** 2) for c in final_coeffs], dtype=np.float32)
    norm_val = np.linalg.norm(energy_vec)
    if norm_val > 1e-10:
        energy_vec = energy_vec / norm_val
    return energy_vec


def compute_full_energy_matrix(pca_streams, n_components, wavelet, actual_level):
    """모든 PC에 대해 에너지 행렬 계산"""
    energy_matrix = np.zeros((n_components, actual_level + 1))
    for k in range(n_components):
        sig = pca_streams[:, k].astype(np.float64)
        _, _, denoised, _ = denoise_signal(sig, wavelet, actual_level)
        e_vec = compute_energy_vector(denoised, wavelet, actual_level)
        energy_matrix[k, :] = e_vec
    return energy_matrix


# =============================================================================
# 메인 실행
# =============================================================================
def main():
    global _SANIT_ROOT, _RESULT_DIR, fig_counter

    parser = argparse.ArgumentParser(description="DWT Feature Extraction 파이프라인 시각화")
    parser.add_argument("--sanit-dir", type=str, default=SANIT_ROOT,
                       help=f"Sanitization NPZ 디렉토리 (default: {SANIT_ROOT})")
    parser.add_argument("--out-dir", type=str, default=RESULT_DIR,
                       help=f"출력 디렉토리 (default: {RESULT_DIR})")
    args = parser.parse_args()

    _SANIT_ROOT = args.sanit_dir
    _RESULT_DIR = args.out_dir
    fig_counter = 0

    os.makedirs(_RESULT_DIR, exist_ok=True)

    print(f'결과 저장 경로: {_RESULT_DIR}')
    print('=' * 60)

    # NPZ 파일 목록 수집
    npz_files = sorted(glob.glob(os.path.join(_SANIT_ROOT, '*.npz')))
    if not npz_files:
        print(f'경고: {_SANIT_ROOT} 에서 npz 파일을 찾을 수 없습니다.')
        return

    # big / small 분류
    big_files = [f for f in npz_files if extract_label(f) == 'big']
    small_files = [f for f in npz_files if extract_label(f) == 'small']
    print(f'전체 파일 수: {len(npz_files)} (big: {len(big_files)}, small: {len(small_files)})')

    if not big_files or not small_files:
        print('경고: big 또는 small 레이블 파일이 없습니다.')
        return

    # 대표 파일 1개씩 선택 (중간 인덱스)
    big_sample = big_files[len(big_files) // 2]
    small_sample = small_files[len(small_files) // 2]
    print(f'Big   대표 파일: {os.path.basename(big_sample)}')
    print(f'Small 대표 파일: {os.path.basename(small_sample)}')
    print('=' * 60)

    # =========================================================================
    # Step 1: 데이터 로드 및 진폭 추출 (대표 1개만 – 전체 구조 파악용)
    # =========================================================================
    print('\n[ Step 1 ] 데이터 로드 및 진폭(Amplitude) 추출')

    amp_big = load_and_extract_amplitude(big_sample)
    amp_small = load_and_extract_amplitude(small_sample)
    print(f'  Big   amplitude shape: {amp_big.shape}')
    print(f'  Small amplitude shape: {amp_small.shape}')

    # 히트맵 (대표 1개만 표시)
    fig, axes = plt.subplots(1, 2, figsize=(16, 4))
    for ax, amp, label in zip(axes, [amp_big, amp_small], ['Big', 'Small']):
        im = ax.imshow(amp.T, aspect='auto', cmap='viridis', origin='lower')
        ax.set_title(f'Step 1: Raw Amplitude Heatmap [{label}]')
        ax.set_xlabel('Packet Index (Time)')
        ax.set_ylabel('Subcarrier Index')
        fig.colorbar(im, ax=ax, label='Amplitude')
    fig.tight_layout()
    save_fig(fig, 'step1_amplitude_heatmap')

    # 시간축 신호 비교
    fig, axes = plt.subplots(2, 1, figsize=(14, 5), sharex=False)
    for ax, amp, label in zip(axes, [amp_big, amp_small], ['Big', 'Small']):
        ax.plot(amp[:, 0], label='Subcarrier 0', alpha=0.8)
        ax.plot(amp[:, 50], label='Subcarrier 50', alpha=0.8)
        ax.plot(amp[:, 107], label='Subcarrier 107', alpha=0.8)
        ax.set_title(f'Time-series Fluctuation [{label}]')
        ax.set_xlabel('Packet Index')
        ax.legend(loc='upper right')
    fig.tight_layout()
    save_fig(fig, 'step1_timeseries')

    # =========================================================================
    # Step 2: PCA 압축 (대표 1개만 – 구조 파악)
    # =========================================================================
    print('\n[ Step 2 ] PCA 압축')

    pca_big, streams_big = apply_pca(amp_big, N_COMPONENTS)
    pca_small, streams_small = apply_pca(amp_small, N_COMPONENTS)

    print(f'  Big   PCA shape: {streams_big.shape}, 분산비율: {pca_big.explained_variance_ratio_}')
    print(f'  Small PCA shape: {streams_small.shape}, 분산비율: {pca_small.explained_variance_ratio_}')

    fig, axes = plt.subplots(N_COMPONENTS, 2, figsize=(16, 1.6 * N_COMPONENTS), sharex=False)
    for col, (streams, pca_obj, label) in enumerate(
            [(streams_big, pca_big, 'Big'), (streams_small, pca_small, 'Small')]):
        for i in range(N_COMPONENTS):
            axes[i, col].plot(streams[:, i], color=f'C{i}')
            axes[i, col].set_title(f'[{label}] PC {i+1} ({pca_obj.explained_variance_ratio_[i]:.2%})')
        axes[-1, col].set_xlabel('Packet Index (Time)')
    fig.tight_layout()
    save_fig(fig, 'step2_pca_streams')

    # =========================================================================
    # Step 3: DWT + Soft Thresholding – Big vs Small 비교
    # =========================================================================
    print('\n[ Step 3 ] DWT + Soft Thresholding (Big vs Small 비교)')

    # 모든 PC에 대해 denoise 수행 (PC 1~6)
    all_results = {}  # {label: {pc_idx: {...}}}
    for streams, label in [(streams_big, 'Big'), (streams_small, 'Small')]:
        all_results[label] = {}
        for pc_idx in range(N_COMPONENTS):
            signal_1d = streams[:, pc_idx].astype(np.float64)
            actual_level = min(LEVEL, pywt.dwt_max_level(len(signal_1d), WAVELET))
            coeffs, thresholded_coeffs, denoised, thr = denoise_signal(
                signal_1d, WAVELET, actual_level)
            all_results[label][pc_idx] = {
                'signal': signal_1d, 'coeffs': coeffs,
                'thresholded': thresholded_coeffs,
                'denoised': denoised, 'threshold': thr,
                'actual_level': actual_level
            }
        print(f'  [{label}] PC 1~{N_COMPONENTS} denoise 완료 '
              f'(Level = {all_results[label][0]["actual_level"]}, '
              f'Threshold PC1 = {all_results[label][0]["threshold"]:.4f})')

    # PC 1 결과를 기존 results 호환용으로 참조
    results = {label: all_results[label][0] for label in ['Big', 'Small']}

    # 3-1. Threshold 전후 계수 비교 (PC 1)
    fig, axes = plt.subplots(1, 2, figsize=(16, 4))
    for ax, label in zip(axes, ['Big', 'Small']):
        r = results[label]
        ax.plot(r['coeffs'][-1], label='Original Detail (Noisy)', alpha=0.7)
        ax.plot(r['thresholded'][-1], label='Thresholded Coeffs', alpha=0.9)
        ax.axhline(r['threshold'], color='red', ls='--', lw=1,
                   label=f'Threshold ±{r["threshold"]:.2f}')
        ax.axhline(-r['threshold'], color='red', ls='--', lw=1)
        ax.set_title(f'Soft Thresholding [{label}] (PC 1)')
        ax.legend(fontsize=8)
    fig.suptitle('Step 3-1: Highest Frequency Detail Coefficients – Big vs Small',
                 fontsize=13, y=1.02)
    fig.tight_layout()
    save_fig(fig, 'step3_1_threshold_coeffs')

    # 3-2. 웨이블릿 분해 계수 시각화 (Approximation + Detail Levels)
    #      Big / Small 각각의 PC 1에 대해 모든 레벨의 계수를 subplot으로 표시
    for label in ['Big', 'Small']:
        r = results[label]
        al = r['actual_level']
        coeffs = r['coeffs']

        # 표시할 레벨 수 결정 (Approx + Detail 최대 7레벨 = 최대 8행)
        n_display = min(al, 7)  # Detail은 최대 7레벨까지 표시
        n_rows = n_display + 1  # Approximation 포함

        fig, axes = plt.subplots(n_rows, 1, figsize=(16, 2.0 * n_rows), sharex=False)

        # Approximation coefficients (cA)
        axes[0].plot(coeffs[0], color='royalblue', linewidth=1.0)
        axes[0].set_title(f'Approximation (cA{al})', fontsize=10)
        axes[0].set_ylabel('Amplitude')

        # Detail coefficients (cD): coeffs[1] = cD_level_al, ..., coeffs[al] = cD_1
        for i in range(1, n_display + 1):
            detail_level = al - i + 1  # 실제 레벨 번호
            axes[i].plot(coeffs[i], color=f'C{i}', linewidth=0.8)
            axes[i].set_title(f'Detail Level {detail_level} (cD{detail_level})', fontsize=10)
            axes[i].set_ylabel('Amplitude')

        axes[-1].set_xlabel('Coefficient Index')
        fig.suptitle(f'Step 3-2: Wavelet Decomposition Coefficients [{label}] (PC 1)\n'
                     f'Wavelet={WAVELET}, Decomposition Level={al}',
                     fontsize=13, y=1.01)
        fig.tight_layout()
        save_fig(fig, f'step3_2_wavelet_decomp_{label.lower()}')

    # 3-3. 잡음 제거 전후 파형 비교 (PC 1)
    fig, axes = plt.subplots(2, 1, figsize=(14, 6))
    for ax, label, color in zip(axes, ['Big', 'Small'], ['tab:blue', 'tab:orange']):
        r = results[label]
        ax.plot(r['signal'], color='gray', alpha=0.4, label='Original (Noisy)')
        ax.plot(r['denoised'], color=color, linewidth=1.5, label='Denoised')
        ax.set_title(f'Original vs Denoised PC 1 [{label}]')
        ax.set_xlabel('Packet Index')
        ax.set_ylabel('Amplitude')
        ax.legend(loc='upper right')
    fig.suptitle('Step 3-3: Wavelet Denoising Comparison – Big vs Small (PC 1)',
                 fontsize=13, y=1.01)
    fig.tight_layout()
    save_fig(fig, 'step3_3_denoised_pc1')

    # 3-4. 같은 축에 오버레이하여 직접 비교 (PC 1)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(results['Big']['denoised'], color='tab:blue', linewidth=1.5,
            alpha=0.8, label='Big (Denoised PC 1)')
    ax.plot(results['Small']['denoised'], color='tab:orange', linewidth=1.5,
            alpha=0.8, label='Small (Denoised PC 1)')
    ax.set_title('Step 3-4: Denoised Signal Overlay – Big vs Small (PC 1)')
    ax.set_xlabel('Packet Index')
    ax.set_ylabel('Amplitude')
    ax.legend()
    fig.tight_layout()
    save_fig(fig, 'step3_4_overlay_pc1')

    # 3-5. PC 2~6 Original vs Denoised 비교
    #      각 PC를 Big/Small 나란히 비교
    for pc_idx in range(1, N_COMPONENTS):  # pc_idx 1~5 → PC 2~6
        pc_num = pc_idx + 1
        fig, axes = plt.subplots(2, 1, figsize=(14, 6))
        for ax, label, color in zip(axes, ['Big', 'Small'],
                                    ['tab:blue', 'tab:orange']):
            r = all_results[label][pc_idx]
            ax.plot(r['signal'], color='gray', alpha=0.4, label='Original (Noisy)')
            ax.plot(r['denoised'], color=color, linewidth=1.5, label='Denoised')
            ax.set_title(f'Original vs Denoised PC {pc_num} [{label}]')
            ax.set_xlabel('Packet Index')
            ax.set_ylabel('Amplitude')
            ax.legend(loc='upper right')
        fig.suptitle(f'Step 3-5: Wavelet Denoising – Big vs Small (PC {pc_num})',
                     fontsize=13, y=1.01)
        fig.tight_layout()
        save_fig(fig, f'step3_5_denoised_pc{pc_num}')

    # =========================================================================
    # Step 4: Energy Distribution – Big vs Small 비교
    # =========================================================================
    print('\n[ Step 4 ] Energy Distribution (Big vs Small 비교)')

    # 4-1. PC 1 에너지 분포 (나란히 바 그래프)
    energy_results = {}
    for label in ['Big', 'Small']:
        r = results[label]
        e_vec = compute_energy_vector(r['denoised'], WAVELET, r['actual_level'])
        energy_results[label] = {'energy_vec': e_vec, 'actual_level': r['actual_level']}

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for ax, label, color_main in zip(axes, ['Big', 'Small'],
                                     ['royalblue', 'darkorange']):
        er = energy_results[label]
        al = er['actual_level']
        bar_labels = ['Approx(cA)'] + [f'cD L{al - i + 1}' for i in range(1, al + 1)]
        colors = [color_main] + ['mediumseagreen'] * al
        bars = ax.bar(bar_labels, er['energy_vec'], color=colors)
        ax.set_title(f'Energy Distribution [{label}] (PC 1)')
        ax.set_ylabel('Normalized Energy')
        ax.tick_params(axis='x', rotation=45)
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, yval, f'{yval:.3f}',
                    va='bottom', ha='center', fontsize=8)
    fig.suptitle('Step 4-1: Energy Distribution – Big vs Small (PC 1)',
                 fontsize=13, y=1.01)
    fig.tight_layout()
    save_fig(fig, 'step4_1_energy_bars_pc1')

    # 4-2. PC 1 겹침 바 그래프로 직접 차이 시각화
    al = min(energy_results['Big']['actual_level'],
             energy_results['Small']['actual_level'])
    bar_labels = ['Approx(cA)'] + [f'cD L{al - i + 1}' for i in range(1, al + 1)]
    x = np.arange(len(bar_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    bars1 = ax.bar(x - width / 2, energy_results['Big']['energy_vec'][:al + 1],
                   width, label='Big', color='royalblue', alpha=0.8)
    bars2 = ax.bar(x + width / 2, energy_results['Small']['energy_vec'][:al + 1],
                   width, label='Small', color='darkorange', alpha=0.8)
    ax.set_title('Step 4-2: Energy Distribution Comparison – Big vs Small (PC 1)')
    ax.set_ylabel('Normalized Energy')
    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels, rotation=45)
    ax.legend()
    # 수치 표시
    for bars in [bars1, bars2]:
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, yval, f'{yval:.3f}',
                    va='bottom', ha='center', fontsize=7)
    fig.tight_layout()
    save_fig(fig, 'step4_2_energy_comparison_pc1')

    # 4-3. PC 2~6 에너지 분포 비교 (각 PC별 Big vs Small 겹침 바 그래프)
    for pc_idx in range(1, N_COMPONENTS):  # pc_idx 1~5 → PC 2~6
        pc_num = pc_idx + 1

        pc_energy = {}
        for label in ['Big', 'Small']:
            r = all_results[label][pc_idx]
            e_vec = compute_energy_vector(r['denoised'], WAVELET, r['actual_level'])
            pc_energy[label] = {'energy_vec': e_vec, 'actual_level': r['actual_level']}

        al = min(pc_energy['Big']['actual_level'],
                 pc_energy['Small']['actual_level'])
        bar_labels = ['Approx(cA)'] + [f'cD L{al - i + 1}' for i in range(1, al + 1)]
        x = np.arange(len(bar_labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 5))
        bars1 = ax.bar(x - width / 2,
                       pc_energy['Big']['energy_vec'][:al + 1],
                       width, label='Big', color='royalblue', alpha=0.8)
        bars2 = ax.bar(x + width / 2,
                       pc_energy['Small']['energy_vec'][:al + 1],
                       width, label='Small', color='darkorange', alpha=0.8)
        ax.set_title(f'Step 4-3: Energy Distribution Comparison – '
                     f'Big vs Small (PC {pc_num})')
        ax.set_ylabel('Normalized Energy')
        ax.set_xticks(x)
        ax.set_xticklabels(bar_labels, rotation=45)
        ax.legend()
        for bars in [bars1, bars2]:
            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2.0, yval, f'{yval:.3f}',
                        va='bottom', ha='center', fontsize=7)
        fig.tight_layout()
        save_fig(fig, f'step4_3_energy_comparison_pc{pc_num}')

    # =========================================================================
    # Step 5: 전체 주성분 에너지 히트맵 – Big vs Small 비교
    # =========================================================================
    print('\n[ Step 5 ] 전체 주성분 에너지 히트맵 (Big vs Small 비교)')

    heatmap_results = {}
    for streams, label in [(streams_big, 'Big'), (streams_small, 'Small')]:
        sig_1d = streams[:, 0].astype(np.float64)
        al = min(LEVEL, pywt.dwt_max_level(len(sig_1d), WAVELET))
        energy_matrix = compute_full_energy_matrix(streams, N_COMPONENTS, WAVELET, al)
        heatmap_results[label] = {'matrix': energy_matrix, 'actual_level': al}
        flat = energy_matrix.flatten()
        print(f'  [{label}] 최종 특징 벡터 Shape: {flat.shape} = '
              f'(PC {N_COMPONENTS}개 × 대역 {al + 1}개)')

    # 나란히 히트맵
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    for ax, label, cmap in zip(axes, ['Big', 'Small'], ['plasma', 'inferno']):
        hr = heatmap_results[label]
        al = hr['actual_level']
        im = ax.imshow(hr['matrix'].T, aspect='auto', cmap=cmap)
        for (j, i), val in np.ndenumerate(hr['matrix'].T):
            ax.text(i, j, f'{val:.2f}', ha='center', va='center',
                    color='white' if val < 0.6 else 'black', fontsize=8)
        fig.colorbar(im, ax=ax, label='Normalized Energy')
        ax.set_title(f'Energy Heatmap [{label}]')
        ax.set_xlabel('Principal Component')
        ax.set_ylabel('DWT Level')
        ytl = ['Approx'] + [f'L{al - i + 1} Detail' for i in range(1, al + 1)]
        ax.set_yticks(np.arange(al + 1))
        ax.set_yticklabels(ytl)
        ax.set_xticks(range(N_COMPONENTS))
        ax.set_xticklabels([f'PC {i+1}' for i in range(N_COMPONENTS)])
    fig.suptitle('Step 5: Consolidated Energy Heatmaps – Big vs Small',
                 fontsize=14, y=1.01)
    fig.tight_layout()
    save_fig(fig, 'step5_energy_heatmaps')

    # 차이 히트맵 (Big − Small)
    al = min(heatmap_results['Big']['actual_level'],
             heatmap_results['Small']['actual_level'])
    diff_matrix = (heatmap_results['Big']['matrix'][:, :al + 1]
                   - heatmap_results['Small']['matrix'][:, :al + 1])

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(diff_matrix.T, aspect='auto', cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    for (j, i), val in np.ndenumerate(diff_matrix.T):
        ax.text(i, j, f'{val:+.3f}', ha='center', va='center',
                color='white' if abs(val) > 0.25 else 'black', fontsize=9)
    fig.colorbar(im, ax=ax, label='Energy Difference (Big − Small)')
    ax.set_title('Step 5: Energy Difference Heatmap (Big − Small)')
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('DWT Level')
    ytl = ['Approx'] + [f'L{al - i + 1} Detail' for i in range(1, al + 1)]
    ax.set_yticks(np.arange(al + 1))
    ax.set_yticklabels(ytl)
    ax.set_xticks(range(N_COMPONENTS))
    ax.set_xticklabels([f'PC {i+1}' for i in range(N_COMPONENTS)])
    fig.tight_layout()
    save_fig(fig, 'step5_energy_diff_heatmap')

    # =========================================================================
    # Step 6 (보너스): 다수 샘플 평균 에너지 비교
    # 대표 1개가 아닌, big/small 전체 파일의 평균 에너지 분포를 비교합니다.
    # =========================================================================
    print('\n[ Step 6 ] 다수 샘플 평균 에너지 비교')
    MAX_SAMPLES = 30  # 속도를 위해 각 레이블에서 최대 N개 샘플링

    avg_energy = {}
    for label, files in [('Big', big_files), ('Small', small_files)]:
        sampled = files[::max(1, len(files) // MAX_SAMPLES)][:MAX_SAMPLES]
        all_energies = []
        for fpath in sampled:
            amp = load_and_extract_amplitude(fpath)
            _, streams = apply_pca(amp, N_COMPONENTS)
            sig_1d = streams[:, 0].astype(np.float64)
            al = min(LEVEL, pywt.dwt_max_level(len(sig_1d), WAVELET))
            _, _, denoised, _ = denoise_signal(sig_1d, WAVELET, al)
            e_vec = compute_energy_vector(denoised, WAVELET, al)
            all_energies.append(e_vec[:al + 1])
        # 패딩 후 평균
        min_len = min(len(e) for e in all_energies)
        trimmed = np.array([e[:min_len] for e in all_energies])
        avg_energy[label] = {
            'mean': trimmed.mean(axis=0),
            'std': trimmed.std(axis=0),
            'n_samples': len(sampled),
            'level': min_len - 1
        }
        print(f'  [{label}] {len(sampled)}개 샘플 평균 계산 완료')

    al = min(avg_energy['Big']['level'], avg_energy['Small']['level'])
    bar_labels = ['Approx(cA)'] + [f'cD L{al - i + 1}' for i in range(1, al + 1)]
    x = np.arange(len(bar_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width / 2, avg_energy['Big']['mean'][:al + 1], width,
           yerr=avg_energy['Big']['std'][:al + 1], capsize=3,
           label=f'Big (n={avg_energy["Big"]["n_samples"]})',
           color='royalblue', alpha=0.8)
    ax.bar(x + width / 2, avg_energy['Small']['mean'][:al + 1], width,
           yerr=avg_energy['Small']['std'][:al + 1], capsize=3,
           label=f'Small (n={avg_energy["Small"]["n_samples"]})',
           color='darkorange', alpha=0.8)
    ax.set_title('Step 6: Average Energy Distribution (PC 1) – Big vs Small')
    ax.set_ylabel('Normalized Energy')
    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels, rotation=45)
    ax.legend()
    fig.tight_layout()
    save_fig(fig, 'step6_avg_energy_comparison')

    print('\n' + '=' * 60)
    print(f'모든 시각화 완료! 저장 위치: {_RESULT_DIR}')
    print(f'총 {fig_counter}개 그래프 저장됨')


if __name__ == '__main__':
    main()
