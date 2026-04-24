# Ablation Study - Parameters & Hyperparameters Analysis
## WiFi CSI Stride Classification Pipeline

**Project**: WiFi CSI 기반 보폭(Stride) 크기 분류 (big/small)  
**마지막 업데이트**: 2026-04-24  
**분석 범위**: Preprocessing, Sliding Window, Sanitization, Feature Extraction, Learning 전체 파이프라인

---

## 📋 Executive Summary

프로젝트는 총 **5개의 주요 모듈**로 구성되며, 최근 슬라이딩 윈도우 기반의 데이터 증강 및 실시간 대응 로직이 강화되었습니다.

| 모듈 | 파라미터 수 | 조정 난이도 | 영향도 | 주요 역할 |
|------|-----------|----------|-------|----------|
| 윈도우(Sliding Window) | 3개 | 낮음 | **매우 높음** | 데이터 증강 및 샘플 해상도 결정 |
| 전처리(Preprocessing) | 7개 | 중간 | 높음 | 노이즈 제거 및 주파수 정규화 |
| 보정(Sanitization) | 4개 | 낮음 | 중간 | 위상 오프셋 및 진폭 보정 |
| 특징추출(Feature Extraction) | 13개 | 중간 | **매우 높음** | 물리적 특징 벡터 생성 (DWT, DFS 등) |
| 학습(Learning/MLP) | 5개 | 낮음 | 높음 | 특징-보폭 레이블 매핑 및 분류 |

---

## 🕒 Module 0: Sliding Window (슬라이딩 윈도우)
**목적**: 실시간 추론 대응 및 데이터셋 확장 (Data Augmentation)

| 파라미터명 | 타입 | 기본값 | 사용 가능한 범위 (Ablation) | 설명 |
|-----------|------|-------|--------------------------|------|
| `enabled` | bool | False | {True, False} | 슬라이딩 윈도우 활성화 여부 |
| `window_sec` | float | 1.0 | 0.5 ~ 5.0 | 단일 샘플의 시간적 길이 (초) |
| `hop_sec` | float | 0.5 | 0.1 ~ window_sec | 윈도우 이동 간격 (중첩률 결정) |

- **핵심 통찰**: `window_sec`이 클수록 보행 정보가 풍부해져 정확도가 상승하며, `hop_sec`이 작을수록 데이터 증강 효과가 극대화됨.
- **예외 처리**: 윈도우보다 짧은 신호는 **Zero-padding**을 통해 유실 없이 처리됨.

---

## 🔧 Module 1: Preprocessing (전처리)
**파일**: `src/sanitization/scripts/preprocess.py`

| # | 파라미터명 | 타입 | 기본값 | 범위 | 조정 가능 | 설명 |
|----|-----------|------|-------|------|---------|------|
| 1 | `target_fs` | int | 100 | 50-200 | ✅ | 목표 샘플링 주파수 (Hz) |
| 2 | `max_gap_ms` | float | 20.0 | 5.0-50.0 | ✅ | 보간 기준 최대 갭 (ms) |
| 3 | `hampel_enabled` | bool | False | {True, False} | ✅ | Hampel 이상치 필터 활성화 |
| 4 | `hampel_window` | int | 7 | 3-11 (홀수) | ✅ | Hampel 필터 윈도우 크기 |
| 5 | `hampel_threshold` | float | 5.0 | 1.0-10.0 | ✅ | Hampel 이상치 감지 임계값 |
| 6 | `lowpass_enabled` | bool | False | {True, False} | ✅ | Butterworth 저역통과 필터 활성화 |
| 7 | `lowpass_cutoff` | float | 11.0 | 5.0-20.0 | ✅ | 저역통과 필터 차단 주파수 (Hz) |

---

## 📊 Module 2: Sanitization & Calibration (보정)
**파일**: `src/sanitization/scripts/sanitization.py`

| # | 파라미터명 | 타입 | 기본값 | 범위 | 조정 가능 | 설명 |
|----|-----------|------|-------|------|---------|------|
| 1 | `enable_ratio` | bool | False | {True, False} | ✅ | CSI Ratio (다중 안테나 처리) 사용 |
| 2 | `linear_start` | int | 30 | 20-40 | ✅ | 선형 부반송파 구간 시작 인덱스 |
| 3 | `linear_end` | int | 78 | 60-90 | ✅ | 종료 인덱스 (exclusive) |
| 4 | `make_template` | bool | False | {True, False} | ✅ | 캘리브레이션 템플릿 자동 생성 여부 |

---

## 🎯 Module 3: Feature Extraction (특징 추출)

### 3.1 DWT (Discrete Wavelet Transform)
| # | 파라미터명 | 타입 | 기본값 | 범위 | 조정 가능 | 설명 |
|----|-----------|------|-------|------|---------|------|
| 1 | `wavelet` | str | "sym3" | sym3, db2, coif1 등 | ✅ | 분석용 웨이블릿 모함수 |
| 2 | `level` | int | 10 | 5-15 | ✅ | 웨이블릿 분해 레벨 |
| 3 | `n_pca` | int | 6 | 3-20 | ✅ | PCA 주성분 수 (차원 압축) |

### 3.2 DFS (Doppler Frequency Spectrum)
| # | 파라미터명 | 타입 | 기본값 | 범위 | 조정 가능 | 설명 |
|----|-----------|------|-------|------|---------|------|
| 1 | `n_fft` | int | 64 | 32-128 | ✅ | STFT용 FFT 크기 |
| 2 | `hop` | int | 4 | 2-8 | ✅ | STFT 시간 해상도 (샘플 단위) |
| 3 | `doppler_hz` | float | 50.0 | 30-50 | ✅ | 분석 유효 도플러 대역 (±Hz) |

### 3.3 SDP (Spatial Doppler Profile)
| # | 파라미터명 | 타입 | 기본값 | 범위 | 조정 가능 | 설명 |
|----|-----------|------|-------|------|---------|------|
| 1 | `n_lag` | int | 20 | 10-50 | ✅ | ACF 오토상관계수 최대 래그 |
| 2 | `wt` | int | 100 | 50-200 | ✅ | 분석용 내부 패킷 윈도우 크기 |
| 3 | `hop` | int | 50 | 25-100 | ✅ | 내부 윈도우 이동 간격 |
| 4 | `n_windows` | int | 1 | 1-5 | ✅ | 샘플당 추출 특징 수 |

---

## 🧠 Module 4: Learning (학습 모델 - MLP)
**파일**: `src/learning/mlp/` 하위 모듈

| # | 파라미터명 | 타입 | 기본값 | 범위 | 조정 가능 | 설명 |
|----|-----------|------|-------|------|---------|------|
| 1 | `hidden` | list | [256, 128, 64] | 가변 (깊이/너비) | ✅ | MLP 은닉층 구조 |
| 2 | `activation` | str | "relu" | relu, tanh, logistic | ✅ | 뉴런 활성화 함수 |
| 3 | `epochs` | int | 300 | 100-1000 | ✅ | 최대 학습 에포크 수 |
| 4 | `lr` | float | 0.001 | 1e-4 ~ 1e-2 | ✅ | 초기 학습률 (Learning Rate) |
| 5 | `dfs_n_pca` | int | 32 | 16-64 | ✅ | DFS 전용 특징 압축 차원 |

---

## 🚀 Ablation Study 실행 가이드

본 프로젝트는 `main.py`를 통해 모든 파라미터의 Ablation을 자동화합니다.

### 1. 개별 파라미터 오버라이드 (CLI)
```bash
# 특정 윈도우와 전처리를 수동으로 지정하여 실행
python main.py --config configs/default.yaml \
    --set sliding_window.enabled=true \
    --set preprocessing.hampel_enabled=true
```

### 2. 하이퍼파라미터 스윕 (YAML)
`configs/ablation/` 폴더의 YAML 파일을 사용하여 여러 지점의 성능을 연속 측정합니다.

```bash
# 전체 데이터 세트를 사용하여 윈도우 크기 스윕 실행
python main.py --ablation configs/ablation/window_size.yaml --set experiment.date_tag=""
```

---

## 📌 주요 기술적 발견 및 주의사항

1.  **전체 데이터 세트(Full Dataset) 처리**: `experiment.date_tag=""` 설정 시, `data/raw/**/*` 패턴의 재귀적 탐색을 통해 1,039개의 모든 파일을 자동 로드합니다.
2.  **경로 격리 (Path Isolation)**: Ablation 실행 시 각 실험 포인트는 개별적인 `preprocessed/` 폴더를 생성합니다. 이는 대규모 데이터 처리 시 파일 시스템의 병목을 방지하기 위함입니다.
3.  **학습 안정성**: 모든 분석은 **5-Fold Cross Validation**을 기반으로 수행되며, 표준 편차(SD)가 낮을수록 해당 파라미터의 신뢰도가 높음을 의미합니다.

---
**작성자**: Antigravity  
**마지막 검토 일자**: 2026-04-24
