# Ablation Study - Parameters & Hyperparameters Analysis
## WiFi CSI Stride Classification Pipeline

**Project**: WiFi CSI 기반 보폭(Stride) 크기 분류 (big/small)  
**분석 날짜**: 2026-04-23  
**분석 범위**: Feature Extraction, Learning, Preprocessing, Sanitization 전체 파이프라인

---

## 📋 Executive Summary

프로젝트는 총 **4개의 주요 모듈**로 구성되며, 각 모듈마다 **조정 가능한 파라미터**들이 있습니다.

| 모듈 | 파라미터 수 | 조정 난이도 | 영향도 |
|------|-----------|----------|-------|
| 전처리(Preprocessing) | 6개 | 중간 | 높음 |
| 특징추출(Feature Extraction) | 12개 | 중간 | 매우 높음 |
| 학습(Learning/MLP) | 5개 | 낮음 | 높음 |
| 기타(Sanitization/Calibration) | 5개 | 낮음 | 중간 |

---

## 🔧 Module 1: Preprocessing (전처리)

**파일**: `sanitization/scripts/preprocess.py` & `sanitization/main_pipeline.py`

### 파라미터 목록

| # | 파라미터명 | 타입 | 기본값 | 범위 | 조정 가능 | 설명 |
|----|-----------|------|-------|------|---------|------|
| 1 | `target_fs` | int | 100 | 50-200 | ✅ | 목표 샘플링 주파수 (Hz) |
| 2 | `max_gap_ms` | float | 20.0 | 5.0-50.0 | ✅ | 보간 기준 최대 갭 (ms) |
| 3 | `hampel_enabled` | bool | True | {True, False} | ✅ | Hampel 이상치 필터 활성화 |
| 4 | `hampel_window` | int | 7 | 3-11 (홀수) | ✅ | Hampel 필터 윈도우 크기 |
| 5 | `hampel_threshold` | float | 5.0 | 1.0-10.0 | ✅ | Hampel 이상치 감지 임계값 |
| 6 | `lowpass_enabled` | bool | False | {True, False} | ✅ | Butterworth 저주파 필터 활성화 |
| 7 | `lowpass_cutoff` | float | 11.0 | 5.0-20.0 | ✅ | 저주파 필터 차단 주파수 (Hz) |

### 현재 설정 상태
```python
# 기본값 (main_pipeline.py 라인 101-106)
target_fs = 100          # ✅ 기본값 사용 (100 Hz)
max_gap_ms = 20.0        # ✅ 기본값 사용
hampel_enabled = True    # ✅ 활성화됨 (이상치 제거)
lowpass_enabled = False  # ❌ 비활성화됨 (추가 필터링 미적용)
```

### Ablation 가능성
- **높음**: `hampel_enabled` / `lowpass_enabled` (on/off 검증)
- **높음**: `hampel_window`, `hampel_threshold` (서로 다른 조합 테스트)
- **중간**: `target_fs` (50, 100, 150 Hz 비교)
- **중간**: `lowpass_cutoff` (필터 활성화 시)

---

## 🎯 Module 2: Feature Extraction (특징 추출)

### 2.1 DWT (Discrete Wavelet Transform)
**파일**: `feature_extraction/dwt/extract_dwt.py` & `learning/mlp/dwt_mlp.py`

#### DWT 추출 파라미터
| # | 파라미터명 | 타입 | 기본값 | 범위 | 조정 가능 | 설명 |
|----|-----------|------|-------|------|---------|------|
| 1 | `wavelet` | str | "sym3" | 모든 PyWavelets 지원 | ✅ | 웨이블릿 함수 |
| 2 | `level` | int | 10 | 3-12 | ✅ | 분해 레벨 수 |
| 3 | `n_pca` | int | 6 | 3-20 | ✅ | PCA 주성분 수 |

#### DWT-MLP 학습 파라미터
| # | 파라미터명 | 타입 | 기본값 | 범위 | 조정 가능 | 설명 |
|----|-----------|------|-------|------|---------|------|
| 4 | `hidden` | list | [256, 128, 64] | 가변 | ✅ | MLP 은닉층 노드 수 |
| 5 | `activation` | str | "relu" | {relu, tanh, logistic} | ✅ | 활성화 함수 |
| 6 | `epochs` | int | 300 | 100-1000 | ✅ | 최대 반복 횟수 |
| 7 | `lr` | float | 1e-3 | 1e-5 ~ 1e-1 | ✅ | 초기 학습률 |
| 8 | `dropout_alpha` | float | 1e-4 | 1e-5 ~ 1e-2 | ✅ | L2 정규화 계수 |

### 현재 설정 상태
```python
# extract_dwt.py 라인 42
wavelet="sym3", level=10, n_pca=6

# dwt_mlp.py 라인 145-149 (기본값)
hidden_layers=(256, 128, 64)
activation="relu"
max_iter=300
lr_init=1e-3
dropout_alpha=1e-4  # L2 정규화
```

### Ablation 가능성
- **매우 높음**: `level` (10→5, 10→15 검증)
- **매우 높음**: `n_pca` (6→3, 6→12 비교)
- **높음**: `wavelet` (sym3 vs db2 vs coif1 등 비교)
- **높음**: `hidden` 구조 변경 (256-128-64 vs 512-256-128 vs 128-64-32)
- **중간**: `activation` (relu vs tanh 비교)
- **중간**: `epochs`, `lr` (수렴성 영향)

---

### 2.2 DFS (Doppler Frequency Spectrum)
**파일**: `feature_extraction/dfs/extract_dfs.py` & `learning/mlp/dfs_mlp.py`

#### DFS 추출 파라미터
| # | 파라미터명 | 타입 | 기본값 | 범위 | 조정 가능 | 설명 |
|----|-----------|------|-------|------|---------|------|
| 1 | `n_fft` | int | 64 | 32-256 | ✅ | STFT FFT 크기 |
| 2 | `hop` | int | 4 | 1-16 | ✅ | STFT hop 크기 (샘플) |
| 3 | `doppler_hz` | float | 50 | 10-50 | ✅ | 유효 도플러 대역 ±Hz |

#### DFS-MLP 학습 파라미터
| # | 파라미터명 | 타입 | 기본값 | 범위 | 조정 가능 | 설명 |
|----|-----------|------|-------|------|---------|------|
| 4 | `n_pca` | int | 16 | 8-32 | ✅ | PCA 주성분 (통계 벡터 압축) |
| 5 | `hidden` | list | [256, 128, 64] | 가변 | ✅ | MLP 은닉층 구조 |

### 현재 설정 상태
```python
# extract_dfs.py (기본값)
n_fft=64, hop=4, doppler_hz=50

# dfs_mlp.py (추정되는 기본값)
n_pca=16
hidden=[256, 128, 64]
```

### Ablation 가능성
- **높음**: `n_fft` (64→128 또는 64→32 비교)
- **높음**: `hop` (시간 해상도 영향)
- **중간**: `doppler_hz` (유효 대역 제약)

---

### 2.3 SDP (Spatial Doppler Profile)
**파일**: `feature_extraction/sdp/extract_sdp.py`

#### SDP 추출 파라미터
| # | 파라미터명 | 타입 | 기본값 | 범위 | 조정 가능 | 설명 |
|----|-----------|------|-------|------|---------|------|
| 1 | `n_lag` | int | 20 | 10-50 | ✅ | ACF 최대 lag 샘플 수 |
| 2 | `wt` | int | 100 | 50-200 | ✅ | 슬라이딩 윈도우 패킷 수 |
| 3 | `hop` | int | 50 | 25-100 | ✅ | 윈도우 이동 간격 |
| 4 | `n_windows` | int | 1 | 1-5 | ✅ | 추출할 최대 윈도우 수 |

### Ablation 가능성
- **중간**: `n_lag` (ACF 깊이 영향)
- **중간**: `wt` (시간 해상도 영향)
- **낮음**: `hop` (윈도우 겹침 정도)
- **낮음**: `n_windows` (추출 전략)

---

### 2.4 TD-DFS (Time-Derivative DFS)
**파일**: `feature_extraction/td-dfs/extract_tddfs.py`

- DFS의 변형으로, 시간 도함수 기반 특징 추출
- 기본 매개변수 구조는 DFS와 유사

---

## 🧠 Module 3: Learning (학습 모델)

### 공통 MLP 하이퍼파라미터 (모든 feature 타입)

**파일**: `learning/mlp/{dwt,dfs,sdp,tddfs}_mlp.py`

#### sklearn MLPClassifier 공통 설정
| # | 파라미터명 | 타입 | 기본값 | 범위 | 현재값 | 조정 가능 | 설명 |
|----|-----------|------|-------|------|-------|---------|------|
| 1 | `solver` | str | - | adam | adam | ❌ | 최적화 알고리즘 (고정) |
| 2 | `batch_size` | int | - | 16-128 | 64 | ❌ | 배치 크기 (고정) |
| 3 | `learning_rate` | str | - | adaptive | adaptive | ❌ | 학습률 스케줄 (고정) |
| 4 | `random_state` | int | - | - | 42 | ❌ | 난수 시드 (재현성) |
| 5 | `early_stopping` | bool | - | - | True | ❌ | 조기 종료 (고정) |
| 6 | `validation_fraction` | float | - | - | 0.1 | ❌ | 검증 데이터 비율 (고정) |
| 7 | `n_iter_no_change` | int | - | - | 20 | ❌ | 조기 종료 판정 기준 (고정) |

#### 가능한 MLP 조정 파라미터
| # | 파라미터명 | 타입 | 기본값 | 범위 | 조정 가능 | 설명 |
|----|-----------|------|-------|------|---------|------|
| A | `hidden_layers` | tuple | 가변 | 가변 | ✅ | 은닉층 구조 |
| B | `activation` | str | relu | {relu, tanh, logistic} | ✅ | 활성화 함수 |
| C | `max_iter` | int | 300 | 100-1000 | ✅ | 최대 반복 횟수 |
| D | `learning_rate_init` | float | 1e-3 | 1e-5 ~ 1e-1 | ✅ | 초기 학습률 |
| E | `alpha` (L2) | float | 1e-4 | 1e-5 ~ 1e-2 | ✅ | L2 정규화 강도 |

### Ablation 가능성
- **높음**: `hidden_layers` (깊이, 너비 변경)
- **중간**: `max_iter` (수렴 여부)
- **중간**: `learning_rate_init` (학습 속도)
- **낮음**: `activation` (모델 복잡도)

---

## 📊 Module 4: Sanitization & Calibration

**파일**: `sanitization/scripts/sanitization.py`, `sanitization/scripts/make_template.py`

### Sanitization 파라미터
| # | 파라미터명 | 타입 | 기본값 | 범위 | 조정 가능 | 설명 |
|----|-----------|------|-------|------|---------|------|
| 1 | `enable_ratio` | bool | False | {True, False} | ✅ | CSI Ratio (다중 안테나 처리) 활성화 |

### Template Calibration 파라미터
| # | 파라미터명 | 타입 | 기본값 | 범위 | 조정 가능 | 설명 |
|----|-----------|------|-------|------|---------|------|
| 2 | `linear_start` | int | 30 | 20-40 | ✅ | 선형 부반송파 범위 시작 |
| 3 | `linear_end` | int | 78 | 60-90 | ✅ | 선형 부반송파 범위 끝 |

### Ablation 가능성
- **낮음**: `enable_ratio` (데이터 특성에 따라)
- **낮음**: `linear_start`, `linear_end` (캘리브레이션 정확도)

---

## 📈 Ablation Study 권장 순서

### Phase 1: 전처리 검증 (영향도 높음)
1. ✅ **hampel_enabled**: On/Off 비교 → 성능 영향 측정
2. ✅ **lowpass_enabled**: Off→On 추가 (비활성화 상태)
3. ✅ **target_fs**: 50/100/150 Hz 비교

**예상 소요 시간**: 2-3시간

### Phase 2: 특징 추출 파라미터 (영향도 매우 높음)
#### DWT 중심:
4. ✅ **level**: 10→5, 10→15 검증
5. ✅ **n_pca**: 6→3, 6→12 비교
6. ✅ **wavelet**: sym3 vs db2 vs coif1 비교

#### DFS 보조:
7. ✅ **n_fft**: 64→32, 64→128 비교
8. ✅ **hop**: 4→2, 4→8 비교

**예상 소요 시간**: 4-6시간

### Phase 3: 모델 구조 (영향도 높음)
9. ✅ **MLP hidden_layers**: 
   - [256, 128, 64] (현재)
   - [512, 256, 128] (더 깊음)
   - [128, 64, 32] (더 얕음)
10. ✅ **activation**: relu vs tanh

**예상 소요 시간**: 2-3시간

### Phase 4: 미세 조정 (영향도 중간)
11. ✅ **learning_rate_init**: 1e-3 변경 시도
12. ✅ **max_iter**: 수렴 여부 체크

---

## 🎯 Feature 간 비교 (선택사항)

| Feature 타입 | 차원 | 계산 복잡도 | 해석 가능성 | 권장 ablation |
|-------------|------|-----------|----------|-------------|
| **DWT** | n_pca × (level+1) | 중간 | 높음 | level, n_pca, wavelet |
| **DFS** | ~256 (n_freq×4) | 중간 | 중간 | n_fft, hop |
| **SDP** | n_lag × n_windows | 낮음 | 낮음 | n_lag, n_windows |
| **TD-DFS** | DFS와 유사 | 중간 | 중간 | DFS와 동일 |

---

## 📝 현재 미적용 기능 & 최적화 기회

### 비활성화된 기능 (쉽게 활성화 가능)
- ❌ **lowpass_enabled**: 저주파 필터 (추가 노이즈 제거 가능성)
- ❌ **enable_ratio**: CSI Ratio (다중 안테나 환경에서 유용)

### 고정된 파라미터 (변경 가능 but 현재 고정)
- **solver**: "adam" 고정 → "sgd", "lbfgs" 시도 가능
- **batch_size**: 64 고정 → 32, 128 시도 가능
- **validation_fraction**: 0.1 고정 → 0.2, 0.15 시도 가능

### 부재 기능 (추가 개발 필요)
- ❌ Dropout층 (MLP에 명시적 Dropout 미적용, L2만 사용)
- ❌ Batch Normalization (전처리 전 정규화만 수행)
- ❌ 앙상블 (여러 feature 결합 미사용)

---

## 💡 Ablation Study Template

### 실행 예시 (DWT-MLP)
```bash
# 기본값
python learning/mlp/dwt_mlp.py
# → 결과 기록: [기본] 정확도, CV Mean/Std

# Ablation 1: level 감소
python feature_extraction/dwt/extract_dwt.py --level 5
python learning/mlp/dwt_mlp.py
# → 결과 기록: [level=5] 정확도, 성능 비교

# Ablation 2: n_pca 증가
python feature_extraction/dwt/extract_dwt.py --level 10 --n_pca 12
python learning/mlp/dwt_mlp.py
# → 결과 기록: [n_pca=12] 정확도, 성능 비교

# Ablation 3: MLP 구조 변경
python learning/mlp/dwt_mlp.py --hidden 512 256 128
# → 결과 기록: [hidden=[512,256,128]] 정확도
```

### 결과 정리 Template
```
[DWT-MLP Ablation Results]

기본값 (level=10, n_pca=6, hidden=[256,128,64]):
  - Test Accuracy: 0.XXXX
  - CV Mean±Std: 0.XXXX ± 0.XXXX

level=5:
  - Test Accuracy: 0.XXXX (Δ: +/- 0.XX%)
  - CV Mean±Std: 0.XXXX ± 0.XXXX

n_pca=12:
  - Test Accuracy: 0.XXXX (Δ: +/- 0.XX%)
  - CV Mean±Std: 0.XXXX ± 0.XXXX

(계속...)
```

---

## 📌 주요 발견 및 주의사항

### ⚠️ 중요한 제약 조건
1. **특징 차원 변화**: DWT의 `level`, `n_pca` 변경 시 최종 특징 벡터 크기 변함
   - DWT: `n_pca × (level + 1)` → MLP 입력 차원 변경
   - 각 조정 후 **MLP 재학습 필수**

2. **데이터 전처리 파이프라인**: 특징 추출 전 반드시 sanitization 완료 필요
   - 순서: 전처리 → sanitization → 특징 추출 → 학습

3. **통계 안정성**: 일부 파라미터 (예: DFS의 `hop`) 변경 시 특징 벡터 크기 변할 수 있음

### ✅ 최적 Ablation 전략
- **데이터 분할 고정**: `random_state=42` (모든 비교 동일 조건)
- **교차 검증 필수**: 5-Fold CV로 안정성 확인
- **점진적 변경**: 한 번에 1-2개 파라미터만 조정

---

## 🔗 파일 참조 맵

| 모듈 | 주요 파일 | CLI 인자 |
|------|---------|---------|
| **전처리** | sanitization/main_pipeline.py | --target_fs, --hampel_enabled, --lowpass_enabled |
| **DWT 추출** | feature_extraction/dwt/extract_dwt.py | --wavelet, --level, --n_pca |
| **DWT 학습** | learning/mlp/dwt_mlp.py | --hidden, --activation, --epochs, --lr |
| **DFS 추출** | feature_extraction/dfs/extract_dfs.py | --n_fft, --hop, --doppler_hz |
| **DFS 학습** | learning/mlp/dfs_mlp.py | --hidden, --activation, --epochs, --lr |
| **SDP 추출** | feature_extraction/sdp/extract_sdp.py | --n_lag, --wt, --hop, --n_windows |
| **Sanitization** | sanitization/main_pipeline.py | --enable_ratio |

---

## 📊 Ablation Matrix (완성도 체크)

```
✅ 조정 가능: 조정 가능한 파라미터
⚠️  검토 필요: 조정 가능하지만 주의 필요
❌ 고정됨: 현재 고정된 파라미터
```

| 파라미터 | 조정가능 | 우선순위 | 비고 |
|---------|--------|---------|------|
| target_fs | ✅ | 중간 | 신호 처리 기초 |
| hampel_enabled | ✅ | **높음** | 이상치 제거 |
| lowpass_enabled | ✅ | 중간 | 현재 미사용 |
| wavelet | ✅ | 높음 | DWT 기초 |
| level | ✅ | **높음** | 차원 결정 |
| n_pca | ✅ | **높음** | 차원 결정 |
| hidden_layers | ✅ | 높음 | 모델 용량 |
| activation | ✅ | 중간 | 표현력 |
| n_fft | ✅ | 높음 | DFS 해상도 |
| hop | ✅ | 중간 | 시간 해상도 |
| n_lag | ✅ | 중간 | SDP 깊이 |
| solver | ❌ | 낮음 | 고정됨 |
| batch_size | ❌ | 낮음 | 고정됨 |

---

## 🚀 다음 단계

1. **Phase 1 실행**: 전처리 파라미터 ablation (hampel 중심)
2. **결과 문서화**: 각 실험 결과를 JSON/CSV로 기록
3. **성능 곡선**: 파라미터 값 vs 정확도 그래프 작성
4. **최적값 도출**: Pareto frontier 분석 (성능 vs 계산량)

---

**작성자**: Claude Code  
**마지막 업데이트**: 2026-04-23
