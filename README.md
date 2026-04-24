# WiFi CSI Stride Classification Pipeline

WiFi CSI(Channel State Information) 신호를 이용하여 보행 보폭(big/small)을 분류하는 머신러닝 파이프라인.  
YAML 기반 단일 진입점 `main.py` 로 전처리 → 윈도우 → 특징 추출 → 학습까지 전체 흐름을 제어한다.

---

## 0. 파이프라인 순서

```
raw/
 │
 ▼  [window]  ← sliding_window.enabled=true 일 때만 실행
windowed/          raw CSV → 시간 기반 슬라이딩 윈도우 분할 → windowed CSV
 │
 ▼  [preprocess]
preprocessed/      CSV → NPZ 변환 + 보간 + Hampel / Low-pass 필터
 │
 ▼  [sanitize]
sanitization/      위상/진폭 보정 (linear subcarrier 활용)
 │
 ▼  [extract]
feature_extraction/ DWT / DFS / SDP / TD-DFS 특징 추출
 │
 ▼  [train]
models/            sklearn MLP 학습 · 평가
```

> `sliding_window.enabled=false` 이면 window 스텝을 건너뛰고 preprocess 가 `raw/` 를 직접 읽어 full-signal NPZ 생성.

---

## 1. 실행 방법 및 명령어

### 1-1. 단일 실험 (`--config`)

```bash
# baseline 전체 파이프라인
python main.py --config configs/baseline.yaml

# 특정 스텝만 실행 (데이터가 이미 있을 때)
python main.py --config configs/baseline.yaml --steps extract train
python main.py --config configs/baseline.yaml --steps preprocess sanitize
python main.py --config configs/baseline.yaml --steps train

# date_tag 없이 전체 데이터셋 사용
python main.py --config configs/baseline.yaml --set experiment.date_tag=""
```

### 1-2. 즉석 파라미터 오버라이드 (`--set KEY=VALUE`)

```bash
# 슬라이딩 윈도우 크기 변경
python main.py --config configs/baseline.yaml \
  --set sliding_window.window_sec=3.0 \
  --set sliding_window.hop_sec=1.0

# 특징 추출 방법 변경
python main.py --config configs/baseline.yaml --set feature_extraction.method=dfs

# DWT 레벨 변경
python main.py --config configs/baseline.yaml --set feature_extraction.dwt.level=8

# MLP 구조 변경
python main.py --config configs/baseline.yaml --set learning.hidden=[512,256,128]

# Hampel 필터 활성화
python main.py --config configs/baseline.yaml \
  --set preprocessing.hampel_enabled=true \
  --set preprocessing.hampel_window=5 \
  --steps preprocess sanitize extract train
```

> **KEY 형식**: YAML dot-path 표기 (예: `feature_extraction.dwt.n_pca=10`)  
> **VALUE 자동 파싱**: `true/false` → bool, 숫자 → int/float, `[..]` → list

### 1-3. Ablation Sweep (`--ablation`)

```bash
python main.py --ablation configs/ablation/window_size.yaml     # 윈도우 크기 비교
python main.py --ablation configs/ablation/window_hop.yaml      # hop 크기 비교
python main.py --ablation configs/ablation/window_enabled.yaml  # 윈도우 활성화 여부
python main.py --ablation configs/ablation/dwt_level.yaml       # DWT 분해 레벨 비교
python main.py --ablation configs/ablation/dwt_n_pca.yaml       # PCA 주성분 수 비교
python main.py --ablation configs/ablation/dwt_wavelet.yaml     # 웨이블릿 종류 비교
python main.py --ablation configs/ablation/preprocess_hampel.yaml  # Hampel 비교
python main.py --ablation configs/ablation/mlp_hidden.yaml      # MLP 구조 비교
python main.py --ablation configs/ablation/full_sweep.yaml      # 통합 sweep
```

---

## 2. Ablation 가능한 파라미터 전체 목록

### [A] 슬라이딩 윈도우 (`sliding_window.*`)

| 파라미터 | YAML 키 | 기본값 | 권장 범위 |
|---|---|---|---|
| 윈도우 크기 [초] | `sliding_window.window_sec` | `2.5` | `[1.0, 1.5, 2.0, 2.5, 3.0]` |
| hop 크기 [초] | `sliding_window.hop_sec` | `0.5` | `[0.25, 0.5, 1.0, 1.5]` |
| 윈도우 활성화 | `sliding_window.enabled` | `true` | `[true, false]` |

> ⚠️ `sliding_window.*` 변경 시 **모든 스텝** (window → preprocess → sanitize → extract → train) 재실행. baseline 재사용 없음.

```bash
python main.py --ablation configs/ablation/window_size.yaml
python main.py --config configs/baseline.yaml --set sliding_window.window_sec=1.5
```

---

### [B] 전처리 (`preprocessing.*`)

| 파라미터 | YAML 키 | 기본값 | 권장 범위 |
|---|---|---|---|
| 목표 샘플링 주파수 | `preprocessing.target_fs` | `100` | `[50, 100]` |
| 보간 갭 기준 [ms] | `preprocessing.max_gap_ms` | `20.0` | `[10.0, 20.0, 50.0]` |
| Hampel 필터 활성화 | `preprocessing.hampel_enabled` | `false` | `[true, false]` |
| Hampel 윈도우 크기 | `preprocessing.hampel_window` | `7` | `[3, 5, 7, 9, 11]` |
| Hampel 이상치 임계값 | `preprocessing.hampel_threshold` | `5.0` | `[2.0, 3.0, 5.0, 7.0]` |
| Low-pass 필터 활성화 | `preprocessing.lowpass_enabled` | `false` | `[true, false]` |
| Low-pass 차단 주파수 [Hz] | `preprocessing.lowpass_cutoff` | `11.0` | `[5.0, 8.0, 11.0, 15.0]` |

> `windowed/` 는 baseline 재사용, `preprocessed/` 부터 새로 생성.

```bash
python main.py --ablation configs/ablation/preprocess_hampel.yaml
python main.py --config configs/baseline.yaml \
  --set preprocessing.hampel_enabled=true \
  --set preprocessing.hampel_window=5 \
  --steps preprocess sanitize extract train
```

---

### [C] 위상/진폭 보정 (`sanitization.*`)

| 파라미터 | YAML 키 | 기본값 | 권장 범위 |
|---|---|---|---|
| CSI Ratio 사용 | `sanitization.enable_ratio` | `false` | `[true, false]` |
| 선형 부반송파 구간 시작 | `sanitization.linear_start` | `30` | `[20, 25, 30]` |
| 선형 부반송파 구간 종료 | `sanitization.linear_end` | `78` | `[70, 78, 90]` |

> `windowed/` + `preprocessed/` 는 baseline 재사용, `sanitization/` 부터 새로 생성.

```bash
python main.py --config configs/baseline.yaml \
  --set sanitization.enable_ratio=true \
  --steps sanitize extract train
```

---

### [D] 특징 추출 (`feature_extraction.*`)

#### 공통

| 파라미터 | YAML 키 | 기본값 | 선택지 |
|---|---|---|---|
| 추출 방법 | `feature_extraction.method` | `"dwt"` | `dwt \| dfs \| sdp \| tddfs` |

#### DWT (Discrete Wavelet Transform)

| 파라미터 | YAML 키 | 기본값 | 권장 범위 |
|---|---|---|---|
| 웨이블릿 함수 | `feature_extraction.dwt.wavelet` | `"sym3"` | `db2, db4, sym3, coif1, haar` |
| 분해 레벨 | `feature_extraction.dwt.level` | `10` | `[5, 7, 8, 10, 12, 15]` |
| PCA 주성분 수 | `feature_extraction.dwt.n_pca` | `6` | `[3, 6, 10, 15, 20]` |

#### DFS (Doppler Frequency Spectrum, STFT 기반)

| 파라미터 | YAML 키 | 기본값 | 권장 범위 |
|---|---|---|---|
| FFT 크기 | `feature_extraction.dfs.n_fft` | `64` | `[32, 64, 128]` |
| STFT hop [샘플] | `feature_extraction.dfs.hop` | `4` | `[2, 4, 8]` |
| 유효 도플러 대역 ±Hz | `feature_extraction.dfs.doppler_hz` | `50.0` | `[30.0, 40.0, 50.0]` |

#### SDP (Spatial Doppler Profile, ACF 기반)

| 파라미터 | YAML 키 | 기본값 | 권장 범위 |
|---|---|---|---|
| ACF 최대 lag [샘플] | `feature_extraction.sdp.n_lag` | `20` | `[10, 20, 30, 50]` |
| 내부 윈도우 패킷 수 | `feature_extraction.sdp.wt` | `100` | `[50, 100, 200]` |
| 내부 윈도우 hop | `feature_extraction.sdp.hop` | `50` | `[25, 50, 100]` |

#### TD-DFS (Time-Derivative DFS)

| 파라미터 | YAML 키 | 기본값 | 권장 범위 |
|---|---|---|---|
| Δt 탐색 최솟값 [샘플] | `feature_extraction.tddfs.delta_t_min` | `1` | `[1, 2, 5]` |
| Δt 탐색 최댓값 [샘플] | `feature_extraction.tddfs.delta_t_max` | `10` | `[5, 10, 20]` |

> `windowed/` + `preprocessed/` + `sanitization/` 은 baseline 재사용, `feature_extraction/` 만 새로 생성.

```bash
python main.py --ablation configs/ablation/dwt_level.yaml
python main.py --ablation configs/ablation/dwt_wavelet.yaml
python main.py --ablation configs/ablation/dwt_n_pca.yaml
python main.py --config configs/baseline.yaml \
  --set feature_extraction.method=dfs --steps extract train
```

---

### [E] 학습 (`learning.*`)

| 파라미터 | YAML 키 | 기본값 | 권장 범위 |
|---|---|---|---|
| 은닉층 구조 | `learning.hidden` | `[256,128,64]` | `[128,64]`, `[256,128,64]`, `[512,256,128]` |
| 활성화 함수 | `learning.activation` | `"relu"` | `relu \| tanh \| logistic` |
| 최대 에폭 | `learning.epochs` | `300` | `[100, 200, 300, 500]` |
| 초기 학습률 | `learning.lr` | `0.001` | `[0.0001, 0.001, 0.01]` |

> 모든 데이터를 baseline 재사용, train 단계만 새로 실행.

```bash
python main.py --ablation configs/ablation/mlp_hidden.yaml
python main.py --config configs/baseline.yaml \
  --set learning.hidden=[512,256,128] --steps train
```

---

## 3. Ablation 시 Baseline 데이터 재사용 정책

| ablation 도메인 | baseline 재사용 | 새로 생성 |
|---|---|---|
| `sliding_window.*` | — (전체 재실행) | windowed → preprocessed → sanitization → feature_extraction |
| `preprocessing.*` | `windowed/` | preprocessed → sanitization → feature_extraction |
| `sanitization.*` | `windowed/` + `preprocessed/` | sanitization → feature_extraction |
| `feature_extraction.*` | `windowed/` + `preprocessed/` + `sanitization/` | feature_extraction |
| `learning.*` | `windowed/` + `preprocessed/` + `sanitization/` + `feature_extraction/` | (모델만 새로) |

---

## 4. 결과물 경로 총정리

### 4-1. Baseline 단일 실험 (`exp_name == "baseline"`)

| 카테고리 | 경로 |
|---|---|
| 원본 데이터 | `data/raw/{date_tag}/` |
| windowed CSV | `data/ablation/baseline/windowed/` |
| 전처리 NPZ | `data/ablation/baseline/preprocessed/` |
| 보정 NPZ | `data/ablation/baseline/sanitization/` |
| 특징 벡터 | `data/ablation/baseline/feature_extraction/{method}/` |
| 학습 모델 | `results/baseline/learning/models/` |
| 특징 결과 | `results/baseline/feature_extraction/` |
| 전처리 로그 JSON | `data/logs/sanitization/json/{run_id}/` |
| 윈도우 로그 JSON | `data/logs/windowing/` |

### 4-2. 일반 단일 실험 (`exp_name != "baseline"`)

| 카테고리 | 경로 |
|---|---|
| 원본 데이터 | `data/raw/{date_tag}/` |
| windowed CSV | `data/windowed/{date_tag}/` |
| 전처리 NPZ | `data/sanitization/preprocessed/{date_tag}/` |
| 보정 NPZ | `data/sanitization/sanitization/{date_tag}/` |
| 특징 벡터 | `data/feature_extraction/{method}/{exp_name}/` |
| 학습 모델 | `results/{exp_name}/learning/models/` |
| 특징 결과 | `results/{exp_name}/feature_extraction/` |
| 전처리 로그 | `data/logs/sanitization/json/{run_id}/` |
| 윈도우 로그 | `data/logs/windowing/` |

### 4-3. Ablation Sweep 실험

| 카테고리 | 경로 |
|---|---|
| 실험 데이터 | `data/ablation/{ts}_{sweep_name}/{exp_name}/{step}/` |
| 학습 모델 | `results/{ts}_{sweep_name}/learning/{exp_name}/models/` |
| 특징 결과 | `results/{ts}_{sweep_name}/feature_extraction/{exp_name}/` |
| 통계 결과 | `results/{ts}_{sweep_name}_results.json` |
| 전처리 로그 | `data/logs/sanitization/json/{run_id}/` |
| 윈도우 로그 | `data/logs/windowing/` |

> baseline 재사용 스텝의 경로는 `data/ablation/baseline/{step}/` 을 직접 참조 (복사 없음)

| 변수 | 설명 | 예시 |
|---|---|---|
| `{ts}` | 실험 시작 타임스탬프 | `20260424_120000` |
| `{sweep_name}` | ablation YAML 파일명 | `window_size` |
| `{exp_name}` | 파라미터 값 기반 이름 | `window_sec_2p5` |
| `{method}` | 특징 추출 방법 | `dwt` |
| `{date_tag}` | 날짜 태그 | `260406` (빈 문자열이면 전체) |

### 4-4. 공통 로그 경로 요약

| 로그 종류 | 경로 |
|---|---|
| 전처리 로그 | `data/logs/sanitization/json/{run_id}/{run_id}_preprocess_log.json` |
| Sanitize 로그 | `data/logs/sanitization/json/{run_id}/{run_id}_sanitize_log.json` |
| Sanitize 플롯 | `data/logs/sanitization/plots/` |
| 윈도우 분할 로그 | `data/logs/windowing/{ts}_windowing_log.json` |
| Ablation 결과 | `results/{ts}_{sweep_name}_results.json` |

---

## 5. 전체 디렉토리 구조

```
stride_wifi4/
├── main.py                          ← 단일 진입점
├── README.md                        ← 이 파일
├── configs/
│   ├── default.yaml                 ← 기본 설정 (full-signal, extract+train)
│   ├── baseline.yaml                ← baseline (sliding window 2.5s, 전체 스텝)
│   └── ablation/
│       ├── window_size.yaml         ← sliding_window.window_sec sweep
│       ├── window_hop.yaml          ← sliding_window.hop_sec sweep
│       ├── window_enabled.yaml      ← sliding_window.enabled sweep
│       ├── dwt_level.yaml           ← feature_extraction.dwt.level sweep
│       ├── dwt_n_pca.yaml           ← feature_extraction.dwt.n_pca sweep
│       ├── dwt_wavelet.yaml         ← feature_extraction.dwt.wavelet sweep
│       ├── preprocess_hampel.yaml   ← preprocessing.hampel_window sweep
│       ├── mlp_hidden.yaml          ← learning.hidden sweep
│       └── full_sweep.yaml          ← 다중 파라미터 통합 sweep
├── pipeline/
│   ├── runner.py                    ← 파이프라인 오케스트레이터 (get_dirs, run_pipeline)
│   ├── windowing.py                 ← CSV 기반 슬라이딩 윈도우 (raw → windowed)
│   ├── config.py                    ← YAML 로드·머지·오버라이드
│   └── result_tracker.py            ← JSON 결과 누적 저장
├── src/
│   ├── sanitization/scripts/
│   │   ├── preprocess.py            ← 전처리 (보간, Hampel, Low-pass)
│   │   └── sanitization.py          ← 위상/진폭 보정
│   └── feature_extraction/
│       ├── extract_dwt.py
│       ├── extract_dfs.py
│       └── ...
└── data/
    ├── raw/                         ← 원본 CSV (변경 금지)
    ├── windowed/                    ← 일반 실험 windowed CSV
    ├── sanitization/                ← 일반 실험 preprocessed / sanitization NPZ
    ├── feature_extraction/          ← 일반 실험 특징 벡터
    ├── ablation/
    │   ├── baseline/                ← baseline 실험 전 스텝 데이터
    │   │   ├── windowed/
    │   │   ├── preprocessed/
    │   │   ├── sanitization/
    │   │   └── feature_extraction/
    │   └── {ts}_{sweep_name}/       ← ablation sweep 실험별 폴더
    │       └── {exp_name}/
    │           ├── windowed/        (새로 생성 또는 baseline 재사용 참조)
    │           ├── preprocessed/
    │           ├── sanitization/
    │           └── feature_extraction/
    └── logs/
        ├── sanitization/
        │   ├── json/                ← 전처리·보정 로그 JSON
        │   └── plots/               ← 보정 시각화 플롯
        └── windowing/               ← 윈도우 분할 로그 JSON
```
