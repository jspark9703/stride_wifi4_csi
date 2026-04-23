# MLP Classifier — WiFi CSI 보폭(Stride) 이진 분류

WiFi CSI 신호에서 추출한 특징(Feature)을 이용해 **보폭이 큰지(big) / 작은지(small)** 를 분류하는 MLP(Multi-Layer Perceptron) 기반 이진 분류기입니다.  
네 가지 특징 표현 방식을 각각 독립된 스크립트로 구현하여 성능을 비교합니다.

---

## 📁 디렉토리 구조

```
mlp/
├── dwt_mlp.py            # DWT 특징 기반 MLP 분류기
├── sdp_mlp.py            # SDP 특징 기반 MLP 분류기
├── dfs_mlp.py            # DFS 특징 기반 MLP 분류기  (신규)
├── tddfs_mlp.py          # TD-DFS 특징 기반 MLP 분류기  (신규)
├── models/
│   ├── dwt_mlp.pkl              # 학습된 DWT-MLP 모델
│   ├── dwt_label_encoder.pkl    # DWT LabelEncoder
│   ├── dwt_mlp_meta.json        # DWT 모델 메타데이터
│   ├── sdp_mlp.pkl              # 학습된 SDP-MLP 모델
│   ├── sdp_label_encoder.pkl    # SDP LabelEncoder
│   ├── sdp_mlp_meta.json        # SDP 모델 메타데이터
│   ├── dfs_mlp.pkl              # 학습된 DFS-MLP 모델
│   ├── dfs_label_encoder.pkl    # DFS LabelEncoder
│   ├── dfs_mlp_meta.json        # DFS 모델 메타데이터
│   ├── tddfs_mlp.pkl            # 학습된 TD-DFS-MLP 모델
│   ├── tddfs_label_encoder.pkl  # TD-DFS LabelEncoder
│   └── tddfs_mlp_meta.json      # TD-DFS 모델 메타데이터
└── results/
    ├── dwt_mlp_report_*.txt     # DWT 분류 리포트
    ├── dwt_confusion_*.png      # DWT 혼동행렬 시각화
    ├── sdp_mlp_report_*.txt     # SDP 분류 리포트
    ├── sdp_confusion_*.png      # SDP 혼동행렬 시각화
    ├── dfs_mlp_report_*.txt     # DFS 분류 리포트
    ├── dfs_confusion_*.png      # DFS 혼동행렬 시각화
    ├── tddfs_mlp_report_*.txt   # TD-DFS 분류 리포트
    ├── tddfs_confusion_*.png    # TD-DFS 혼동행렬 시각화
    └── tddfs_feature_importance_*.png  # TD-DFS 특징 중요도 비교
```

---

## 📄 파일별 상세 설명

### 1. `dwt_mlp.py` — DWT 특징 기반 분류기

#### 사용 특징 (Feature)

| 항목 | 내용 |
|------|------|
| 특징 종류 | **DWT (Discrete Wavelet Transform)** 에너지 특징 |
| 웨이블릿 | `sym3` (Symlet 3) |
| 분해 레벨 | 10레벨 다해상도 분해 |
| 차원 축소 | 서브밴드별 에너지 벡터 → **PCA 6성분** |
| 최종 특징 차원 | **54차원** (`n_pca=6` × 레벨별 서브밴드 수) |
| 데이터 소스 | `data/feature_extraction/dwt/*.npz` (없으면 sanitization NPZ에서 on-the-fly 계산) |

**특징 추출 과정:**  
1. Sanitization된 CSI 진폭 신호에 sym3 웨이블릿으로 10레벨 DWT 분해 수행  
2. 각 서브밴드(detail/approximation coefficients)의 에너지를 계산  
3. 서브캐리어 축 방향으로 PCA를 적용하여 6개 주성분만 보존  
4. 레벨별 PCA 성분을 연결(concatenate)하여 최종 특징 벡터 구성

**DWT를 사용한 이유:**  
보폭(stride)은 반복적인 신체 움직임이므로 CSI 신호에 특정 주파수 대역의 에너지 패턴이 나타납니다. DWT는 시간-주파수 다해상도 표현을 제공하므로, 보행 주기에 대응하는 서브밴드 에너지를 효과적으로 포착할 수 있습니다.

---

#### MLP 구조 (Architecture)

```
입력층 (Input)
  └─ 54차원 특징 벡터
  
StandardScaler (Z-score 정규화)

은닉층 1 (Hidden Layer 1)
  └─ 256 neurons, ReLU 활성화

은닉층 2 (Hidden Layer 2)
  └─ 128 neurons, ReLU 활성화

은닉층 3 (Hidden Layer 3)
  └─ 64 neurons, ReLU 활성화

출력층 (Output)
  └─ 2 neurons (big / small), Softmax
```

| 하이퍼파라미터 | 값 |
|---|---|
| 은닉층 구성 | `[256, 128, 64]` |
| 활성화 함수 | `ReLU` |
| 최적화기 | `Adam` |
| 학습률 (초기) | `1e-3` (adaptive 감소) |
| 배치 크기 | `64` |
| 최대 에폭 | `300` |
| 조기 종료 | `True` (20 epoch no-change) |
| L2 정규화 | `alpha=1e-4` |
| 검증 비율 | `10%` (early stopping용) |

---

### 2. `sdp_mlp.py` — SDP 특징 기반 분류기

#### 사용 특징 (Feature)

| 항목 | 내용 |
|------|------|
| 특징 종류 | **SDP (Spatial Doppler Profile)** — 공간적 도플러 분포 |
| 계산 방법 | CSI 진폭의 자기상관함수 (ACF, Autocorrelation Function) |
| 최대 지연(lag) | `n_lag=20` |
| 윈도우 크기 | `wt=100` 패킷 |
| 최종 특징 차원 | **2,000차원** (`n_lag=20` × `wt=100`) |
| 데이터 소스 | `data/feature_extraction/sdp/*.npz` (없으면 on-the-fly 계산) |

**특징 추출 과정:**  
1. 중앙 100패킷 윈도우를 추출하고 평균을 제거(mean-centering)  
2. 각 지연(τ=1~20)에서 ACF 계산: `acf[τ, t] = w[t] * w[t+τ]`  
3. 서브캐리어 축으로 평균을 취해 `(20, 100)` SDP 행렬 생성  
4. 열 방향으로 정규화(column-wise normalization) 후 flatten → **2,000차원** 벡터

**SDP를 사용한 이유:**  
SDP는 환경(방, 반사 경로)에 의존하지 않고 신체의 미세 움직임에 의한 도플러 변이의 시간적 패턴을 포착합니다. 보폭 크기는 보행 속도(velocity)와 연관되므로, 도플러 기반 특징이 이론적으로 효과적입니다. 특히 하드웨어 변동에 강인한 상대적 분포를 특징으로 사용합니다.

---

#### MLP 구조 (Architecture)

```
입력층 (Input)
  └─ 2,000차원 특징 벡터
  
StandardScaler (Z-score 정규화)

은닉층 1 (Hidden Layer 1)
  └─ 512 neurons, ReLU 활성화

은닉층 2 (Hidden Layer 2)
  └─ 256 neurons, ReLU 활성화

은닉층 3 (Hidden Layer 3)
  └─ 128 neurons, ReLU 활성화

출력층 (Output)
  └─ 2 neurons (big / small), Softmax
```

| 하이퍼파라미터 | 값 |
|---|---|
| 은닉층 구성 | `[512, 256, 128]` |
| 활성화 함수 | `ReLU` |
| 최적화기 | `Adam` |
| 학습률 (초기) | `1e-3` (adaptive 감소) |
| 배치 크기 | `64` |
| 최대 에폭 | `300` |
| 조기 종료 | `True` (20 epoch no-change) |
| L2 정규화 | `alpha=1e-4` |
| 검증 비율 | `10%` (early stopping용) |

> 입력 차원이 더 크므로 (2,000 vs 54) DWT 모델보다 첫 번째 은닉층이 2배(512) 큽니다.

---

## 🔄 공통 학습 파이프라인

두 스크립트 모두 동일한 학습-평가 프레임워크를 따릅니다.

```
데이터 로드
    ↓
LabelEncoder (big=0, small=1)
    ↓
Hold-out 분할 (80% 학습 / 20% 테스트, stratified)
    ↓
StandardScaler → MLPClassifier 학습 (Pipeline)
    ↓
테스트셋 평가 (Accuracy, Precision, Recall, F1, Confusion Matrix)
    ↓
5-Fold Stratified Cross Validation
    ↓
모델 / 메타데이터 / 결과 저장
```

---

## 🚀 실행 방법

### DWT-MLP 학습 및 평가
```bash
python dwt_mlp.py
# 옵션: 웨이블릿/레벨/PCA 변경
python dwt_mlp.py --wavelet db4 --level 8 --n_pca 8 --hidden 512 256 128
```

### SDP-MLP 학습 및 평가
```bash
python sdp_mlp.py
# 옵션: SDP 파라미터 변경
python sdp_mlp.py --n_lag 30 --wt 150 --hidden 1024 512 256
```

### DFS-MLP 학습 및 평가
```bash
python dfs_mlp.py
# 옵션: PCA 주성분 / 특징 파라미터 변경
python dfs_mlp.py --n_pca 32 --n_fft 128 --hidden 256 128 64
```

### TD-DFS-MLP 학습 및 평가
```bash
python tddfs_mlp.py
# 옵션: Δt 탐색 범위 변경
python tddfs_mlp.py --delta_t_max 15 --hidden 128 64 32
```

### 저장된 모델로 추론
```bash
python dwt_mlp.py --inference --model_path models/dwt_mlp.pkl
python sdp_mlp.py --inference --model_path models/sdp_mlp.pkl
python dfs_mlp.py --inference --model_path models/dfs_mlp.pkl
python tddfs_mlp.py --inference --model_path models/tddfs_mlp.pkl
```

---

## 📊 실험 결과 (Results)

> 실행 시각: `2026-04-07 03:50~03:50` KST  
> 데이터: 전체 922샘플 → 학습 737 / 테스트 185 (big: 100, small: 85)

### DWT-MLP 결과

| 지표 | 값 |
|---|---|
| **Test Accuracy** | **92.97%** |
| **5-Fold CV Mean ± Std** | **92.73% ± 1.26%** |

| 클래스 | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| big    | 0.91      | 0.96   | 0.94     | 100     |
| small  | 0.95      | 0.89   | 0.92     | 85      |
| **macro avg** | **0.93** | **0.93** | **0.93** | 185 |

**혼동행렬:**
```
             Predicted
             big   small
True  big  [  96      4 ]
      small[   9     76 ]
```

---

### SDP-MLP 결과

| 지표 | 값 |
|---|---|
| **Test Accuracy** | **59.46%** |
| **5-Fold CV Mean ± Std** | **53.15% ± 2.72%** |

| 클래스 | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| big    | 0.60      | 0.73   | 0.66     | 100     |
| small  | 0.58      | 0.44   | 0.50     | 85      |
| **macro avg** | **0.59** | **0.58** | **0.58** | 185 |

**혼동행렬:**
```
             Predicted
             big   small
True  big  [  73     27 ]
      small[  48     37 ]
```

---

### 네 모델 비교 요약

| 항목 | DWT-MLP | SDP-MLP | DFS-MLP | TD-DFS-MLP |
|------|---------|---------|---------|------------|
| 특징 종류 | DWT 에너지 | ACF 2D 매트릭스 | STFT 전력 스펙트로그램 | 순간 도플러 속도 |
| 입력 신호 | 진폭 | 진폭 | 복소 CSI | 복소 CSI |
| 특징 차원 | 54 | 2,000 | ~256 → PCA 32 | 19 (고정) |
| PCA 적용 | 있음 (PCA 내장) | 없음 | 있음 (차원 압축) | 없음 |
| 은닉층 | [256, 128, 64] | [512, 256, 128] | [256, 128, 64] | [128, 64, 32] |
| Test Accuracy | **92.97%** ✅ | 59.46% ❌ | TBD | TBD |
| 5-Fold CV | **92.73% ±1.26%** | 53.15% ±2.72% | TBD | TBD |

---

## 🔍 결과 분석 및 원인 고찰

### ✅ DWT-MLP: 높은 성능 (92.97%)

**성능이 높은 이유:**

1. **저차원 압축 특징의 효과**  
   PCA를 통해 54차원으로 축약된 DWT 에너지 특징은 노이즈를 제거하고 분류에 유효한 분산 방향만을 보존합니다. 샘플 수(922개) 대비 특징 차원이 적으므로 과적합 위험이 낮습니다.

2. **DWT의 보행 신호 적합성**  
   보폭이 크면 단위 시간당 다리가 더 넓게 이동하며, 이는 CSI 신호의 특정 주파수 대역(저주파 에너지 집중)에 명확히 드러납니다. sym3 웨이블릿의 10레벨 분해는 느린 보행 움직임(~1~2 Hz)에 해당하는 서브밴드를 정밀하게 포착합니다.

3. **낮은 CV 표준편차 (±1.26%)**  
   CV 평균(92.73%)과 테스트 정확도(92.97%)가 근접하고, fold 간 표준편차가 매우 작아 일반화가 잘 된 안정적인 모델임을 나타냅니다.

4. **small 클래스의 오분류 패턴**  
   혼동행렬에서 small → big 오분류(9개)가 big → small 오분류(4개)보다 많습니다. small 보폭은 에너지가 집중되지 않아 big과의 경계가 모호할 수 있으며, 이는 에너지 기반 특징의 본질적 한계입니다.

---

### ❌ SDP-MLP: 낮은 성능 (59.46% ≈ 무작위 수준)

**성능이 낮은 이유:**

1. **극단적 고차원 특징 (차원의 저주)**  
   SDP 특징은 `n_lag(20) × wt(100) = 2,000차원`인데, 학습 샘플은 737개에 불과합니다. **샘플 수 < 특징 차원** (0.37 비율)에 해당하여 MLP가 통계적으로 신뢰할 수 있는 결정 경계를 찾기 어렵습니다.

2. **5-Fold CV 평균 53.15% — 거의 무작위 분류**  
   CV에서는 학습 데이터가 더 줄어들어 성능이 추가로 하락했습니다. 이는 고차원 raw SDP를 MLP에 직접 입력하는 방식의 근본적인 문제를 시사합니다.

3. **SDP 정규화의 부작용**  
   SDP는 열 방향(column-wise) 정규화를 수행하여 절대적인 에너지 크기 정보를 제거합니다. 보폭 구분에는 에너지 크기의 차이도 중요한 단서인데, 이를 상대적 분포로만 표현하면 클래스 간 변별력이 감소할 수 있습니다.

4. **ACF 기반 특징의 노이즈 취약성**  
   ACF 계산에는 100패킷 윈도우 전체를 사용하는데, 보행 도중 자세 변화나 RF 간섭이 있으면 ACF 패턴에 큰 잡음이 유입됩니다. DWT는 서브밴드별 에너지로 평균화 효과가 있는 반면, ACF는 시간 축을 그대로 유지하여 노이즈에 더 민감합니다.

5. **big 클래스 편향**  
   big 클래스 Recall(73%) > small 클래스 Recall(44%) 로 모델이 big으로 치우친 분류를 합니다. small 48개(26%)가 big으로 오분류된 반면 big 27개(27%)만 small로 오분류되어, 모델이 명확한 패턴을 학습하지 못하고 다수 클래스 쪽으로 bias된 것을 확인할 수 있습니다.

---

## 💡 개선 방향

| 개선 방법 | 대상 | 예상 효과 |
|-----------|------|-----------|
| **SDP에 PCA/AutoEncoder 적용** | SDP-MLP | 2,000→50차원 압축으로 차원 저주 해소 |
| **DWT + SDP 특징 융합** | 공통 | 보완적 정보 결합으로 성능 향상 기대 |
| **DWT small 클래스 증강** | DWT-MLP | small → big 오분류 9건 감소 |
| **SDP 윈도우 크기 축소** | SDP-MLP | `wt=50, n_lag=10` → 특징 차원 500으로 감소 |
| **CNN/LSTM 계열 모델** | SDP/DFS-MLP | SDP의 시공간 패턴, DFS 2D 스펙트로그램을 더 효과적으로 처리 |
| **더 많은 학습 데이터 수집** | SDP/DFS/TD-DFS | 샘플 수 증가로 고차원 문제 완화 |
| **DFS + TD-DFS 특징 융합** | 공통 | 주파수 영역 + 시간 영역 앤스앙글 성능 기대 |
| **DFS-MLP n_fft 튜닝** | DFS-MLP | 스펙트로그램 주파수 해상도 줄파란 최적화 |
