# Feature Extraction 모듈 (DWT / SDP / DFS / TD-DFS)

이 디렉터리(`feature_extraction`)는 정제된 CSI NPZ 데이터(`data/sanitization/sanitization/`)를 읽어
네 가지 방식으로 특징을 추출합니다.  

| 모듈 | 도메인 | 사용 신호 | 핵심 출력 |
|------|--------|----------|-----------|
| **DWT** | 시간-주파수 | 진폭 (Amplitude) | 에너지 벡터 (66-dim) |
| **SDP** | 시간 지연 | 진폭 (Amplitude) | ACF 2D 매트릭스 (2000-dim) |
| **DFS** | 주파수 (STFT) | **복소 CSI (진폭 + 위상)** | 전력 스펙트로그램 2D |
| **TD-DFS** | 시간 영역 차분 | **복소 CSI (진폭 + 위상)** | 순간 도플러 속도 1D |

---

## 📁 디렉터리 구조

```
feature_extraction/
├── dwt/
│   ├── extract_dwt.py        ← DWT 에너지 특징 추출기
│   └── visualization.py      ← DWT 시각화
├── sdp/
│   ├── extract_sdp.py        ← SDP (ACF 기반 2D 매트릭스) 특징 추출기
│   └── visualization.py      ← SDP 시각화
├── dfs/
│   ├── extract_dfs.py        ← DFS (STFT 전력 스펙트로그램) 특징 추출기
│   └── visualization.py      ← DFS 시각화
├── td-dfs/
│   ├── extract_tddfs.py      ← TD-DFS (시간 영역 순간 도플러 속도) 특징 추출기
│   └── visualization.py      ← TD-DFS 시각화
└── README.md                 ← 이 문서
```

---

## 1. DWT — `dwt/extract_dwt.py`

### 알고리즘 파이프라인

| 단계 | 내용 |
|------|------|
| ① 진폭 추출 | `\|CSI\|` → (N_packets × 108) float |
| ② PCA 압축 | 서브캐리어 108차원 → 상위 K 주성분 시계열 |
| ③ DWT 노이즈 제거 | MAD 기반 Universal Threshold + Soft Thresholding (최상위 레벨) |
| ④ DWT 에너지 특징 | 각 레벨(Approx + Detail) 에너지 계산 후 L2 정규화 |
| ⑤ 특징 벡터 | K × (L+1) 크기로 평탄화 |

**출력 특징 차원** = `n_pca × (level + 1)` (기본: `6 × 11 = 66`)

### 실행

```bash
cd c:\Users\jspar\Desktop\SCHOOL\idealab\WIFI_CSI\project_csi\demo\stride_wifi4\feature_extraction\dwt

# 기본 실행
python extract_dwt.py

# 파라미터 지정
python extract_dwt.py --wavelet sym3 --level 10 --n_pca 6
```

### 파라미터

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--wavelet` | `sym3` | PyWavelets 웨이블릿 (sym3, db4, haar 등) |
| `--level`   | `10`   | DWT 분해 레벨 (시계열 길이가 짧으면 자동 축소) |
| `--n_pca`   | `6`    | PCA 주성분 수 |
| `--sanit_dir` | `data/sanitization/sanitization` | 정제된 NPZ 입력 경로 |
| `--out_dir`   | `data/feature_extraction/dwt`     | 특징 NPZ 출력 경로 |
| `--log_dir`   | `data/result/feature_extraction/dwt` | JSON 로그 경로 |

### 출력 NPZ 키

| 키 | 형태 | 설명 |
|----|------|------|
| `features` | `(66,)` float32 | DWT 에너지 특징 벡터 |
| `label`    | scalar str | `'big'` / `'small'` |
| `subject`  | scalar str | 피험자 이름 |
| `filename` | scalar str | 원본 파일명 |

---

## 2. SDP — `sdp/extract_sdp.py`

### 알고리즘 파이프라인

| 단계 | 내용 |
|------|------|
| ① 진폭 추출 | `\|CSI\|` → (N_packets × 108) float |
| ② 슬라이딩 윈도우 | 중앙 기준 `n_windows` 개 윈도우 분할 (W_T × N_S) |
| ③ ACF 텐서 생성 | lag τ = 1..N_lag → (N_lag × W_T × N_S) 3D 텐서 (NumPy 벡터화) |
| ④ 서브캐리어 평균 | (N_lag × W_T) 2D SDP 매트릭스 |
| ⑤ 확률적 정규화 | 각 열(시간축)을 합=1 로 정규화 |
| ⑥ 특징 벡터 | N_lag × W_T 평탄화 |

**출력 특징 차원** = `n_windows × n_lag × wt` (기본: `1 × 20 × 100 = 2000`)

### 실행

```bash
cd c:\Users\jspar\Desktop\SCHOOL\idealab\WIFI_CSI\project_csi\demo\stride_wifi4\feature_extraction\sdp

# 기본 실행
python extract_sdp.py

# 파라미터 지정
python extract_sdp.py --n_lag 20 --wt 100 --hop 50 --n_windows 1
```

### 파라미터

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--n_lag`     | `20`  | ACF 최대 lag 샘플 수 (속도 해상도) |
| `--wt`        | `100` | 슬라이딩 윈도우 크기 (패킷 수, 100Hz 기준 1초) |
| `--hop`       | `50`  | 윈도우 이동 간격 (50% 겹침) |
| `--n_windows` | `1`   | 파일 당 사용할 윈도우 수 (1=중앙 단일) |
| `--sanit_dir` | `data/sanitization/sanitization` | 정제된 NPZ 입력 경로 |
| `--out_dir`   | `data/feature_extraction/sdp`    | 특징 NPZ 출력 경로 |
| `--log_dir`   | `data/result/feature_extraction/sdp` | JSON 로그 경로 |

### 출력 NPZ 키

| 키 | 형태 | 설명 |
|----|------|------|
| `features` | `(2000,)` float32 | SDP 2D 매트릭스 평탄화 특징 벡터 |
| `label`    | scalar str | `'big'` / `'small'` |
| `subject`  | scalar str | 피험자 이름 |
| `filename` | scalar str | 원본 파일명 |

---

## 3. DFS — `dfs/extract_dfs.py`

> **복소 CSI (진폭 + 위상) 모두 사용** — 위상 정정(Phase Conjugate Multiplication)으로 CFO/SFO 하드웨어 오프셋 제거 후 STFT 적용

### 알고리즘 파이프라인

| 단계 | 내용 |
|------|------|
| ① 복소 CSI 로드 | `(N, 108)` complex64 |
| ② 위상 정정 | `Δ(t,s) = H(t,s+1) × conj(H(t,s))` → CFO/SFO 의존 위상 제거 |
| ③ 서브캐리어 평균 | 단일 복소 시계열 `z(t)` 생성 |
| ④ Hanning STFT | 슬라이딩 윈도우 FFT + fftshift (DC 중심화, 양방향 도플러) |
| ⑤ 전력 스펙트로그램 | `P = |STFT|²` → dB 변환 → [0,1] 정규화 |
| ⑥ 유효 대역 필터링 | `±doppler_hz Hz` 이내의 주파수만 잘라냄 (인체 이동 유효 대역) |

**출력 형태** = `(n_valid_freq, n_frames)` float32 2D 매트릭스

### 실행

```bash
cd feature_extraction/dfs

# 기본 실행
python extract_dfs.py

# 파라미터 지정
python extract_dfs.py --n_fft 128 --hop 8 --doppler_hz 50

# 이미지(.png)도 함께 저장
python extract_dfs.py --save_img

# 시각화 (4종 플롯)
python visualization.py
```

### 파라미터

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--n_fft`      | `64`   | STFT FFT 포인트 수 (주파수 해상도) |
| `--hop`        | `4`    | STFT hop 크기 (시간 해상도) |
| `--doppler_hz` | `50.0` | 유효 도플러 대역 ±Hz (인체 이동 최대 속도) |
| `--sanit_dir`  | `data/sanitization/sanitization` | 입력 경로 |
| `--out_dir`    | `data/feature_extraction/dfs`    | 출력 경로 |
| `--save_img`   | `False` | PNG 스펙트로그램 이미지 저장 여부 |

### 출력 NPZ 키

| 키 | 형태 | 설명 |
|----|------|------|
| `dfs_power`   | `(n_freq, n_time)` float32 | 정규화 전력 스펙트로그램 |
| `dfs_freq_hz` | `(n_freq,)` float32 | 주파수 축 [Hz] |
| `label`       | scalar str | `'big'` / `'small'` |
| `subject`     | scalar str | 피험자 이름 |
| `filename`    | scalar str | 원본 파일명 |

### 시각화 출력 (4종)

| 파일명 | 내용 |
|--------|------|
| `dfs_spectrogram_grid.png` | big 10 / small 10 스펙트로그램 그리드 |
| `dfs_mean_compare.png` | 클래스 평균 스펙트로그램 + 차분 맵 |
| `dfs_doppler_profile.png` | 시간 평균 도플러 세기 프로파일 비교 |
| `dfs_feature_heatmap.png` | 평탄화 스펙트로그램 히트맵 |

---

## 4. TD-DFS — `td-dfs/extract_tddfs.py`

> **복소 CSI (진폭 + 위상) 모두 사용** — STFT 없이 시간 영역 차분만으로 순간 도플러 속도를 역산. 정적 배경이 완전히 제거되고 프레임 단위 시간 해상도 확보.

### 알고리즘 파이프라인

| 단계 | 내용 |
|------|------|
| ① 복소 CSI 로드 | `(N, 108)` complex64 |
| ② TD-CSI 생성 | `TD(t) = H(t+Δt) − H(t)` → 정적 반사(벽·가구) 완전 제거 |
| ③ 최적 Δt 탐색 | `min Var[∠mean_s(TD)]` 기준 — 위상 노이즈 최소 Δt 선택 |
| ④ 방향 추정 | `sign(Im[mean_s(TD)])` → 접근(+) / 후퇴(−) |
| ⑤ 속도 역산 | `fd = arcsin(|TD|/2|H_ref|) × fs/(πΔt)` → `v = fd × λ/2` [cm/s] |

**출력 형태** = `(N − Δt,)` float32 1D 순간 도플러 속도 시계열

**물리 근거:**
```
H_dyn(t) = A · exp(j·2π·fd·t/fs)
TD(t) = H_dyn(t+Δt) − H_dyn(t)
       = A · exp(j·2π·fd·t/fs) · (exp(j·2π·fd·Δt/fs) − 1)
|TD(t)| = 2A · |sin(π·fd·Δt/fs)|   →   fd 역산 가능
```

### 실행

```bash
cd feature_extraction/td-dfs

# 기본 실행
python extract_tddfs.py

# Δt 탐색 범위 확장
python extract_tddfs.py --delta_t_max 15

# CSV 시계열도 함께 저장
python extract_tddfs.py --save_csv

# 시각화 (4종 플롯)
python visualization.py
```

### 파라미터

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--delta_t_min` | `1`      | Δt 탐색 최솟값 [샘플] |
| `--delta_t_max` | `10`     | Δt 탐색 최댓값 [샘플] |
| `--fc_hz`       | `5.18e9` | WiFi 중심 주파수 [Hz] |
| `--sanit_dir`   | `data/sanitization/sanitization` | 입력 경로 |
| `--out_dir`     | `data/feature_extraction/td-dfs` | 출력 경로 |
| `--save_npy`    | `False`  | `.npy` 1D 배열 별도 저장 |
| `--save_csv`    | `False`  | `.csv` 시계열 별도 저장 |

### 출력 NPZ 키

| 키 | 형태 | 설명 |
|----|------|------|
| `velocity` | `(N-Δt,)` float32 | 순간 도플러 속도 [cm/s] |
| `fdop`     | `(N-Δt,)` float32 | 도플러 주파수 [Hz] |
| `delta_t`  | scalar int | 사용된 최적 Δt [샘플] |
| `label`    | scalar str | `'big'` / `'small'` |
| `subject`  | scalar str | 피험자 이름 |
| `filename` | scalar str | 원본 파일명 |

### 시각화 출력 (4종)

| 파일명 | 내용 |
|--------|------|
| `tddfs_velocity_grid.png` | big 10 / small 10 속도 시계열 그리드 (접근/후퇴 색상 분리) |
| `tddfs_velocity_compare.png` | 클래스별 평균 속도 + 차이(Big−Small) 3패널 |
| `tddfs_speed_dist.png` | \|v\| 분포 히스토그램 + KDE + 박스플롯 |
| `tddfs_delta_t_stats.png` | 최적 Δt 분포 + 샘플별 속도 통계 |


```
data/
├── feature_extraction/
│   ├── dwt/          ← DWT 특징 NPZ
│   ├── sdp/          ← SDP 특징 NPZ
│   ├── dfs/          ← DFS 특징 NPZ (+ .npy / .png 선택)
│   └── td-dfs/       ← TD-DFS 특징 NPZ (+ .npy / .csv 선택)
└── result/
    └── feature_extraction/
        ├── dwt/   └── YYYYMMDD/   ← JSON 실행 로그
        ├── sdp/   └── YYYYMMDD/
        ├── dfs/   └── YYYYMMDD/
        └── td-dfs/ └── YYYYMMDD/
```

---

## 🔑 설계 원칙

### DWT / SDP — 진폭(Amplitude) 전용
- **이유**: 위상은 CFO/SFO·안테나 방향·설치 위치에 의해 크게 달라짐 (도메인 쉬프트 취약)
- **진폭**은 하드웨어 보정(M-Shape) 이후 공간 환경 변화에 상대적으로 안정적

### DFS / TD-DFS — 복소 CSI (진폭 + 위상) 사용
- **위상 정정(Phase Conjugate Multiplication)**: `Δ(t,s) = H(t,s+1) × conj(H(t,s))`  
  → 절대 위상 오프셋(CFO/SFO) 제거 후 **상대 위상**만 추출 → Doppler 정보 획득
- 위상의 Doppler 정보는 진폭이 제공할 수 없는 **이동 방향(접근/후퇴)** 과 **미세 속도** 에 해당

### DWT — 다중 해상도 속도 분석
- Sym3 웨이블릿: 보행 주기 신호(1~3 Hz 저주파 + 발 착지 충격 고주파)를 균형 있게 포착
- PCA 선행 적용: 108개 서브캐리어의 공통 패턴 추출 후 DWT → 노이즈 억제 + 차원 절감

### SDP — 도메인 불변 속도 분포
- ACF는 산란체(신체 부위별) 속도 분포를 Sinc 필터 형태로 인코딩
- 서브캐리어 평균으로 주파수 선택적 페이딩 영향 제거
- 확률적 정규화로 절대 신호 세기(환경 의존) 제거

### DFS — 주파수 도메인 Doppler 스펙트로그램
- STFT로 시간에 따른 도플러 주파수 변화를 2D 이미지로 포착
- Hanning 윈도우로 스펙트럼 누설 억제, fftshift로 DC 중심화 (양방향 도플러 표현)
- 유효 대역 `±50Hz` 필터링: 발걸음 속도 범위 내 에너지 집중

### TD-DFS — 시간 영역 순간 Doppler 속도
- STFT 윈도우 없이 **프레임 단위 시간 해상도** 확보 (민감도 ↑)
- `TD(t) = H(t+Δt) − H(t)` 차분으로 정적 배경(벽·가구) **완전 제거** (≈0으로 상쇄)
- 최적 Δt 탐색: 위상 노이즈 분산 최소화 → SNR 최적 시간 간격 자동 선택
- 물리 수식 기반 역산: arcsin 모델로 절대 속도 [cm/s] 추정 가능
