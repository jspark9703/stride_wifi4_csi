# WIFI CSI 데이터 전처리 및 정제 파이프라인 (Sanitization Module)

이 디렉터리(`sanitization`)는 Wi-Fi CSI(Channel State Information) 원본 데이터를 받아 누락된 패킷을 보간(Interpolation)하고, 하드웨어 고유의 비선형 왜곡(위상 S-Shape, 진폭 M-Shape)을 제거하며, 최종 데이터 품질을 검증하기 위한 시각화 및 진단 로그를 생성하는 일괄 파이프라인 모듈입니다.

---

## 📁 파일 및 세부 기능 명세 (Feature Breakdown)

### 1. `main_pipeline.py` ─ 전체 파이프라인 오케스트레이터
전체 4단계 파이프라인을 순차적으로 총괄합니다.

| 단계 | 모듈 | 내용 |
|------|------|------|
| STEP 0 | `make_template` | 동축 캘리브레이션 CSV → 하드웨어 왜곡 템플릿 생성 |
| STEP 1 | `preprocess` | 원본 CSV 파싱 → 108 서브캐리어 추출 → 선형 보간 |
| STEP 2 | `sanitization` | 위상/진폭 정제 (CFO 제거, 템플릿 보정) |
| STEP 3 | `visualization` | 클래스별 시각화 + JSON 진단 로그 생성 |

- **`run_id` 동기화**: 파이프라인 시작 시점에 단 한 번 `YYYYMMDD_HHMMSS` 형식의 `run_id`를 생성하여 하위 모든 모듈에 주입합니다. 이로 인해 JSON 로그, plots 폴더, NPZ 중간 파일이 모두 동일한 타임스탬프로 묶여 추적이 용이합니다.
- **모듈 독립 실행 시**: 각 모듈을 단독 실행하면 `run_id`가 `YYYYMMDD` 날짜 단위 폴더에 저장되어 당일 실행 결과가 하나의 폴더로 묶입니다.

---

### 2. `make_template.py` ─ 하드웨어 캘리브레이션 템플릿 생성기
MATLAB `set_template()` 함수의 완벽한 파이썬 이식체입니다.

**원리**: ESP32의 Tx/Rx 포트를 안테나 대신 **동축 케이블(Coaxial Cable)** 로 직접 연결하여 수집한 CSI 데이터에는 공기 중의 무선 채널(다중 경로 반사 등)이 개입하지 않으므로, 순수한 하드웨어 고유의 왜곡 패턴만이 남아있습니다. 이 데이터를 분석해 보정용 템플릿을 추출합니다.

**세부 기능:**
- **`extract_valid_108(csi_str)`**: 길이 384의 원시 배열에서 가드밴드와 파일럿을 제외한 **108개의 유효 복소 서브캐리어**를 추출합니다.
  - 가드밴드 제외: index 2~58, 70~126 → 114개
  - 파일럿 제거: index 11, 25, 53, 75, 103, 117 (6개) → 최종 **108개**
- **`set_template(csi_calib, linear_interval)`**: MATLAB 로직 완벽 이식
  - **진폭 템플릿 (M-Shape)**: 각 패킷의 진폭을 해당 패킷의 평균 진폭으로 정규화한 뒤, 전체 패킷을 시간 축으로 평균하여 1D M-Shape 커브 추출
  - **위상 템플릿 (S-Shape)**: 위상 언래핑 후, 비선형 왜곡이 없는 **`linear_interval`(기본: index 30~77)** 구간만을 기준으로 1차 선형 피팅(poly1)을 수행하고, 원본에서 해당 선형 모델을 차감하여 잔류 비선형 위상 에러를 추출
  - `linear_interval` 구간은 명시적으로 `0`으로 클램프 (MATLAB: `csi_phase_template(1, linear_interval) = 0`)
- **출력**: `data/sanitization/template/template.csv` (컬럼: `amplitude`, `phase`) + 시각적 확인용 `template_preview.png`
- **캘리브레이션 데이터 없을 때**: "No files found" 출력 후 건너뜀, 파이프라인은 정상 진행

---

### 3. `preprocess.py` ─ CSI 데이터 전처리 및 선형 보간
원본 CSV 파일을 읽어 머신러닝 모델이 사용 가능한 고정 주파수 시계열 데이터로 변환합니다.

**세부 기능:**
- **CSI 파싱**: 길이 384의 문자열 배열을 파싱하여 HT-LTF 구간(index 128~)에서 복소수 배열 추출
- **108 서브캐리어 필터링**: 가드밴드 Null + 파일럿 서브캐리어 6개를 제거하여 **108개의 실제 데이터 서브캐리어**만 사용
  > **왜 전처리가 필수인가?** UDP 기반 Wi-Fi 패킷은 OS 네트워킹 부하로 도착 간격이 불규칙합니다. 이 불균일한 시간 축을 LSTM 등 딥러닝 모델에 그대로 입력하면 주기적 특성이 소실됩니다.
- **패킷 간격(Gap) 분석**: 인접 패킷 간 타임스탬프 차이를 측정하여 패킷 유실/지연 구간을 감지
- **선형 보간 (Linear Interpolation)**: `MAX_GAP_MS`(기본: 20ms)를 초과하는 구간에 대해 `TARGET_FS`(기본: 100Hz) 기준으로 선형 보간하여 고정 샘플링 간격 복원
- **Gap 통계 JSON 로깅**: 보간 전/후 패킷 간격, 사분위수(q25, q75), 진폭 편차를 집계하여 로그 저장

**출력**: `data/sanitization/preprocessed/*.npz` (컬럼: `time`, `csi`)

---

### 4. `sanitization.py` ─ 위상 및 진폭 하드웨어 왜곡 정제
전처리된 NPZ 파일의 복소 CSI 데이터에서 하드웨어 고유 왜곡을 제거합니다.

**세부 기능:**
- **`PhaseSanitizer.unwrap_and_linear_fit()`**:
  1. 각 패킷의 위상을 언래핑(Unwrapping) → 연속적인 위상 시계열 복원
  2. 서브캐리어 인덱스를 X축으로 1차 선형 회귀(Least Squares) 수행 → CFO/SFO 선형 기울기 계산
  3. 원본 위상에서 선형 기울기 차감(Detrending) → 하드웨어 클럭 오프셋 제거
  4. `template_phase` 있을 경우 S-Shape 비선형 잔류 에러도 추가 차감
  5. R² 값 계산하여 품질 지표로 반환
- **`AmplitudeSanitizer.correct_nonlinear_distortion()`**:
  - `template_amplitude` 있을 경우: 진폭을 M-Shape 템플릿으로 나눗셈(Division)하여 서브캐리어 양단 감쇠 보정
  - 템플릿 없을 경우: 원본 진폭 그대로 반환 (건너뜀)
- **`compute_csi_ratio()`**: 다중 안테나 환경에서 두 안테나 CSI의 비율(Division)로 SFO를 완벽 상쇄 (`--enable_ratio` 활성화 시)
- **템플릿 없을 때** Fallback: CFO/SFO 선형 제거만 수행, 비선형 보정 건너뜀

**출력**: `data/sanitization/sanitization/*.npz` + JSON 품질 메트릭

---

### 5. `visualization.py` ─ 클래스별 시각화 및 진단
정제 결과를 사람이 눈으로 검증할 수 있는 플롯과 JSON 통계로 출력합니다.

**세부 기능:**
- **클래스 자동 분류**: 파일명에서 `big`/`small` 키워드를 탐지하여 보폭 클래스별로 분리 처리
- **Timegap 히스토그램 (서브플롯)**: 보간 **전(Raw)** 과 보간 **후(Preprocessed)** 의 패킷 간격 분포를 나란히 비교
  - `big` class / `small` class 각각 별도 서브플롯으로 표시
- **진폭 & 위상 주파수 커브**: 108개 서브캐리어에 대한 평균 진폭선과 위상 평탄선을 그려 정제 알고리즘의 효과를 즉시 시각 확인
- **JSON 진단 로그**: 클래스별 평균 진폭, 위상 안정도, Timegap 분포 등 주요 메트릭을 JSON으로 저장

**출력**: `data/result/sanitization/plots/{run_id}/` + `data/result/sanitization/json/{run_id}/`

---

## 🚀 실행 명령어 가이드 (Usage)

> 실행 전 `sanitization` 디렉터리에서 작업하세요.
> ```bash
> cd c:\Users\jspar\Desktop\SCHOOL\idealab\WIFI_CSI\project_csi\demo\stride_wifi4\sanitization
> ```

### ① 기본 실행 (기본값으로 전체 파이프라인)
```bash
python main_pipeline.py
```
동축 캘리브레이션 파일이 없으면 STEP 0을 건너뛰고 STEP 1~3을 수행합니다.

---

### ② 전체 파라미터 지정 실행 (권장)
```bash
python main_pipeline.py ^
  --target_fs 100 ^
  --max_gap_ms 20.0 ^
  --calib_dir data/calibration ^
  --template_csv data/sanitization/template/template.csv ^
  --linear_start 30 ^
  --linear_end 78 ^
  --hist_bins 100 ^
  --hist_range_max 40
```

---

### ③ 파라미터 상세 설명

| 카테고리 | 인자 | 기본값 | 설명 |
|----------|------|--------|------|
| **전처리** | `--target_fs` | `100` | 선형 보간 목표 샘플링 주파수 (Hz) |
| **전처리** | `--max_gap_ms` | `20.0` | 이 값(ms) 이상의 패킷 간격에 대해서만 보간 수행 |
| **정제** | `--enable_ratio` | `False` | 다중 안테나 CSI Ratio 활성화 (플래그, 인자값 없음) |
| **정제** | `--template_csv` | `data/sanitization/template/template.csv` | 동축 하드웨어 템플릿 CSV 경로 |
| **캘리브레이션** | `--calib_dir` | `data/calibration` | 동축 원시 CSI .csv 파일이 들어있는 폴더 |
| **캘리브레이션** | `--linear_start` | `30` | 선형 구간 시작 서브캐리어 인덱스 (비선형 왜곡이 없는 구간) |
| **캘리브레이션** | `--linear_end` | `78` | 선형 구간 끝 서브캐리어 인덱스 (exclusive) |
| **시각화** | `--hist_bins` | `100` | 타임갭 히스토그램 막대 개수 |
| **시각화** | `--hist_range_max` | `40` | 히스토그램 X축 최대값 (ms) |

---

### ④ 개별 모듈 단독 실행

**템플릿 생성만 실행 (동축 데이터 있을 때):**
```bash
python make_template.py --calib_dir ../data/calibration --linear_start 30 --linear_end 78
```

**전처리만 실행:**
```bash
python preprocess.py
```

**정제만 실행:**
```bash
python sanitization.py
```

**시각화만 실행:**
```bash
python visualization.py
```

> ⚠️ 단독 실행 시 결과 폴더는 `YYYYMMDD` 날짜 단위로 자동 그룹핑됩니다.

---

### ⑤ 동작 시나리오별 정리

| 시나리오 | 명령어 예시 |
|---------|-----------|
| 동축 데이터 없이 빠른 파이프라인 | `python main_pipeline.py` |
| 동축 데이터 있을 때 템플릿 포함 전체 실행 | `python main_pipeline.py --calib_dir data/calibration` |
| 다중 안테나 환경 (CSI Ratio 활성화) | `python main_pipeline.py --enable_ratio` |
| 모든 인자 조회 | `python main_pipeline.py --help` |

---

## 📊 실행 결과 분석 가이드 (Result Metrics)

최근 실행 결과(`data/result/sanitization/json/20260406/`) 기준 분석입니다.

### [STEP 1] 전처리 (Preprocessing) 품질
> **922개 파일 모두 실패 없이 (100% Success) 약 151초 내에 완료**

| 지표 | 값 | 설명 |
|------|---|------|
| 처리 파일 수 | 922 / 922 | 전체 성공 |
| 중간 패킷 간격 (Median) | 11.0 ms | 정상 수신 간격 |
| Small Gap 진폭 변화량 (≤q25) | 17.0 | 짧은 간격일수록 이동 변화 적음 |
| Large Gap 진폭 변화량 (≥q75) | 18.8 | 긴 간격일수록 이동량 증가 (물리적으로 타당) |

### [STEP 2] 정제 (Sanitization) 품질
> **922개 파일 SFO/CFO 위상 제거 + 진폭 보정 약 30초 내 완료**

| 지표 | 값 | 해석 |
|------|---|------|
| `mean_r_squared` | 0.63 | 위상의 63%가 SFO 선형 트렌드 → 성공적으로 제거됨 |
| `phase_stability_score` | 93.40 | Detrending 후 위상 분산이 매우 낮아짐 (안정적) |
| `amplitude_cv_mean` | 0.47 | M-Shape 보정 후 서브캐리어 간 진폭 균일성 정상 |

---

## 📂 디렉터리 구조

```
stride_wifi4/
├── data/
│   ├── raw/                        ← 원본 CSI CSV 파일
│   ├── calibration/                ← 동축 캘리브레이션 CSV (있을 경우)
│   └── sanitization/
│       ├── preprocessed/           ← STEP 1 출력 (.npz)
│       ├── sanitization/           ← STEP 2 출력 (.npz)
│       └── template/
│           ├── template.csv        ← M/S-Shape 하드웨어 보정 템플릿
│           └── template_preview.png
└── sanitization/                   ← 이 디렉터리
    /result/sanitization/
    ├── json/{YYYYMMDD_HHMMSS}/         ← JSON 진단 로그
    └── plots/{YYYYMMDD_HHMMSS}/        ← 시각화 플롯
  __script
    ├── main_pipeline.py
    ├── make_template.py
    ├── preprocess.py
    ├── sanitization.py
    ├── visualization.py
    └── README.md
    

data
```
