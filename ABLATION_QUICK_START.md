# Ablation Study - 빠른 시작 가이드

**목적**: 프로젝트의 모든 조정 가능 파라미터를 체계적으로 검증  
**대상**: Preprocessing → Feature Extraction → Learning 파이프라인

---

## 🎯 핵심 파라미터 요약 (우선순위 순)

### Tier 1: 매우 높은 영향도 (필수 검증)
```
우선순위 1️⃣ : DWT level (10 → 5, 8, 12, 15)
우선순위 2️⃣ : DWT n_pca (6 → 3, 6, 12, 20)
우선순위 3️⃣ : Hampel Filter (enabled 여부)
우선순위 4️⃣ : MLP hidden_layers ([256,128,64] → [512,256,128], [128,64])
우선순위 5️⃣ : DFS n_fft (64 → 32, 64, 128)
```

### Tier 2: 중간 영향도 (추천 검증)
```
우선순위 6️⃣ : Wavelet 함수 (sym3 → db2, coif1)
우선순위 7️⃣ : Target FS (100 → 50, 150)
우선순위 8️⃣ : DFS hop (4 → 2, 8)
우선순위 9️⃣ : MLP activation (relu → tanh)
```

### Tier 3: 낮은 영향도 (선택 검증)
```
우선순위 10️⃣ : Lowpass Filter (disabled → enabled)
우선순위 11️⃣ : SDP n_lag (20 → 10, 30)
우선순위 12️⃣ : Learning rate (1e-3 조정)
```

---

## ⚡ 빠른 명령어 모음

### 📌 기본값 (Baseline) 실행
```bash
# 전체 파이프라인 (기본값)
cd /path/to/stride_wifi4
python sanitization/main_pipeline.py

# DWT 특징 추출 (기본값)
python feature_extraction/dwt/extract_dwt.py

# DWT 모델 학습 (기본값)
python learning/mlp/dwt_mlp.py
```

---

## 🔬 Ablation 실험 템플릿

### Template 1️⃣: DWT Level 변경
```bash
# level=5 테스트
python feature_extraction/dwt/extract_dwt.py \
  --level 5 \
  --wavelet sym3 \
  --n_pca 6 \
  --out_dir "data/feature_extraction/dwt_level5"

# 모델 학습
python learning/mlp/dwt_mlp.py \
  --feat_dir "data/feature_extraction/dwt_level5"

# 결과 기록: [level=5] Test Accuracy 기록
```

### Template 2️⃣: DWT n_pca 변경
```bash
# n_pca=12 테스트
python feature_extraction/dwt/extract_dwt.py \
  --level 10 \
  --wavelet sym3 \
  --n_pca 12 \
  --out_dir "data/feature_extraction/dwt_npca12"

python learning/mlp/dwt_mlp.py \
  --feat_dir "data/feature_extraction/dwt_npca12"

# 결과 기록: [n_pca=12] Test Accuracy 기록
```

### Template 3️⃣: MLP 구조 변경
```bash
# 은닉층을 [512, 256, 128]로 변경
python learning/mlp/dwt_mlp.py \
  --feat_dir "data/feature_extraction/dwt" \
  --hidden 512 256 128

# 결과 기록: [hidden=[512,256,128]] Test Accuracy 기록
```

### Template 4️⃣: Preprocessing - Hampel 비활성화
```bash
# Hampel 필터 OFF
python sanitization/main_pipeline.py \
  --no-hampel_enabled \
  --target_fs 100 \
  --lowpass_enabled False

# 이후 DWT 재추출 & 학습
```

### Template 5️⃣: DFS FFT 크기 변경
```bash
# n_fft=128 테스트
python feature_extraction/dfs/extract_dfs.py \
  --n_fft 128 \
  --hop 4 \
  --doppler_hz 50 \
  --out_dir "data/feature_extraction/dfs_fft128"

python learning/mlp/dfs_mlp.py \
  --feat_dir "data/feature_extraction/dfs_fft128"

# 결과 기록: [n_fft=128] Test Accuracy 기록
```

---

## 📊 결과 추적 시트

### CSV 형식 (추천: `ablation_results.csv`)
```csv
실험번호,파라미터명,파라미터값,기본값,테스트정확도,CV_Mean,CV_Std,특징차원,학습시간,비고
1,level,5,10,0.8234,0.8156,0.0134,42,2.3m,특징차원 감소
2,level,10,10,0.8567,0.8512,0.0098,66,3.1m,기본값 (baseline)
3,level,15,10,0.8456,0.8398,0.0125,88,4.2m,과피팅 경향
4,n_pca,3,6,0.8234,0.8167,0.0156,22,2.1m,정보손실
5,n_pca,6,6,0.8567,0.8512,0.0098,66,3.1m,기본값 (baseline)
6,n_pca,12,6,0.8634,0.8589,0.0089,132,3.8m,최고 성능
...
```

### JSON 형식 (추천: `ablation_results.json`)
```json
{
  "baseline": {
    "level": 10,
    "n_pca": 6,
    "hidden": [256, 128, 64],
    "test_accuracy": 0.8567,
    "cv_mean": 0.8512,
    "cv_std": 0.0098
  },
  "experiments": [
    {
      "experiment_id": 1,
      "parameter": "level",
      "value": 5,
      "test_accuracy": 0.8234,
      "cv_mean": 0.8156,
      "cv_std": 0.0134,
      "feature_dim": 42,
      "delta_accuracy": -0.0333
    },
    ...
  ]
}
```

---

## 🔄 권장 실험 순서 (3일 집중 Ablation)

### 📅 Day 1: 전처리 & DWT 기본 파라미터
```
08:00 - 09:00: Baseline 실행 (기본값 재확인)
09:00 - 11:00: DWT level 변경 (5, 10, 12, 15)
11:00 - 13:00: DWT n_pca 변경 (3, 6, 12, 20)
13:00 - 14:00: 중간 정리 & 결과 저장

14:00 - 16:00: Hampel Filter 검증 (On/Off)
16:00 - 18:00: Target FS 변경 (50, 100, 150)
18:00 - 19:00: Day 1 결과 분석
```

### 📅 Day 2: 특징추출 심화 & MLP 구조
```
08:00 - 10:00: Wavelet 함수 비교 (sym3, db2, coif1)
10:00 - 12:00: DFS n_fft 변경 (32, 64, 128)
12:00 - 13:00: 중간 정리

13:00 - 15:00: MLP hidden_layers 변경
                - [256,128,64] (기본)
                - [512,256,128] (더 깊음)
                - [128,64,32] (더 얕음)
15:00 - 17:00: MLP activation 변경 (relu, tanh)
17:00 - 19:00: Day 2 결과 분석
```

### 📅 Day 3: 미세 조정 & 종합 분석
```
08:00 - 10:00: DFS hop 변경 (2, 4, 8)
10:00 - 12:00: Lowpass Filter 활성화 테스트
12:00 - 13:00: 중간 정리

13:00 - 15:00: 최적 파라미터 조합 재확인
15:00 - 17:00: 최종 결과 정리 & 비교 분석
17:00 - 18:00: 보고서 작성
```

---

## 📈 결과 분석 체크리스트

각 실험마다 다음을 기록하세요:

```
✅ Test Accuracy (메인 지표)
✅ 5-Fold CV Mean / Std (안정성)
✅ 특징 벡터 차원 (계산량)
✅ 학습 시간 (실행 효율)
✅ Confusion Matrix (클래스별 성능)
✅ 조기 종료 여부 (수렴 성공)
```

### 분석 후 결론 (예시)
```
[DWT Level 분석]
- level=5: 낮은 정확도 (0.823), 하지만 빠른 학습 ⚡
- level=10: 최고 정확도 (0.857), 안정적 (std=0.0098) ✅
- level=15: 약간 높은 정확도 (0.846), 하지만 과피팅 경향 ⚠️

결론: level=10 유지 (현재 기본값 최적)

---

[DWT n_pca 분석]
- n_pca=6: 기본 (0.857)
- n_pca=12: 가장 높음 (0.863), +0.6% 개선 ⭐
- n_pca=20: 약간 낮음 (0.859), 과피팅?

결론: n_pca=12로 업그레이드 권장 (미세 성능 개선, 안정성 유지)
```

---

## 🛠️ 트러블슈팅

### ❌ 문제: 특징 차원 불일치 오류
```
오류: "shape (X,) not suitable for input (X+N,)"
원인: DWT level 변경 후 특징 차원 변경됨
해결: MLP 재학습 또는 --feat_dir 명시 (새로 추출한 특징 디렉토리)
```

### ❌ 문제: 데이터 부재
```
오류: "[Data] No valid samples loaded"
원인: sanitization이 완료되지 않음
해결: 먼저 python sanitization/main_pipeline.py 실행
```

### ❌ 문제: 메모리 부족
```
오류: "MemoryError"
원인: 너무 많은 PCA 성분 또는 DWT level로 인한 차원 폭발
해결: n_pca 감소, batch 처리 추가, 컴퓨터 메모리 확인
```

### ❌ 문제: 학습 불수렴
```
증상: 정확도가 0.5 근처 (동전 던지기 수준)
원인: 학습률이 너무 높거나, 파라미터가 부적절
해결: learning_rate 감소 (--lr 1e-4) 또는 max_iter 증가 (--epochs 500)
```

---

## 📌 주의사항

### ⚠️ 꼭 기억하세요!

1. **특징 추출은 시간이 오래 걸립니다** (특히 DWT, SDP)
   - 한 번 추출된 특징은 재사용 가능
   - `--out_dir` 옵션으로 다양한 파라미터별 특징 저장

2. **MLP 학습은 빠르지만 여러 번 실행해야 합니다**
   - `random_state=42` 고정으로 재현성 보장
   - 각 파라미터 조합마다 5-Fold CV 포함 (~3-5분)

3. **특징 차원 변화에 주의**
   - DWT: `dim = n_pca × (level + 1)`
   - DFS: `dim = n_freq × 4` (n_freq는 n_fft에 따라 변함)
   - 같은 MLP 구조도 입력 차원이 다르면 재학습 필수

4. **파라미터 범위를 초과하면 오류 가능**
   - `level`: PyWavelets의 최대 분해 레벨 제한
   - `n_fft`: 신호 길이보다 커야 함
   - `hampel_window`: 홀수만 가능

---

## 📊 결과 비교 팁

### 성능 향상 평가 기준
```
±0.1% 미만:    변화 없음 (노이즈 수준)
±0.1% ~ 0.5%: 미미한 개선 (고려 가능)
±0.5% ~ 1%:   의미 있는 개선 (권장)
±1% 이상:      유의미한 개선 (강력 권장)
```

### 안정성 평가 (CV Std)
```
Std < 0.005:   매우 안정적 ✅
Std 0.005-0.01: 안정적 ✅
Std 0.01-0.02: 적당함 (주의)
Std > 0.02:    불안정 ⚠️ (파라미터 재검토)
```

---

## 🎓 학습 자료 연결

- **상세 문서**: `ABLATION_ANALYSIS.md` 참고
- **코드 실행 예시**: 각 스크립트의 docstring 참고
  ```bash
  python feature_extraction/dwt/extract_dwt.py --help
  python learning/mlp/dwt_mlp.py --help
  ```

---

## 💾 최종 보고서 템플릿

```markdown
# Ablation Study 최종 보고서

## 1. 실험 설정
- 데이터: [샘플 수], [클래스 분포]
- 기본값: level=10, n_pca=6, hidden=[256,128,64]
- 검증 방식: 5-Fold Cross Validation

## 2. 주요 발견 (상위 5개)
1. **[파라미터] 영향도 +X.X%**: [최적값]
2. ...

## 3. 최적 파라미터 조합
```
level: 10 ✅
n_pca: 12 ⭐ (개선 +0.6%)
wavelet: sym3 ✅
hidden: [256,128,64] ✅
activation: relu ✅
```

## 4. 예상 성능
- 개선 전: 85.67% ± 0.98%
- 개선 후: 86.34% ± 0.89% (+0.67%)

## 5. 계산 비용 (시간, 메모리)
- 특징 추출: ~10분 (모든 파라미터 재계산 기준)
- 모델 학습: ~3분/조합 (5-Fold 포함)
```

---

**작성일**: 2026-04-23  
**버전**: 1.0
