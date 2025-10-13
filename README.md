# SignGlove Data Analysis

> 수어(수화) 제스처 인식을 위한 데이터 기반 분석 및 시각화

## 목차

1. 소개
2. 프로젝트 구조
3. 설치 및 실행 방법
4. 데이터 설명
5. 분석 및 시각화
6. 주요 기능 및 사용 예
7. 향후 계획
8. 라이선스

---

## 1. 소개

**SignGlove Data Analysis** 프로젝트는 데이터 장갑(SignGlove)으로 수집한 제스처 데이터를 전처리하고, 탐색적 분석과 시각화를 수행하는 데 중점을 둡니다.
본 저장소는 수어(수화) 인식 연구의 데이터 분석 단계를 담당하며, 분석 코드와 시각화 도구를 제공합니다.

이 저장소의 목적은 다음과 같습니다:

* 센서 기반 제스처 데이터의 품질 확인
* 이상치 제거, 보정, 정규화 등의 전처리
* 다양한 그래프 및 통계 지표를 통한 데이터 이해
* 모델 학습 전 기초 인사이트 확보

---

## 2. 프로젝트 구조

```
SignGlove-DataAnalysis/
│
├── requirements.txt
│
├── summary statistics/  
│   └── … (요약 통계 관련 코드 및 리포트)  
├── data cleaning/  
│   └── … (데이터 정제, 이상치 처리 코드)  
├── data visualization/  
│   └── … (시각화 코드: 박스플롯, 히스토그램 등)  
├── unified/  
│   └── … (원본 센서 데이터)  
├── boxplot.py  
└── cleaned_dataset (보정).csv  
```

* `summary statistics/` : 각 제스처별 통계 요약, 분포 분석 코드
* `data cleaning/` : 누락치 처리, 이상치 탐지, 보정 알고리즘
* `data visualization/` : 시계열 플롯, 박스 플롯, 히트맵 등
* `unified/` : 아무런 전처리 없이 그대로 저장한 센서 데이터
* `boxplot.py` : 박스 플롯을 그리는 독립 실행 스크립트
* `cleaned_dataset (보정).csv` : 전처리 및 보정이 완료된 최종 데이터셋

---

## 3. 설치 및 실행 방법

### 요구 사항

* Python 3.7 이상
* 필수 라이브러리:

  * numpy
  * pandas
  * matplotlib
  * seaborn
  * (선택사항) scikit-learn 등

### 설치

```bash
git clone https://github.com/wodu2s/SignGlove-DataAnalysis.git
cd SignGlove-DataAnalysis
```

### 사용 예시

1. 전처리 실행

   ```bash
   python data_cleaning/xxx_cleaning_script.py
   pip install -r requirements.txt
   ```

2. 시각화 실행

   ```bash
   python data_visualization/yyy_visualize.py
   ```

3. 박스플롯 그리기

   ```bash
   python boxplot.py
   ```

> **팁**: 각 스크립트 내부에 `if __name__ == "__main__":` 블록이나 인수 파싱(argparse)이 있다면, 해당 옵션을 참고하세요.

---

## 4. 데이터 설명

* `cleaned_dataset (보정).csv` 파일이 본 프로젝트의 주요 데이터셋입니다.
* 각 행(row)은 하나의 측정 샘플을 나타내며, 각 열(column)은 센서 축, 손가락 굽힘 정도, 타임스탬프 등 다양한 센서 특성값으로 구성됩니다.
* 보정 값이 적용된 데이터로, 이상치 제거와 정규화 과정을 거친 상태입니다.

> **주의**: 원본(raw) 데이터는 이 저장소에 없거나 비공개일 수 있으므로, 원본 수집 코드나 전처리 과정을 별도로 보유하시는 것이 좋습니다.

---

## 5. 분석 및 시각화

이 저장소에는 다음과 같은 분석 및 시각화 기능이 포함되어 있습니다:

* 제스처별 분포 비교 (박스 플롯, 바이올린 플롯 등)
* 히스토그램 및 밀도 곡선
* 상관관계 히트맵 (센서 간 상관성)
* 시간 축에 따른 변화 플롯
* 이상치 탐지 결과 시각화

이 분석을 통해 데이터의 품질, 센서별 특성, 노이즈 또는 비정상 샘플 등을 탐지할 수 있습니다.

---

## 6. 주요 기능 및 사용 예

* **이상치 탐지 및 제거**: IQR 기반, Z-score 기반 등 다양한 방법
* **데이터 정규화/표준화**: Min-Max, Standard Scaler 등을 적용
* **데이터 통합**: 여러 실험 또는 다른 조건의 데이터를 하나의 통합 형식으로 병합
* **시각화 스크립트 활용**: `boxplot.py` 또는 `data_visualization` 내의 스크립트로 빠르게 그래프 생성
* **전처리 파이프라인 점검**: 각 단계의 중간 결과를 저장하고 비교 가능

예:

```python
from data_cleaning.cleaner import clean_data
import pandas as pd

df = pd.read_csv("raw_data.csv")
df_clean = clean_data(df)
df_clean.to_csv("cleaned.csv", index=False)
```

---

## 7. 향후 계획 / 개선 아이디어

* **원본(raw) 데이터 수집 코드 통합**
* **자동화 파이프라인 구축**
* **더 다양한 시각화 (예: 동적 플롯, 인터랙티브 대시보드)**
* **머신러닝 모델 훈련 전 탐색적 분석 확장**
* **이상치 탐지 알고리즘 고도화 (예: Isolation Forest, LOF 등)**
* **다양한 실험 조건 데이터 통합 및 비교 분석**

---

## 8. 라이선스

본 저장소는 별도의 라이선스 파일이 없는 상태이므로, 본인이 원하는 라이선스를 선택해 추가하시는 걸 권장드립니다. 예: MIT, Apache 2.0 등.
