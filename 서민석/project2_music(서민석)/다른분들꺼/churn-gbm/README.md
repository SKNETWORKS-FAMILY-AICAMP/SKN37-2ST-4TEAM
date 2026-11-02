# 음악 스트리밍 고객 이탈 예측 — gradient boosting

팀 공통 데이터(`music_churn_data_v1`)의 고객별 피처 표로 **gradient boosting 모델만** 학습·비교하는 VS Code 프로젝트입니다.

- 문제: 최근 28일(2018-10-04 ~ 11-01) 이용 행동으로 이후 14일(~ 11-15) 안의 해지 예측
- 데이터: 18,139명, 피처 14개, 이탈률 5.71%, 팀 공통 분할 train 12,697 / valid 2,721 / test 2,721
- 모델: HistGradientBoosting, GradientBoosting (sklearn 내장), LightGBM, XGBoost, CatBoost (설치돼 있으면)
- 결과: `reports/학습_결과서.md`와 그림까지 들어 있는 `reports/학습_결과서.html` 자동 생성, 최종 모델 `models/best_model.joblib` + `models/model_card.json`

> 압축 파일에는 HistGradientBoosting·GradientBoosting 두 모델로 먼저 실행한 결과가 들어 있습니다.
> LightGBM·XGBoost·CatBoost를 설치하고 `python -m churn all`을 다시 실행하면 다섯 모델을 비교한 결과로 바뀝니다.
> 어떤 모델이 라이브러리 버전 문제로 실패해도 나머지 모델은 계속 학습하고, 실패 내용은 결과서에 적힙니다.

---

## 1. 설치

```bash
conda activate agent
pip install -r requirements.txt
python -m churn check        # 설치된 패키지와 데이터 파일 확인
```

VS Code에서 **파일 → 폴더 열기**로 이 폴더를 열고, `Ctrl+Shift+P` → **Python: Select Interpreter**에서 `agent` 환경을 고릅니다.

## 2. 실행

| 방법 | 하는 법 |
|---|---|
| 실행 버튼 | 실행 및 디버그(`Ctrl+Shift+D`) → 드롭다운에서 `▶ 전체 실행` → ▶ |
| 단축키 | `Ctrl+Shift+B` = 전체 실행 |
| 터미널 | `python -m churn all` |

```bash
python -m churn all                        # eda → train → evaluate → explain → report
python -m churn report                     # HTML 결과서만 다시 만들기
python -m churn all --quick                # 탐색 5회·2겹으로 빠르게 (코드 확인용)
python -m churn all --models lightgbm xgboost catboost   # 일부 모델만
python -m churn all --from evaluate        # 평가부터 다시
python -m churn predict --input 새고객.csv  # 저장된 최종 모델로 예측
python -m pytest -q                        # 테스트
```

## 3. 폴더 구조

```
churn-gbm/
├─ configs/config.yaml     피처 목록, 모델, 튜닝, 임계값 설정
├─ data/                   팀 공통 피처 표 + README_DATA.md + 해시 목록
├─ src/churn/
│  ├─ data.py              데이터 읽기·점검 (분할 유지, 값 변경 없음, sha256 확인)
│  ├─ eda.py               1 EDA (학습셋만)
│  ├─ preprocess.py          전처리 (결측 대비, level_last 인코딩)
│  ├─ models.py              gradient boosting 5종과 탐색 범위
│  ├─ train.py             2 학습·튜닝 (교차검증)
│  ├─ evaluate.py          3 평가·임계값·최종 모델·학습 결과서
│  ├─ explain.py           4 순열 중요도·부분 의존도·SHAP
│  ├─ report_html.py       5 결과서 md → 그림 포함 HTML 한 파일
│  └─ predict.py             새 고객 예측
├─ tests/test_gbm.py
├─ models/                 학습된 모델, 모델 카드, 예측값
└─ reports/                학습 결과서, 그림, 표
```

## 4. 팀 데이터 기준을 지킨 방법

| 팀 기준 (README_DATA.md) | 이 프로젝트 |
|---|---|
| 공통 분할 유지, 다시 나누지 않음 | `split` 열을 그대로 사용, 테스트로 고객 수·이탈 수 확인 |
| user_id는 문자열, 입력에서 제외 | 문자열로 읽고 피처 14개만 입력 |
| 전처리 기준은 학습 데이터로만 | sklearn `Pipeline`에 넣어 학습셋(교차검증 때는 학습 폴드)으로만 계산 |
| 0인 횟수·음수 변화량을 바꾸지 않음 | 값 변경 없음 (테스트로 확인) |
| 큰 값만으로 자르거나 삭제하지 않음 | 이상치 처리 없음 — 트리 모델은 극단값 영향이 작음 |
| 최종 평가 데이터로 튜닝·임계값 선택 금지 | 튜닝은 학습셋 교차검증, 임계값·모델 선택은 검증셋, 테스트는 마지막 1회 |
| 전처리기·피처 순서·설정을 모델과 함께 저장 | `best_model.joblib`(전처리 포함) + `model_card.json`(피처 순서, 인코딩, 중앙값, 파라미터, 임계값, 데이터 해시, 버전) |
| SHAP은 원인 증명이 아님 | 결과서에 명시 |

## 5. 평가 항목별 위치 (`reports/학습_결과서.md`)

| 평가 항목 | 결과서 위치 |
|---|---|
| EDA | 2장 (`reports/eda/` 그림·표) |
| 결측치·이상치·인코딩·스케일링 | 3장 (처리와 이유) |
| 하이퍼파라미터 튜닝 | 4장 (기본값 vs 튜닝, 교차검증·검증 점수) |
| 과적합 확인·해결 | 5장 (학습-검증 차이, 트리 개수별 곡선, 학습 곡선) |
| 적절한 지표 | 2·6·8장 (PR-AUC 주 지표, 95% 구간, 상위 k% 포착률) |
| 모델 해석 | 10장 |

## 6. 결과서 보기·공유

- `reports/학습_결과서.html`: 그림이 파일 안에 들어 있어 **이 파일 하나만** 보내면 됩니다. 아무 브라우저에서 더블클릭으로 열립니다.
- PDF가 필요하면 브라우저에서 `Ctrl+P` → 대상 'PDF로 저장'. A4에 맞게 표·그림이 페이지 중간에서 잘리지 않도록 해 두었습니다.
- md 파일은 VS Code(`Ctrl+Shift+V`), GitHub, Typora·Obsidian 등에서 볼 수 있습니다. 이때는 `reports` 폴더째로 옮겨야 그림이 보입니다.

## 7. 설정 바꾸기 (`configs/config.yaml`)

| 하고 싶은 것 | 바꿀 값 |
|---|---|
| 모델 빼기·넣기 | `models` |
| 튜닝을 더 많이 | `tuning.n_iter` (예: 50), `tuning.cv_folds` |
| 놓치지 않기 우선 / 헛연락 줄이기 우선 | `evaluation.beta` (크면 재현율 중시) 또는 `threshold_strategy: target_recall` |
| 다른 버전 데이터 | `paths.data` |

## 8. 새 고객 예측

```python
import joblib, json, pandas as pd
model = joblib.load("models/best_model.joblib")
card = json.load(open("models/model_card.json", encoding="utf-8"))
X = pd.read_csv("새고객.csv")[card["input_columns"]]
prob = model.predict_proba(X)[:, 1]
pred = (prob >= card["threshold"]).astype(int)
```

또는 `python -m churn predict --input 새고객.csv` (결과: `새고객_예측.csv`).
