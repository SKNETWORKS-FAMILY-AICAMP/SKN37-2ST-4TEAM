# 🎧 Churn Care — 이탈 위험 고객 관리 서비스

최근 28일 이용 기록으로 이후 14일 안에 해지할 고객을 예측하고, 고객별 유지 활동을 추천하는 Streamlit 앱입니다.
음악 스트리밍 앱처럼 다크 테마, 플레이리스트형 헤더, 트랙 리스트형 고객 목록, 하단 플레이어 바(현재 운영 설정)로 꾸몄습니다.

## 실행

```bash
conda activate agent
pip install -r requirements-service.txt
streamlit run service/app.py        # 저장소 폴더에서 실행 → http://localhost:8501
```

- 처음 실행하면 팀 공통 데이터(`data/data/processed/customer_features_20181101_v1_with_split.csv`)로
  머신러닝·신경망 모델을 학습해 `model/service/`에 저장합니다 (10초 안팎, 처음 한 번만).
- 모델을 다시 만들려면 `model/service/` 폴더를 지우거나 `python service/core.py`를 실행합니다.
- 고객 자료 추가·갱신, 운영 설정, 저장 기록은 `service_data/`에 저장됩니다. 지우면 기본 상태로 돌아갑니다.
- `.gitignore`에 `model/service/`, `service_data/`를 넣어 두는 것을 권장합니다.

## 화면

| 메뉴 | 내용 |
|---|---|
| 🏠 홈 | 오늘 먼저 확인할 고객: 분석 고객 · 운영 기준의 이탈 예상 · 두 모델의 판정이 다른 고객, 두 모델 산점도, 현재 운영 설정, 우선 확인할 고객 10명 |
| 🎯 관리 목록 | 판정 기준·상위 N명·예산으로 선정, 판정 비교(일치/다름)·요금제 필터, CRM용 CSV |
| 👤 고객 상세 | 위험 점수 게이지, 머신러닝·신경망 점수, 추천 유지 활동, 행동별 영향, what-if |
| 💰 캠페인 | 고객 가치·비용·성공률·예산 → 이익이 가장 큰 연락 인원, 손익분기 정밀도 |
| 🗂️ 고객 자료 | 파일 추가·갱신(추가/교체, 같은 고객 번호 처리), 고객 직접 입력, 저장 기록, 전체 내려받기, 기본 자료로 되돌리기 |
| ⚙️ 모델 설정 | 머신러닝·신경망 모델 선택, 운영 기준(신경망/머신러닝/두 모델 결합), 결합 비중, 검증 판정 기준 사용, 미리 보기, 개발 시점 검증 성능 |
| 📊 성능 지표 | ① 모델이 잘 골랐나: 재현율 · 정밀도 · 재현율@관리 가능 인원 · PR-AUC · 오경고와 혼동행렬 / ② 대응 후 이탈이 줄었나: A/B 효과 계산기 |

## 모델과 점수

- 머신러닝: XGBoost · LightGBM(팀 노트북 설정) · CatBoost · HistGradientBoosting (설치된 것만)
- 신경망: 다층 신경망(MLP) 2종 — sklearn이라 추가 설치 불필요
- 입력: 팀 공통 피처 14개 (`user_id` 제외, `level_last`는 free=0 / paid=1)
- 분할: 팀 공통 분할. train 학습 / valid 조기 종료·판정 기준 / test 성능 보고만
- **위험 점수(0~100)**: 모델마다 확률 크기가 달라서, 검증 고객 확률 분포에서의 순위(백분위)로 바꿉니다.
  90점 = 검증 고객 90%보다 위험. 그래서 머신러닝과 신경망 점수를 같은 눈금으로 섞을 수 있습니다.
- **판정 기준**: 운영 설정마다 검증셋에서 F2(재현율 중시)가 최대인 점수로 자동 계산. 비중을 바꿔도 검증셋에서 다시 계산
- **판정 비교**: 각 모델을 단독으로 썼을 때의 판정이 같으면 '일치', 다르면 '다름'
- **고객별 이유**: 피처 하나를 일반 유지 고객 수준(학습셋 유지 고객 중앙값)으로 바꾸면 점수가 얼마나 내려가는지
- **추천 활동**: `service/core.py`의 `RULES`에 정의한 규칙 예시. 실제 효과는 '성능 지표 → 대응 효과'처럼 대조군을 두고 확인

## 팀 딥러닝·하이브리드를 서비스에 넣지 않은 이유

2026-10-30 저장소 기준입니다. 팀 딥러닝 입력이 고쳐지면 다시 검토합니다.

- 팀 딥러닝 노트북의 시퀀스 입력(`train.npy`·`valid.npy`·`test.npy`, 그 전에는 `dl_sequence_tensor.npy`)은 **마지막 칸 하나만으로 테스트 ROC-AUC 0.84**가 나옵니다. 관찰 기간의 행동만 들어 있다면 0.55 근처여야 합니다.
- 이 숫자들은 팀 공통 `customer_sequences_20181101_v1.jsonl`의 행동과 대응하지 않습니다. 같은 위치를 비교하면 어떤 숫자든 65~80%가 NextSong입니다.
- 공통 jsonl로 시퀀스를 다시 만들어 같은 구조의 LSTM을 학습하면 테스트 ROC-AUC는 0.60이었습니다. 이 LSTM과 섞은 하이브리드를 검증셋으로 골라도 LightGBM 단독(0.737)보다 낫지 않았습니다.
- 그래서 서비스는 정형 피처 14개로 학습한 머신러닝 4종과 다층 신경망 2종만 씁니다.

시퀀스 입력 점검 코드 (저장소 폴더에서 실행, 0.55 근처면 정상):

```python
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
D = "./data/data/split"
Xtr, Xte = np.load(f"{D}/train.npy"), np.load(f"{D}/test.npy")
ytr = pd.read_csv(f"{D}/train_20181101_v1.csv")["churn"].values
yte = pd.read_csv(f"{D}/test_20181101_v1.csv")["churn"].values
rate = pd.Series(ytr).groupby(Xtr[:, -1]).mean()           # 학습셋: 마지막 칸 값별 이탈률
score = pd.Series(Xte[:, -1]).map(rate).fillna(ytr.mean())
print(roc_auc_score(yte, score))
```
