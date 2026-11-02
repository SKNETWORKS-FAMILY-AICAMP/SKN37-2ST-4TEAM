# MLP v1

원본 v1 공통 70/15/15 분할을 사용해 정형 고객 피처로 MLP 이탈 예측을 학습합니다.

- 수치형 결측 대체와 표준화, 범주형 원-핫 인코더는 train에만 fit합니다.
- 클래스 불균형은 train에서 계산한 양성 클래스 가중치로 처리합니다.
- valid에서 임계값을 탐색한 뒤 F2가 가장 높은 epoch를 선택합니다. 기본 운영 임계값도 valid F2 최대 기준이며, 재현율 70%/80% 시나리오를 함께 기록합니다.
- PR-AUC와 ROC-AUC는 순위 성능을 확인하는 보조 지표로 기록합니다.
- test는 최종 평가에만 사용합니다.
- 원본 CSV와 split 지정은 수정하지 않습니다.

## 실행

프로젝트 루트에서 PyTorch가 설치된 Python 환경으로 실행합니다.

~~~powershell
python "music_churn_data_v1/deep_learning/mlp/mlp_v1.py"
~~~

스크립트 실행 후 이 폴더에 모델 가중치, 전처리기, 설정 JSON, 평가 TXT, test 예측 CSV가 저장됩니다. 평가 TXT 사본은 music_churn_data_v1/result에도 저장됩니다.

Codex 기본 .venv에는 PyTorch가 없어 여기서는 학습을 실행하지 않았습니다. VS Code에서 PyTorch가 설치된 인터프리터를 선택해 실행하세요.
