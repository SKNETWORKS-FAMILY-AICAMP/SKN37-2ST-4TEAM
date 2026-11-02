# GRU v1

원본 v1 JSONL 행동 시퀀스로 GRU 이탈 예측을 학습합니다. 정형 CSV와 user_id, split, churn 라벨이 일치하는지 확인하고 기존 70/15/15 split을 그대로 사용합니다.

- 행동 사전은 train 고객의 시퀀스만으로 구성하고 새로운 행동은 UNK로 처리합니다.
- 최대 50개의 시간순 이벤트를 사용하고 짧은 시퀀스는 반복 없이 PAD로 오른쪽 패딩합니다.
- 실제 시퀀스 길이를 GRU에 전달해 PAD를 건너뜁니다.
- train 클래스 비율로 BCEWithLogitsLoss의 양성 가중치를 계산합니다.
- valid에서 임계값을 최적화한 F2로 epoch와 기본 임계값을 선택합니다.
- 재현율 70%/80% 목표의 임계값도 비교합니다. PR-AUC/ROC-AUC는 보조 지표로 기록합니다.
- test는 최종 평가에만 사용합니다.

프로젝트 루트에서 PyTorch가 설치된 Python 환경으로 실행:

~~~powershell
python "music_churn_data_v1/deep_learning/gru/gru_v1.py"
~~~

모델 가중치와 설정 JSON, 상세 결과 TXT, test 예측 CSV는 이 폴더에 저장됩니다. 결과 TXT 사본은 music_churn_data_v1/result에도 저장됩니다.

Codex 기본 .venv에는 PyTorch가 없어 여기서는 학습을 실행하지 않았습니다. VS Code에서 PyTorch가 설치된 인터프리터를 선택해 실행하세요.
