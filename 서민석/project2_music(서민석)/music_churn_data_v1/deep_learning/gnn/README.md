# GNN v1

v1의 고객 행동을 이벤트 그래프로 구성합니다. 시퀀스에서 시간상 이웃한 이벤트를 양방향 간선으로 연결하고, 그래프 합성곱으로 이탈 분류기를 학습합니다. 기존 train/valid/test 분할을 유지하며 검증 F2 최대값으로 최적 epoch와 기본 임계값을 선택하고, 재현율 70%·80% 기준도 기록합니다. 결과 TXT 사본은 `music_churn_data_v1/result`에 저장됩니다.

프로젝트 루트에서 PyTorch가 설치된 인터프리터로 실행:

```powershell
python "music_churn_data_v1/deep_learning/gnn/gnn_v1.py"
```

이 GNN은 고객 사이의 관계망을 학습하는 방식이 아니라, 고객 한 명의 시퀀스 안에서 연속 이벤트를 그래프로 연결하는 방식입니다.
