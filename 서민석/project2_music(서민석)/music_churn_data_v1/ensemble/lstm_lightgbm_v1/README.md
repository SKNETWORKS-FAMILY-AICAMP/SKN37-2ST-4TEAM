# LSTM + LightGBM Ensemble v1

원본 `원본(22000명)/5_모델링.ipynb`의 late-fusion 구조를 현재 `music_churn_data_v1` 데이터와 학습 완료된 모델에 적용합니다. LightGBM 정형 예측과 LSTM 행동 시퀀스 예측 확률을 가중 평균합니다.

- LightGBM 가중치 후보: 0.70~0.90 (0.01 간격), LSTM은 나머지 비율
- 가중치 선택: valid ROC-AUC 최대
- 분류 임계값 선택: valid F2 최대 및 재현율 70%·80% 기준
- test는 마지막 성능 평가에만 사용
- 결과 TXT는 이 폴더와 `music_churn_data_v1/result`에 저장
- 모델 묶음, 설정 JSON, test 예측 CSV도 이 폴더에 저장

프로젝트 루트에서 기존 모델 파일과 필요한 라이브러리가 설치된 VS Code Python 환경으로 실행:

```powershell
python "music_churn_data_v1/ensemble/lstm_lightgbm_v1/ensemble_v1.py"
```

입력 모델은 `deep_learning/lstm/lstm_v1.pt`, `machine_learning/lightgbm/lightgbm_v1.joblib`입니다. 앙상블 실행은 이미 학습된 모델을 재사용하며 LSTM이나 LightGBM을 다시 학습하지 않습니다.
