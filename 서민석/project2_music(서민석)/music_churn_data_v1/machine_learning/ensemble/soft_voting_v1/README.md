# Soft voting ensemble v1

원본 70/15/15 공통 분할에서 이미 학습한 CatBoost, XGBoost, LightGBM 모델의 예측 확률을 결합하는 실험입니다.

## 실행

프로젝트 폴더 터미널에서 실행합니다.

```powershell
python "music_churn_data_v1/machine_learning/ensemble/soft_voting_v1/ensemble_v1.py"
```

기존 세 모델 파일이 필요합니다. valid PR-AUC 기준으로 제한된 7개 조합 중 가중치를 고르고, valid에서 F1 최대 및 재현율 60%/70% 목표 임계값을 선택합니다. test는 최종 평가에만 사용합니다.

결과 TXT는 이 폴더와 `music_churn_data_v1/result`에 저장되고, 선택된 앙상블 구성은 joblib 파일로 저장됩니다. 앙상블이 단일 모델보다 항상 좋아지는 것은 아니며, test 결과로 가중치나 임계값을 재조정하지 않습니다.
