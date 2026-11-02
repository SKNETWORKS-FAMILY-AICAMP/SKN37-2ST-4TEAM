# CatBoost performance improvement v3

원본 70/15/15 고객 분할과 원본 CSV를 그대로 사용합니다. 두 실험을 같은 방식으로 비교합니다.

- 기본 14개 피처와 클래스 가중치 3종
- 기존 14개 피처에 행동률·비율·최근 변화 파생 피처 12개를 추가한 구성과 클래스 가중치 3종
- 각 구성에서 기존 기준 설정을 포함한 CatBoost 설정 5종씩 비교
- valid PR-AUC로 모델을 선택하고, valid에서 F1 최대 및 재현율 60%/70% 임계값을 찾음
- test는 최종 평가에만 사용

파생 피처는 스크립트 실행 중 메모리에서 계산하며 원본 데이터 파일은 수정하지 않습니다. 기존 실험 파일도 덮어쓰지 않습니다.

프로젝트 루트 터미널에서 실행:

```powershell
python "music_churn_data_v1/machine_learning/catboost/performance_improvement_v3/catboost_performance_improvement_v3.py"
```

결과 TXT는 이 폴더와 `music_churn_data_v1/result`에 저장됩니다.
