# CatBoost recall optimization v4

원본 70/15/15 분할과 원본 CSV를 유지하며, 이탈 고객 재현율을 우선하는 CatBoost 실험입니다.

- 기본 14개 피처와 행동 비율 파생 피처 추가 버전을 비교
- CatBoost 파라미터 5종과 클래스 가중치 3종을 비교
- valid에서 F2가 가장 높은 설정과 임계값을 선택 (F2는 재현율에 더 큰 비중)
- valid에서 재현율 70%, 75%, 80%, 85%, 90% 목표 임계값도 계산
- test는 최종 평가에만 사용

원본 CSV는 읽기만 하며 수정하지 않습니다. 오경고가 얼마나 늘어나는지도 혼동 행렬과 정밀도로 함께 기록합니다.

프로젝트 루트 터미널에서 실행:

~~~powershell
python "music_churn_data_v1/machine_learning/catboost/recall_optimization_v4/catboost_recall_optimization_v4.py"
~~~

결과 TXT는 이 폴더와 music_churn_data_v1/result에 저장됩니다.
