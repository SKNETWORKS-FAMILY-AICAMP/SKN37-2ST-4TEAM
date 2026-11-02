# Customer split v2

이 폴더는 `music_churn_data_v1/data/processed/customer_features_20181101_v1.csv`를 읽어 새로 나눈 데이터 복사본입니다. 원본 파일은 수정하지 않았습니다. 분할 결과 CSV는 `data` 하위 폴더에 모았습니다.

## 분할 방식

- 비율: train 80%, valid 10%, test 10%
- 방식: `churn` 0과 1을 각각 섞어 분할하는 층화 무작위 분할
- 난수 시드: 2026
- 각 CSV의 컬럼과 값은 원본 CSV 형식을 유지합니다.
- `customer_features_20181101_v2_with_split.csv`는 원본 컬럼에 `split` 컬럼을 덧붙인 통합 파일입니다.

## 파일

- `data/train_20181101_v2.csv`
- `data/valid_20181101_v2.csv`
- `data/test_20181101_v2.csv`
- `data/customer_features_20181101_v2_with_split.csv`
- `data/customer_split_manifest_v2.csv`: 고객 ID, 정답, 분할 이름
- `create_customer_split_v2.ps1`: 같은 분할을 다시 생성하는 스크립트
- `machine_learning/catboost/catboost_split_v2.py`: v2 분할 데이터용 CatBoost 학습·평가 스크립트
- `machine_learning/catboost/`: 실행 스크립트, 학습 모델, 상세 결과
- `result/`: v2 모델 결과 TXT 모음

## 해석 시 주의

학습 데이터가 기존 70%에서 80%로 늘어나 모델이 나아질 가능성은 있지만, 성능 향상이 보장되지는 않습니다. 검증·시험 데이터가 각각 10%로 줄어 기존보다 측정 결과의 변동성이 커질 수 있습니다. 기존 분할과 새 분할의 test 점수를 여러 번 비교해 더 높게 나온 분할을 고르면 평가 데이터에 맞춰 고르는 셈이므로, 새 분할은 별도 실험으로 기록하고 선택 기준은 valid에서 정해야 합니다.

이 분할은 같은 기준일의 고객을 무작위로 나눈 것입니다. 미래 시점의 고객에 대한 성능을 직접 측정하는 시간 기반 평가와는 다릅니다.
