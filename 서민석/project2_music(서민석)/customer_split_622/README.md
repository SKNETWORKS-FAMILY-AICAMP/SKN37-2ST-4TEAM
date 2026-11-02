# Customer split 6:2:2

이 실험은 원본 `customer_features_20181101_v1.csv`를 읽어 churn 비율을 유지하는 층화 무작위 60/20/20 분할을 만듭니다. 원본 및 다른 분할 데이터는 수정하지 않습니다.

## 폴더 구성

- `data/`: 6:2:2 데이터 CSV 및 고객 분할 명단
- `machine_learning/catboost/`: 이 분할 전용 CatBoost 코드, 모델, 상세 결과
- `result/`: CatBoost 결과 TXT 모음
- `create_customer_split_622.ps1`: 같은 분할을 다시 만드는 스크립트

## 데이터 파일

- `data/train_20181101_622.csv`
- `data/valid_20181101_622.csv`
- `data/test_20181101_622.csv`
- `data/customer_features_20181101_622_with_split.csv`
- `data/customer_split_manifest_622.csv`

## 실행

분할 데이터 CSV는 이미 생성되어 있습니다. CatBoost 실행은 프로젝트 폴더 터미널에서 다음 명령을 사용합니다.

```powershell
python "customer_split_622/machine_learning/catboost/catboost_split_622.py"
```

학습 표본을 줄이고 valid/test를 각각 늘리는 실험입니다. 성능이 좋아진다고 보장되지는 않으며, 기존 분할과 test 고객이 달라 결과 차이를 분할 비율의 효과 하나로만 해석할 수 없습니다.
