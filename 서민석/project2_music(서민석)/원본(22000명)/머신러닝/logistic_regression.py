"""정제된 Parquet 데이터로 고객 이탈 여부를 예측하는 로지스틱 회귀 모델을 학습합니다."""

# 파일 경로를 운영체제에 맞게 다루기 위한 모듈
from pathlib import Path

# 학습된 모델 저장, Parquet 데이터 읽기
import joblib
import pandas as pd

# 수치형/범주형 피처 전처리와 결측값 대체 도구
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# 사용할 분류 모델과 평가 지표
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

# 데이터 분리, 전처리와 모델을 하나의 흐름으로 묶는 도구
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# 이 파이썬 파일이 들어 있는 머신러닝 폴더와 프로젝트 최상위 폴더
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

# 정제된 Parquet 파일을 찾을 위치 후보
DATA_CANDIDATES = [
    PROJECT_DIR / "데이터" / "ml_tabular_features_clean.parquet",
    SCRIPT_DIR / "데이터" / "ml_tabular_features_clean.parquet",
]

# 예측할 정답 열 이름과 결과를 재현할 때 사용할 난수 시드
TARGET = "is_churn"
RANDOM_STATE = 42


def main() -> None:
    # 후보 경로 중 실제로 존재하는 데이터 파일을 선택
    data_path = next((path for path in DATA_CANDIDATES if path.exists()), None)
    if data_path is None:
        # 파일을 찾지 못하면 확인한 경로를 오류 메시지에 함께 표시
        searched = "\n".join(str(path) for path in DATA_CANDIDATES)
        raise FileNotFoundError(f"정제된 Parquet 파일을 찾을 수 없습니다. 확인 경로:\n{searched}")

    # Parquet 파일을 표 형태의 DataFrame으로 읽고, 타깃 열 존재 여부 확인
    df = pd.read_parquet(data_path)
    if TARGET not in df.columns:
        raise ValueError(f"타깃 컬럼 '{TARGET}'이 Parquet 파일에 없습니다.")

    # 정답(y)과 입력 피처(X)를 분리
    y = df[TARGET]
    # 정답 열은 입력에서 제외합니다. userId는 식별자라 예측 피처로 쓰지 않고,
    # session_item_sequence는 별도의 시퀀스 모델용 데이터이므로 여기서 제외합니다.
    excluded = [column for column in [TARGET, "userId", "session_item_sequence"] if column in df.columns]
    X = df.drop(columns=excluded).copy()

    # 열 자료형에 따라 숫자형 피처와 범주형 피처를 나눕니다.
    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = X.select_dtypes(exclude="number").columns.tolist()

    # 각 피처 종류에 맞는 전처리 단계를 준비합니다.
    transformers = []
    if numeric_features:
        # 수치 결측값은 중앙값으로 채우고, 로지스틱 회귀에 맞게 표준화합니다.
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_features))
    if categorical_features:
        # 범주 결측값은 최빈값으로 채우고, 범주를 원-핫 숫자 열로 변환합니다.
        # 학습에서 보지 못한 범주가 평가 데이터에 있어도 오류 없이 처리합니다.
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_features))
    if not transformers:
        # 입력 피처가 하나도 없으면 모델을 만들 수 없으므로 중단합니다.
        raise ValueError("모델 학습에 사용할 피처가 없습니다.")

    # 열별 전처리를 합친 뒤 로지스틱 회귀 분류기를 연결합니다.
    preprocessing = ColumnTransformer(transformers=transformers)
    model = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,  # 반복 횟수 제한을 늘려 모델이 수렴할 시간을 줍니다.
                    class_weight="balanced",  # 클래스 비율이 불균형할 때 소수 클래스를 고려합니다.
                    random_state=RANDOM_STATE,  # 학습 결과 재현을 위한 시드
                ),
            ),
        ]
    )

    # 데이터를 학습용 80%, 평가용 20%로 나눕니다.
    # stratify=y로 두 데이터의 이탈/비이탈 비율을 비슷하게 유지합니다.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # 학습 데이터에 전처리와 로지스틱 회귀 모델을 학습합니다.
    model.fit(X_train, y_train)

    # 평가 데이터의 이탈 여부 예측과 이탈 클래스 확률을 계산합니다.
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    # 평가 데이터 크기와 모델 성능 지표를 출력합니다.
    print(f"데이터 파일: {data_path}")
    print(f"전체 데이터: {len(df):,}행 / 학습: {len(X_train):,}행 / 평가: {len(X_test):,}행")
    print(f"사용한 피처: {', '.join(X.columns)}")
    print(f"정확도(Accuracy): {accuracy_score(y_test, predictions):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, probabilities):.4f}")
    print("\n분류 리포트:")
    print(classification_report(y_test, predictions, zero_division=0))
    print("혼동 행렬 (행=실제, 열=예측):")
    print(confusion_matrix(y_test, predictions))

    # 전처리 단계까지 포함한 학습 모델을 나중에 재사용할 수 있도록 저장합니다.
    model_path = SCRIPT_DIR / "logistic_regression_model.joblib"
    joblib.dump(model, model_path)
    print(f"\n학습된 모델 저장: {model_path}")


# 이 파일을 직접 실행했을 때만 학습을 시작합니다.
if __name__ == "__main__":
    main()
