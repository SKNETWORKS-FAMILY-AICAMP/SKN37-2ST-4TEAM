"""공통 train/valid/test 분할 데이터를 이용해 이탈 로지스틱 회귀 모델을 학습합니다."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# 스크립트와 결과물은 이 로지스틱 회귀 전용 폴더에 저장합니다.
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = (
    SCRIPT_DIR.parents[1]
    / "data"
    / "processed"
    / "customer_features_20181101_v1_with_split.csv"
)
MODEL_PATH = SCRIPT_DIR / "logistic_regression_v1.joblib"
RESULTS_PATH = SCRIPT_DIR / "logistic_regression_evaluation_results.txt"
COLLECTED_RESULTS_PATH = SCRIPT_DIR.parents[1] / "result" / RESULTS_PATH.name

# 데이터 설명서에 정의된 14개 입력 변수. 고객 ID, 정답(churn), 분할명(split)은 입력에서 제외합니다.
NUMERIC_FEATURES = [
    "active_days",
    "session_count",
    "song_count",
    "days_since_last_activity",
    "recent_7d_song_count",
    "previous_7d_song_count",
    "song_count_change",
    "thumbs_up_count",
    "thumbs_down_count",
    "playlist_add_count",
    "ad_count",
    "error_count",
    "help_count",
]
CATEGORICAL_FEATURES = ["level_last"]
TARGET = "churn"
SPLIT_COLUMN = "split"
RANDOM_STATE = 42


def evaluate_split(name: str, model: Pipeline, split_df: pd.DataFrame) -> str:
    """지정한 분할에서 성능을 계산해 화면과 결과 파일에 반환합니다."""
    X = split_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = split_df[TARGET]

    # churn=1 클래스의 예측 확률을 구합니다. 기본 임계값 0.5로 분류합니다.
    positive_class_index = list(model.classes_).index(1)
    probabilities = model.predict_proba(X)[:, positive_class_index]
    predictions = (probabilities >= 0.5).astype(int)

    report = [
        f"===== {name} 성능 ({len(split_df):,}명) =====",
        f"Accuracy:  {accuracy_score(y, predictions):.4f}",
        f"Precision: {precision_score(y, predictions, zero_division=0):.4f}",
        f"Recall:    {recall_score(y, predictions, zero_division=0):.4f}",
        f"F1-score:  {f1_score(y, predictions, zero_division=0):.4f}",
        f"ROC-AUC:   {roc_auc_score(y, probabilities):.4f}",
        f"PR-AUC:    {average_precision_score(y, probabilities):.4f}",
        "분류 리포트:",
        classification_report(y, predictions, zero_division=0),
        "혼동 행렬 (행=실제, 열=예측):",
        str(confusion_matrix(y, predictions)),
    ]
    report_text = "\n".join(report)
    print("\n" + report_text)
    return report_text


def main() -> None:
    # 예상 위치에서 통합 CSV를 읽습니다. 이 파일에는 train/valid/test 구분이 들어 있습니다.
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, dtype={"user_id": "string"})

    # 필요한 열이 있는지 먼저 검사해 잘못된 파일로 학습하는 것을 막습니다.
    required_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET, SPLIT_COLUMN]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"데이터에 필요한 열이 없습니다: {missing_columns}")

    # 제공된 공통 분할을 그대로 사용합니다. 전체 데이터를 다시 무작위 분할하지 않습니다.
    train_df = df[df[SPLIT_COLUMN] == "train"].copy()
    valid_df = df[df[SPLIT_COLUMN] == "valid"].copy()
    test_df = df[df[SPLIT_COLUMN] == "test"].copy()
    if train_df.empty or valid_df.empty or test_df.empty:
        raise ValueError("train, valid, test 데이터가 모두 있는지 확인해 주세요.")

    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X_train = train_df[feature_columns]
    y_train = train_df[TARGET]

    # 수치형은 학습 데이터 중앙값으로 결측값을 채운 뒤 표준화합니다.
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    # 범주형 level_last는 학습 데이터 최빈값으로 결측값을 채우고 원-핫 인코딩합니다.
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )

    # 이탈 고객이 전체의 소수이므로 class_weight로 학습 중 클래스 불균형을 보정합니다.
    model = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    # 스케일러와 인코더는 train 데이터만으로 학습되어 검증/테스트 정보가 섞이지 않습니다.
    model.fit(X_train, y_train)
    print(f"데이터 파일: {DATA_PATH}")
    print(
        f"분할 크기: train={len(train_df):,}, "
        f"valid={len(valid_df):,}, test={len(test_df):,}"
    )
    print(f"학습 이탈 비율: {y_train.mean():.2%}")

    # valid는 모델 점검용, test는 최종 일반화 성능 확인용으로 각각 평가합니다.
    # 각 분할의 평가 결과를 화면에 출력하고 전용 텍스트 파일에도 저장합니다.
    validation_report = evaluate_split("검증(valid)", model, valid_df)
    test_report = evaluate_split("최종 평가(test)", model, test_df)
    run_summary = [
        f"데이터 파일: {DATA_PATH}",
        f"분할 크기: train={len(train_df):,}, valid={len(valid_df):,}, test={len(test_df):,}",
        f"학습 이탈 비율: {y_train.mean():.2%}",
    ]
    RESULTS_PATH.write_text(
        "\n".join(run_summary + [validation_report, test_report]) + "\n",
        encoding="utf-8",
    )
    COLLECTED_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    COLLECTED_RESULTS_PATH.write_text(
        "\n".join(run_summary + [validation_report, test_report]) + "\n",
        encoding="utf-8",
    )

    # 전처리 과정까지 포함된 파이프라인을 저장해 추후 같은 방식으로 예측할 수 있습니다.
    joblib.dump(model, MODEL_PATH)
    print(f"평가 결과 저장: {RESULTS_PATH}")
    print(f"결과 모음 폴더에도 저장: {COLLECTED_RESULTS_PATH}")
    print(f"\n학습 모델 저장: {MODEL_PATH}")


if __name__ == "__main__":
    main()
