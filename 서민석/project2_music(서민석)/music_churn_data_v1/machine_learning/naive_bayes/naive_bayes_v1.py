"""공통 데이터 분할로 Gaussian Naive Bayes 이탈 분류기를 학습하고 평가합니다."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
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
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# 스크립트, 학습 모델, 실행 시 생성될 결과 파일을 이 모델 폴더에 저장합니다.
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = (
    SCRIPT_DIR.parents[1]
    / "data"
    / "processed"
    / "customer_features_20181101_v1_with_split.csv"
)
MODEL_PATH = SCRIPT_DIR / "naive_bayes_v1.joblib"
RESULTS_PATH = SCRIPT_DIR / "naive_bayes_evaluation_results.txt"
COLLECTED_RESULTS_PATH = SCRIPT_DIR.parents[1] / "result" / RESULTS_PATH.name

# 데이터 정의서의 입력 변수입니다. ID, 타깃, 분할 열은 입력에서 제외합니다.
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
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "churn"
SPLIT_COLUMN = "split"

# valid 데이터의 PR-AUC로 비교할 GaussianNB 분산 평활화 값
VAR_SMOOTHING_VALUES = [1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]


def make_preprocessor() -> ColumnTransformer:
    """결측값을 처리하고 범주형 변수를 숫자로 변환하는 전처리기를 만듭니다."""
    # 수치형 변수의 결측값을 중앙값으로 대체하고, 학습 데이터 기준으로 표준화합니다.
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # 범주형 변수의 결측값을 최빈값으로 대체한 뒤 원-핫 인코딩합니다.
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # GaussianNB는 밀집 배열을 사용하므로 변환 결과를 밀집 행렬로 반환하게 합니다.
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        sparse_threshold=0,
    )


def choose_f1_threshold(y_true: pd.Series, probabilities: np.ndarray) -> tuple[float, float]:
    """valid 예측으로 F1 점수가 가장 높은 분류 임계값을 선택합니다."""
    # 임계값은 valid에서만 선택하여 최종 test 평가에 정보가 섞이지 않도록 합니다.
    thresholds = np.unique(np.concatenate(([0.0, 0.5, 1.0], probabilities)))
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        score = f1_score(y_true, predictions, zero_division=0)
        if score > best_f1:
            best_threshold = float(threshold)
            best_f1 = float(score)
    return best_threshold, best_f1


def evaluate_split(
    name: str,
    model: Pipeline,
    split_df: pd.DataFrame,
    threshold: float,
) -> str:
    """지정한 분할의 성능을 계산해 화면과 결과 파일에 쓸 문자열로 반환합니다."""
    X = split_df[FEATURE_COLUMNS]
    y = split_df[TARGET]

    # 이탈(churn=1) 클래스의 확률을 구하고 선택된 임계값으로 클래스를 결정합니다.
    positive_class_index = list(model.classes_).index(1)
    probabilities = model.predict_proba(X)[:, positive_class_index]
    predictions = (probabilities >= threshold).astype(int)

    # 주요 분류 지표와 상세 리포트를 구성합니다.
    report = [
        f"===== {name} 성능 ({len(split_df):,}명) =====",
        f"적용 임계값: {threshold:.6f}",
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
    return "\n".join(report)


def main() -> None:
    # 고객별 피처와 사전에 정해진 split이 포함된 CSV를 불러옵니다.
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, dtype={"user_id": "string"})

    # 학습 데이터에 필요한 모든 입력 및 관리 열이 있는지 검사합니다.
    required_columns = FEATURE_COLUMNS + [TARGET, SPLIT_COLUMN]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"데이터에 필요한 열이 없습니다: {missing_columns}")

    # 이미 만들어진 train/valid/test 구분을 유지하고, 전체 데이터를 다시 섞지 않습니다.
    train_df = df[df[SPLIT_COLUMN] == "train"].copy()
    valid_df = df[df[SPLIT_COLUMN] == "valid"].copy()
    test_df = df[df[SPLIT_COLUMN] == "test"].copy()
    if train_df.empty or valid_df.empty or test_df.empty:
        raise ValueError("train, valid, test 데이터가 모두 있는지 확인해 주세요.")

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET]
    X_valid = valid_df[FEATURE_COLUMNS]
    y_valid = valid_df[TARGET]

    # 전처리 기준은 train에서만 학습하고, 같은 기준으로 valid를 변환합니다.
    preprocessor = make_preprocessor()
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_valid_transformed = preprocessor.transform(X_valid)

    # 분산 평활화 설정을 비교해 valid PR-AUC가 가장 높은 모델을 선택합니다.
    best_model = None
    best_var_smoothing = None
    best_valid_pr_auc = -1.0
    for var_smoothing in VAR_SMOOTHING_VALUES:
        candidate = GaussianNB(var_smoothing=var_smoothing)
        candidate.fit(X_train_transformed, y_train)
        positive_class_index = list(candidate.classes_).index(1)
        valid_probabilities = candidate.predict_proba(X_valid_transformed)[:, positive_class_index]
        valid_pr_auc = average_precision_score(y_valid, valid_probabilities)
        print(f"valid PR-AUC | var_smoothing={var_smoothing:g}: {valid_pr_auc:.4f}")

        if valid_pr_auc > best_valid_pr_auc:
            best_model = candidate
            best_var_smoothing = var_smoothing
            best_valid_pr_auc = float(valid_pr_auc)

    if best_model is None or best_var_smoothing is None:
        raise RuntimeError("Naive Bayes 설정을 선택하지 못했습니다.")

    # valid에서 이탈 고객 F1을 가장 높이는 임계값을 정합니다.
    positive_class_index = list(best_model.classes_).index(1)
    best_valid_probabilities = best_model.predict_proba(X_valid_transformed)[:, positive_class_index]
    threshold, valid_best_f1 = choose_f1_threshold(y_valid, best_valid_probabilities)

    # 전처리기와 선택된 분류기를 하나의 파이프라인으로 묶어 저장합니다.
    model = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("classifier", best_model),
        ]
    )

    # 선택 설정과 분할 규모를 결과 파일에 기록합니다.
    summary = [
        f"데이터 파일: {DATA_PATH}",
        f"분할 크기: train={len(train_df):,}, valid={len(valid_df):,}, test={len(test_df):,}",
        f"선택된 var_smoothing: {best_var_smoothing:g}",
        f"valid PR-AUC: {best_valid_pr_auc:.4f}",
        f"valid에서 선택한 임계값: {threshold:.6f} (F1={valid_best_f1:.4f})",
    ]

    # valid 성능을 확인하고, test는 최종 평가에만 사용합니다.
    validation_report = evaluate_split("검증(valid)", model, valid_df, threshold)
    test_report = evaluate_split("최종 평가(test)", model, test_df, threshold)
    full_report = "\n".join(summary + [validation_report, test_report])
    print("\n" + full_report)
    RESULTS_PATH.write_text(full_report + "\n", encoding="utf-8")
    COLLECTED_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    COLLECTED_RESULTS_PATH.write_text(full_report + "\n", encoding="utf-8")

    # 추후 예측에 쓸 전처리기, 모델, 임계값, 피처 순서를 함께 저장합니다.
    joblib.dump(
        {
            "model": model,
            "decision_threshold": threshold,
            "var_smoothing": best_var_smoothing,
            "feature_columns": FEATURE_COLUMNS,
        },
        MODEL_PATH,
    )
    print(f"\n결과 저장: {RESULTS_PATH}")
    print(f"결과 모음 폴더에도 저장: {COLLECTED_RESULTS_PATH}")
    print(f"학습 모델 저장: {MODEL_PATH}")


if __name__ == "__main__":
    main()
