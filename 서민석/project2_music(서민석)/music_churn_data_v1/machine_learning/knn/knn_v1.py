"""공통 train/valid/test 분할 데이터로 K-최근접이웃 이탈 분류기를 학습합니다."""

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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# 스크립트와 결과물은 이 KNN 전용 폴더에 저장합니다.
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = (
    SCRIPT_DIR.parents[1]
    / "data"
    / "processed"
    / "customer_features_20181101_v1_with_split.csv"
)
MODEL_PATH = SCRIPT_DIR / "knn_v1.joblib"
RESULTS_PATH = SCRIPT_DIR / "knn_evaluation_results.txt"
COLLECTED_RESULTS_PATH = SCRIPT_DIR.parents[1] / "result" / RESULTS_PATH.name

# 데이터 정의서의 입력 변수 14개입니다. user_id, churn, split은 모델 입력에서 제외합니다.
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

# valid에서 비교할 이웃 수와 이웃 투표 방식
K_VALUES = [5, 11, 21, 31, 51]
VOTE_WEIGHTS = ["uniform", "distance"]


def make_preprocessor() -> ColumnTransformer:
    """수치형과 범주형 변수를 KNN에 사용할 숫자 피처로 변환합니다."""
    # 수치형은 결측값을 중앙값으로 채우고 평균 0, 표준편차 1로 표준화합니다.
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # 범주형 변수는 최빈값으로 채운 뒤 원-핫 인코딩합니다.
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # 각 열 그룹에 맞는 변환기를 지정합니다.
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def choose_f1_threshold(y_true: pd.Series, probabilities: np.ndarray) -> tuple[float, float]:
    """valid 예측에서 F1 점수가 가장 높은 분류 임계값을 선택합니다."""
    # 여러 임계값을 비교하되 test 데이터는 임계값 선택에 사용하지 않습니다.
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
    """선택된 임계값으로 분류 성능을 계산해 화면과 결과 파일에 반환합니다."""
    X = split_df[FEATURE_COLUMNS]
    y = split_df[TARGET]

    # 이탈 클래스(1)의 예측 확률과 분류 결과를 계산합니다.
    positive_class_index = list(model.classes_).index(1)
    probabilities = model.predict_proba(X)[:, positive_class_index]
    predictions = (probabilities >= threshold).astype(int)

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
    report_text = "\n".join(report)
    print("\n" + report_text)
    return report_text


def main() -> None:
    # 사전에 만든 통합 CSV를 읽습니다. 고객 ID는 문자열로 유지합니다.
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, dtype={"user_id": "string"})

    # 학습 전에 파일 구조를 검사해 잘못된 입력을 조기에 알립니다.
    required_columns = FEATURE_COLUMNS + [TARGET, SPLIT_COLUMN]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"데이터에 필요한 열이 없습니다: {missing_columns}")

    # 제공된 고객 단위 분할을 보존합니다.
    train_df = df[df[SPLIT_COLUMN] == "train"].copy()
    valid_df = df[df[SPLIT_COLUMN] == "valid"].copy()
    test_df = df[df[SPLIT_COLUMN] == "test"].copy()
    if train_df.empty or valid_df.empty or test_df.empty:
        raise ValueError("train, valid, test 데이터가 모두 있는지 확인해 주세요.")

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET]
    X_valid = valid_df[FEATURE_COLUMNS]
    y_valid = valid_df[TARGET]

    # 전처리기는 train에만 맞춘 뒤 valid를 변환해 정보 누수를 방지합니다.
    preprocessor = make_preprocessor()
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_valid_transformed = preprocessor.transform(X_valid)

    # 각 이웃 수와 투표 방식을 학습하고 valid의 PR-AUC를 비교합니다.
    # PR-AUC는 이탈 고객처럼 양성 클래스가 적을 때 유용한 비교 지표입니다.
    best_config = None
    best_valid_pr_auc = -1.0
    search_results = []
    for k in K_VALUES:
        for weights in VOTE_WEIGHTS:
            candidate = KNeighborsClassifier(
                n_neighbors=k,
                weights=weights,
                metric="minkowski",
                p=2,
                n_jobs=-1,
            )
            candidate.fit(X_train_transformed, y_train)
            positive_class_index = list(candidate.classes_).index(1)
            valid_probabilities = candidate.predict_proba(X_valid_transformed)[:, positive_class_index]
            valid_pr_auc = average_precision_score(y_valid, valid_probabilities)
            search_line = f"valid PR-AUC | k={k:>2}, weights={weights:<8}: {valid_pr_auc:.4f}"
            search_results.append(search_line)
            print(search_line)

            if valid_pr_auc > best_valid_pr_auc:
                best_valid_pr_auc = float(valid_pr_auc)
                best_config = {"n_neighbors": k, "weights": weights}

    if best_config is None:
        raise RuntimeError("KNN 설정을 선택하지 못했습니다.")

    # 선택된 설정의 valid 확률로 F1을 가장 높이는 임계값을 정합니다.
    best_knn = KNeighborsClassifier(
        n_neighbors=best_config["n_neighbors"],
        weights=best_config["weights"],
        metric="minkowski",
        p=2,
        n_jobs=-1,
    )
    best_knn.fit(X_train_transformed, y_train)
    positive_class_index = list(best_knn.classes_).index(1)
    best_valid_probabilities = best_knn.predict_proba(X_valid_transformed)[:, positive_class_index]
    threshold, valid_best_f1 = choose_f1_threshold(y_valid, best_valid_probabilities)

    # 선택된 전처리와 KNN을 하나의 파이프라인으로 묶어 재사용 가능하게 합니다.
    model = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("classifier", best_knn),
        ]
    )

    summary = [
        f"데이터 파일: {DATA_PATH}",
        f"분할 크기: train={len(train_df):,}, valid={len(valid_df):,}, test={len(test_df):,}",
        f"선택된 설정: {best_config}",
        f"valid PR-AUC: {best_valid_pr_auc:.4f}",
        f"valid에서 선택한 임계값: {threshold:.6f} (F1={valid_best_f1:.4f})",
    ]
    print("\n" + "\n".join(summary))

    # 검증 성능을 보고하고, test는 최종 평가에만 사용합니다.
    validation_report = evaluate_split("검증(valid)", model, valid_df, threshold)
    test_report = evaluate_split("최종 평가(test)", model, test_df, threshold)
    RESULTS_PATH.write_text(
        "\n".join(summary + ["하이퍼파라미터 비교:"] + search_results + [validation_report, test_report]) + "\n",
        encoding="utf-8",
    )
    COLLECTED_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    COLLECTED_RESULTS_PATH.write_text(
        "\n".join(summary + ["하이퍼파라미터 비교:"] + search_results + [validation_report, test_report]) + "\n",
        encoding="utf-8",
    )

    # 임계값과 선택 설정을 모델과 함께 저장해 추론 때 동일한 기준을 사용합니다.
    joblib.dump(
        {
            "model": model,
            "decision_threshold": threshold,
            "best_config": best_config,
            "feature_columns": FEATURE_COLUMNS,
        },
        MODEL_PATH,
    )
    print(f"\n평가 결과 저장: {RESULTS_PATH}")
    print(f"결과 모음 폴더에도 저장: {COLLECTED_RESULTS_PATH}")
    print(f"\n학습 모델 저장: {MODEL_PATH}")


if __name__ == "__main__":
    main()
