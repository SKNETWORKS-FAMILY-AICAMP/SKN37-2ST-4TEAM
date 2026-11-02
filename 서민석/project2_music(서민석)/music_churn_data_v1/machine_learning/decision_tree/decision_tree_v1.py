"""공통 train/valid/test 분할 데이터로 결정트리 이탈 분류기를 학습합니다."""

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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


# 스크립트, 학습 모델, 평가 결과를 결정트리 전용 폴더에 저장합니다.
SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR.parents[1]
DATA_PATH = (
    DATASET_DIR
    / "data"
    / "processed"
    / "customer_features_20181101_v1_with_split.csv"
)
MODEL_PATH = SCRIPT_DIR / "decision_tree_v1.joblib"
RESULTS_PATH = SCRIPT_DIR / "decision_tree_evaluation_results.txt"
COLLECTED_RESULTS_PATH = DATASET_DIR / "result" / RESULTS_PATH.name

# 데이터 정의서에 있는 입력 변수입니다. 고객 ID, 정답, 분할 열은 입력에서 제외합니다.
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
RANDOM_STATE = 42

# valid PR-AUC로 비교할 트리 깊이와 최소 리프 크기 설정 후보입니다.
PARAMETER_CANDIDATES = [
    {"max_depth": 5, "min_samples_leaf": 10, "criterion": "gini"},
    {"max_depth": 8, "min_samples_leaf": 10, "criterion": "gini"},
    {"max_depth": 12, "min_samples_leaf": 10, "criterion": "gini"},
    {"max_depth": None, "min_samples_leaf": 10, "criterion": "gini"},
    {"max_depth": 8, "min_samples_leaf": 25, "criterion": "entropy"},
    {"max_depth": 12, "min_samples_leaf": 25, "criterion": "entropy"},
]


def make_pipeline(parameters: dict) -> Pipeline:
    """결측값과 범주형 변수 전처리 후 결정트리를 연결합니다."""
    # 숫자형 결측값은 중앙값으로 대체합니다.
    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    # 범주형 결측값은 최빈값으로 대체하고 One-Hot 인코딩합니다.
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

    # 클래스 가중치로 적은 이탈 고객의 오분류를 더 크게 고려합니다.
    classifier = DecisionTreeClassifier(
        **parameters,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    return Pipeline(
        steps=[("preprocessing", preprocessing), ("classifier", classifier)]
    )


def choose_f1_threshold(y_true: pd.Series, probabilities: np.ndarray) -> tuple[float, float]:
    """valid 예측에서 F1이 가장 높은 분류 임계값을 선택합니다."""
    # 임계값은 검증 데이터로만 선택해 test 정보가 튜닝에 섞이지 않게 합니다.
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
    split_name: str,
    model: Pipeline,
    split_df: pd.DataFrame,
    threshold: float,
) -> tuple[str, dict]:
    """분할별 분류 성능을 계산해 TXT에 넣을 내용과 지표를 반환합니다."""
    X = split_df[FEATURE_COLUMNS]
    y = split_df[TARGET]
    positive_class_index = list(model.classes_).index(1)
    probabilities = model.predict_proba(X)[:, positive_class_index]
    predictions = (probabilities >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y, predictions),
        "precision": precision_score(y, predictions, zero_division=0),
        "recall": recall_score(y, predictions, zero_division=0),
        "f1": f1_score(y, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y, probabilities),
        "pr_auc": average_precision_score(y, probabilities),
    }
    report = [
        f"===== {split_name} 성능 ({len(split_df):,}명) =====",
        f"적용 임계값: {threshold:.6f}",
        f"Accuracy:  {metrics['accuracy']:.4f}",
        f"Precision: {metrics['precision']:.4f}",
        f"Recall:    {metrics['recall']:.4f}",
        f"F1-score:  {metrics['f1']:.4f}",
        f"ROC-AUC:   {metrics['roc_auc']:.4f}",
        f"PR-AUC:    {metrics['pr_auc']:.4f}",
        "분류 리포트:",
        classification_report(y, predictions, zero_division=0),
        "혼동 행렬 (행=실제, 열=예측):",
        str(confusion_matrix(y, predictions)),
    ]
    return "\n".join(report), metrics


def main() -> None:
    # 사전 분할이 들어 있는 통합 파일을 읽고 필수 열을 확인합니다.
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, dtype={"user_id": "string"})
    required_columns = FEATURE_COLUMNS + [TARGET, SPLIT_COLUMN]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"데이터에 필요한 열이 없습니다: {missing_columns}")

    # 제공된 고객 단위 train/valid/test 분할을 그대로 유지합니다.
    train_df = df[df[SPLIT_COLUMN] == "train"].copy()
    valid_df = df[df[SPLIT_COLUMN] == "valid"].copy()
    test_df = df[df[SPLIT_COLUMN] == "test"].copy()
    if train_df.empty or valid_df.empty or test_df.empty:
        raise ValueError("train, valid, test 데이터가 모두 있는지 확인해 주세요.")

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET]
    X_valid = valid_df[FEATURE_COLUMNS]
    y_valid = valid_df[TARGET]

    # 설정 후보별로 학습한 뒤 검증 PR-AUC가 가장 높은 모델을 고릅니다.
    best_model = None
    best_parameters = None
    best_valid_pr_auc = -1.0
    search_results = []
    for parameters in PARAMETER_CANDIDATES:
        candidate = make_pipeline(parameters)
        candidate.fit(X_train, y_train)
        positive_class_index = list(candidate.classes_).index(1)
        probabilities = candidate.predict_proba(X_valid)[:, positive_class_index]
        valid_pr_auc = average_precision_score(y_valid, probabilities)
        search_line = f"valid PR-AUC | {parameters}: {valid_pr_auc:.4f}"
        search_results.append(search_line)
        print(search_line)

        if valid_pr_auc > best_valid_pr_auc:
            best_model = candidate
            best_parameters = parameters
            best_valid_pr_auc = float(valid_pr_auc)

    if best_model is None or best_parameters is None:
        raise RuntimeError("결정트리 설정을 선택하지 못했습니다.")

    # valid에서 F1이 높은 임계값을 정하고 test에는 변경 없이 적용합니다.
    positive_class_index = list(best_model.classes_).index(1)
    valid_probabilities = best_model.predict_proba(X_valid)[:, positive_class_index]
    threshold, valid_best_f1 = choose_f1_threshold(y_valid, valid_probabilities)
    validation_report, _ = evaluate_split("검증(valid)", best_model, valid_df, threshold)
    test_report, _ = evaluate_split("최종 평가(test)", best_model, test_df, threshold)

    # 변환된 입력 열 중 결정트리가 중요하게 사용한 피처 상위를 기록합니다.
    preprocessing = best_model.named_steps["preprocessing"]
    classifier = best_model.named_steps["classifier"]
    feature_names = preprocessing.get_feature_names_out()
    top_features = sorted(
        zip(feature_names, classifier.feature_importances_),
        key=lambda item: item[1],
        reverse=True,
    )[:15]
    summary = [
        f"데이터 파일: {DATA_PATH}",
        f"분할 크기: train={len(train_df):,}, valid={len(valid_df):,}, test={len(test_df):,}",
        f"학습 이탈 비율: {y_train.mean():.2%}",
        f"선택된 설정: {best_parameters}",
        f"valid PR-AUC: {best_valid_pr_auc:.4f}",
        f"valid에서 선택한 임계값: {threshold:.6f} (F1={valid_best_f1:.4f})",
        "상위 15개 피처 중요도:",
    ]
    summary.extend(f"  {name}: {importance:.4f}" for name, importance in top_features)
    full_report = "\n".join(summary + search_results + [validation_report, test_report])
    print("\n" + full_report)

    # 결과를 모델 폴더와 중앙 result 폴더에 각각 저장합니다.
    COLLECTED_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(full_report + "\n", encoding="utf-8")
    COLLECTED_RESULTS_PATH.write_text(full_report + "\n", encoding="utf-8")

    # 전처리, 트리 모델, 임계값과 설정을 다음 예측에 사용할 수 있게 저장합니다.
    joblib.dump(
        {
            "model": best_model,
            "decision_threshold": threshold,
            "best_parameters": best_parameters,
            "feature_columns": FEATURE_COLUMNS,
        },
        MODEL_PATH,
    )
    print(f"\n평가 결과 저장: {RESULTS_PATH}")
    print(f"결과 모음 폴더에도 저장: {COLLECTED_RESULTS_PATH}")
    print(f"학습 모델 저장: {MODEL_PATH}")


if __name__ == "__main__":
    main()
