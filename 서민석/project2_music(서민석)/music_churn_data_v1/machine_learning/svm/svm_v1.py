"""공통 train/valid/test 분할 데이터로 선형 SVM 이탈 분류기를 학습합니다."""

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
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC


# 스크립트, 모델, 평가 TXT를 SVM 전용 폴더에 저장합니다.
SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR.parents[1]
DATA_PATH = (
    DATASET_DIR
    / "data"
    / "processed"
    / "customer_features_20181101_v1_with_split.csv"
)
MODEL_PATH = SCRIPT_DIR / "svm_v1.joblib"
RESULTS_PATH = SCRIPT_DIR / "svm_evaluation_results.txt"
COLLECTED_RESULTS_PATH = DATASET_DIR / "result" / RESULTS_PATH.name

# 데이터 정의서의 14개 입력 변수입니다. ID, 정답, 분할 열은 모델 입력에서 뺍니다.
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

# C 값이 클수록 학습 데이터 오분류를 더 강하게 줄이려는 후보가 됩니다.
C_VALUES = [0.01, 0.1, 1.0, 10.0]


def make_pipeline(c_value: float) -> Pipeline:
    """결측값 대체, 범주형 인코딩, 수치 표준화 후 선형 SVM을 연결합니다."""
    # 수치형 결측값은 중앙값으로 채우고, SVM 거리 계산에 맞게 표준화합니다.
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    # 범주형 결측값은 최빈값으로 채우고 원-핫 인코딩합니다.
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

    # LinearSVC는 확률 대신 decision score를 출력합니다.
    # balanced 가중치로 적은 이탈 클래스를 학습에서 보정합니다.
    classifier = LinearSVC(
        C=c_value,
        class_weight="balanced",
        dual=False,
        max_iter=10000,
        random_state=RANDOM_STATE,
    )
    return Pipeline(
        steps=[("preprocessing", preprocessing), ("classifier", classifier)]
    )


def choose_f1_threshold(y_true: pd.Series, scores: np.ndarray) -> tuple[float, float]:
    """valid decision score 중 F1이 가장 높은 임계값을 고릅니다."""
    # 이 임계값은 확률 기준이 아니라 LinearSVC의 decision score 기준입니다.
    thresholds = np.unique(np.concatenate(([scores.min() - 1e-9, 0.0, scores.max() + 1e-9], scores)))
    best_threshold = 0.0
    best_f1 = -1.0
    for threshold in thresholds:
        predictions = (scores >= threshold).astype(int)
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
    """분할별 지표를 계산해 TXT 보고서 내용과 숫자 지표를 반환합니다."""
    X = split_df[FEATURE_COLUMNS]
    y = split_df[TARGET]

    # LinearSVC decision score로 이탈 여부를 예측합니다.
    scores = model.decision_function(X)
    predictions = (scores >= threshold).astype(int)
    metrics = {
        "accuracy": accuracy_score(y, predictions),
        "precision": precision_score(y, predictions, zero_division=0),
        "recall": recall_score(y, predictions, zero_division=0),
        "f1": f1_score(y, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y, scores),
        "pr_auc": average_precision_score(y, scores),
    }
    report = [
        f"===== {split_name} 성능 ({len(split_df):,}명) =====",
        f"적용 decision score 임계값: {threshold:.6f} (확률이 아님)",
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
    # 공통 train/valid/test 분할 CSV를 읽고 필요한 열을 확인합니다.
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, dtype={"user_id": "string"})
    required_columns = FEATURE_COLUMNS + [TARGET, SPLIT_COLUMN]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"데이터에 필요한 열이 없습니다: {missing_columns}")

    # 이미 지정된 고객별 분할을 보존합니다.
    train_df = df[df[SPLIT_COLUMN] == "train"].copy()
    valid_df = df[df[SPLIT_COLUMN] == "valid"].copy()
    test_df = df[df[SPLIT_COLUMN] == "test"].copy()
    if train_df.empty or valid_df.empty or test_df.empty:
        raise ValueError("train, valid, test 데이터가 모두 있는지 확인해 주세요.")

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET]
    X_valid = valid_df[FEATURE_COLUMNS]
    y_valid = valid_df[TARGET]

    # C 후보별로 학습하고 valid PR-AUC가 가장 높은 설정을 선택합니다.
    best_model = None
    best_c_value = None
    best_valid_pr_auc = -1.0
    search_results = []
    for c_value in C_VALUES:
        candidate = make_pipeline(c_value)
        candidate.fit(X_train, y_train)
        valid_scores = candidate.decision_function(X_valid)
        valid_pr_auc = average_precision_score(y_valid, valid_scores)
        search_line = f"valid PR-AUC | C={c_value:g}: {valid_pr_auc:.4f}"
        search_results.append(search_line)
        print(search_line)

        if valid_pr_auc > best_valid_pr_auc:
            best_model = candidate
            best_c_value = c_value
            best_valid_pr_auc = float(valid_pr_auc)

    if best_model is None or best_c_value is None:
        raise RuntimeError("SVM 설정을 선택하지 못했습니다.")

    # 검증 데이터에서 F1 기준 score 임계값을 정하고 최종 test에 적용합니다.
    valid_scores = best_model.decision_function(X_valid)
    threshold, valid_best_f1 = choose_f1_threshold(y_valid, valid_scores)
    validation_report, _ = evaluate_split("검증(valid)", best_model, valid_df, threshold)
    test_report, _ = evaluate_split("최종 평가(test)", best_model, test_df, threshold)

    # 선형 SVM 계수의 절댓값이 큰 변환 피처 상위 15개를 기록합니다.
    preprocessing = best_model.named_steps["preprocessing"]
    classifier = best_model.named_steps["classifier"]
    feature_names = preprocessing.get_feature_names_out()
    coefficients = classifier.coef_[0]
    top_features = sorted(
        zip(feature_names, coefficients),
        key=lambda item: abs(item[1]),
        reverse=True,
    )[:15]
    summary = [
        f"데이터 파일: {DATA_PATH}",
        f"분할 크기: train={len(train_df):,}, valid={len(valid_df):,}, test={len(test_df):,}",
        f"학습 이탈 비율: {y_train.mean():.2%}",
        "모델: LinearSVC (선형 커널, 확률 추정 대신 decision score 사용)",
        f"선택된 C: {best_c_value:g}",
        f"valid PR-AUC: {best_valid_pr_auc:.4f}",
        f"valid에서 선택한 decision score 임계값: {threshold:.6f} (F1={valid_best_f1:.4f})",
        "계수 절댓값 상위 15개 피처 (부호는 예측 방향):",
    ]
    summary.extend(f"  {name}: {coefficient:+.4f}" for name, coefficient in top_features)
    full_report = "\n".join(summary + search_results + [validation_report, test_report])
    print("\n" + full_report)

    # 평가 결과는 모델 폴더와 중앙 result 폴더에 모두 기록합니다.
    COLLECTED_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(full_report + "\n", encoding="utf-8")
    COLLECTED_RESULTS_PATH.write_text(full_report + "\n", encoding="utf-8")

    # 전처리 파이프라인과 SVM, 임계값 및 C 설정을 재사용할 수 있도록 저장합니다.
    joblib.dump(
        {
            "model": best_model,
            "decision_score_threshold": threshold,
            "best_c": best_c_value,
            "feature_columns": FEATURE_COLUMNS,
        },
        MODEL_PATH,
    )
    print(f"\n평가 결과 저장: {RESULTS_PATH}")
    print(f"결과 모음 폴더에도 저장: {COLLECTED_RESULTS_PATH}")
    print(f"학습 모델 저장: {MODEL_PATH}")


if __name__ == "__main__":
    main()
