"""공통 train/valid/test 분할 데이터로 LightGBM 이탈 분류기를 학습합니다."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
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


# 스크립트, 학습 모델, 결과 TXT를 LightGBM 전용 폴더에 저장합니다.
SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR.parents[1]
DATA_PATH = (
    DATASET_DIR
    / "data"
    / "processed"
    / "customer_features_20181101_v1_with_split.csv"
)
MODEL_PATH = SCRIPT_DIR / "lightgbm_v1.joblib"
RESULTS_PATH = SCRIPT_DIR / "lightgbm_evaluation_results.txt"
COLLECTED_RESULTS_PATH = DATASET_DIR / "result" / RESULTS_PATH.name

# 데이터 정의서에 있는 14개 입력 변수입니다. ID, 정답, 분할 열은 제외합니다.
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

# valid PR-AUC로 선택할 트리 수와 트리 복잡도 후보입니다.
PARAMETER_CANDIDATES = [
    {"n_estimators": 400, "max_depth": 3, "num_leaves": 7, "learning_rate": 0.05, "min_child_samples": 20},
    {"n_estimators": 600, "max_depth": 4, "num_leaves": 15, "learning_rate": 0.03, "min_child_samples": 30},
    {"n_estimators": 500, "max_depth": 5, "num_leaves": 31, "learning_rate": 0.05, "min_child_samples": 30},
    {"n_estimators": 700, "max_depth": 6, "num_leaves": 31, "learning_rate": 0.03, "min_child_samples": 50},
    {"n_estimators": 500, "max_depth": -1, "num_leaves": 15, "learning_rate": 0.05, "min_child_samples": 50},
    {"n_estimators": 800, "max_depth": -1, "num_leaves": 31, "learning_rate": 0.03, "min_child_samples": 50},
]


def make_pipeline(parameters: dict, scale_pos_weight: float) -> Pipeline:
    """결측값·범주형 변수 처리와 LightGBM 분류기를 연결합니다."""
    # 수치형 결측값은 중앙값으로, 범주형 결측값은 최빈값으로 대체합니다.
    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
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

    # scale_pos_weight로 소수 이탈 클래스의 학습 가중치를 조정합니다.
    classifier = LGBMClassifier(
        **parameters,
        objective="binary",
        scale_pos_weight=scale_pos_weight,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        importance_type="gain",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )
    return Pipeline(
        steps=[("preprocessing", preprocessing), ("classifier", classifier)]
    )


def choose_f1_threshold(y_true: pd.Series, probabilities: np.ndarray) -> tuple[float, float]:
    """valid에서 F1이 가장 높은 이탈 확률 임계값을 선택합니다."""
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
    """분할별 평가 지표와 혼동 행렬을 TXT용 문자열로 반환합니다."""
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
    # 고객별 사전 분할 파일을 읽고 사용할 열이 있는지 검사합니다.
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, dtype={"user_id": "string"})
    required_columns = FEATURE_COLUMNS + [TARGET, SPLIT_COLUMN]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"데이터에 필요한 열이 없습니다: {missing_columns}")

    # 준비된 train/valid/test 분할을 그대로 사용합니다.
    train_df = df[df[SPLIT_COLUMN] == "train"].copy()
    valid_df = df[df[SPLIT_COLUMN] == "valid"].copy()
    test_df = df[df[SPLIT_COLUMN] == "test"].copy()
    if train_df.empty or valid_df.empty or test_df.empty:
        raise ValueError("train, valid, test 데이터가 모두 있는지 확인해 주세요.")

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET]
    X_valid = valid_df[FEATURE_COLUMNS]
    y_valid = valid_df[TARGET]

    # 이탈/비이탈 가중치 비율은 train의 정답만으로 계산합니다.
    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())
    if positive_count == 0:
        raise ValueError("train 분할에 이탈 고객(churn=1)이 없습니다.")
    scale_pos_weight = negative_count / positive_count

    # 설정 후보별 학습 결과를 valid PR-AUC로 비교합니다.
    best_model = None
    best_parameters = None
    best_valid_pr_auc = -1.0
    search_results = []
    for parameters in PARAMETER_CANDIDATES:
        candidate = make_pipeline(parameters, scale_pos_weight)
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
        raise RuntimeError("LightGBM 설정을 선택하지 못했습니다.")

    # valid에서 F1 기준 임계값을 선택하고 test에는 그대로 적용합니다.
    positive_class_index = list(best_model.classes_).index(1)
    valid_probabilities = best_model.predict_proba(X_valid)[:, positive_class_index]
    threshold, valid_best_f1 = choose_f1_threshold(y_valid, valid_probabilities)
    validation_report, _ = evaluate_split("검증(valid)", best_model, valid_df, threshold)
    test_report, _ = evaluate_split("최종 평가(test)", best_model, test_df, threshold)

    # 변환된 피처 이름과 LightGBM gain 중요도를 연결합니다.
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
        f"scale_pos_weight (train 음성/양성 비율): {scale_pos_weight:.4f}",
        f"선택된 설정: {best_parameters}",
        f"valid PR-AUC: {best_valid_pr_auc:.4f}",
        f"valid에서 선택한 임계값: {threshold:.6f} (F1={valid_best_f1:.4f})",
        "상위 15개 피처 중요도 (gain):",
    ]
    summary.extend(f"  {name}: {importance:.4f}" for name, importance in top_features)
    full_report = "\n".join(summary + search_results + [validation_report, test_report])
    print("\n" + full_report)

    # 평가 결과를 전용 폴더와 중앙 result 폴더에 함께 저장합니다.
    COLLECTED_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(full_report + "\n", encoding="utf-8")
    COLLECTED_RESULTS_PATH.write_text(full_report + "\n", encoding="utf-8")

    # 모델과 재사용할 임계값, 설정, 입력 피처 순서를 저장합니다.
    joblib.dump(
        {
            "model": best_model,
            "decision_threshold": threshold,
            "best_parameters": best_parameters,
            "scale_pos_weight": scale_pos_weight,
            "feature_columns": FEATURE_COLUMNS,
        },
        MODEL_PATH,
    )
    print(f"\n평가 결과 저장: {RESULTS_PATH}")
    print(f"결과 모음 폴더에도 저장: {COLLECTED_RESULTS_PATH}")
    print(f"학습 모델 저장: {MODEL_PATH}")


if __name__ == "__main__":
    main()
