"""공통 train/valid/test 분할 데이터로 CatBoost 이탈 분류기를 학습하고 평가합니다."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
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


# 스크립트, 모델, 평가 결과를 CatBoost 전용 폴더에 함께 저장합니다.
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = (
    SCRIPT_DIR.parents[1]
    / "data"
    / "processed"
    / "customer_features_20181101_v1_with_split.csv"
)
MODEL_PATH = SCRIPT_DIR / "catboost_v1.joblib"
RESULTS_PATH = SCRIPT_DIR / "catboost_evaluation_results.txt"
COLLECTED_RESULTS_PATH = SCRIPT_DIR.parents[1] / "result" / RESULTS_PATH.name

# 데이터 정의서에 지정된 입력 피처입니다. user_id, churn, split은 모델 입력에서 제외합니다.
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

# valid PR-AUC로 비교할 학습 설정 조합입니다.
PARAMETER_CANDIDATES = [
    {"iterations": 500, "depth": 4, "learning_rate": 0.05, "l2_leaf_reg": 3},
    {"iterations": 800, "depth": 4, "learning_rate": 0.03, "l2_leaf_reg": 5},
    {"iterations": 500, "depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 5},
    {"iterations": 800, "depth": 6, "learning_rate": 0.03, "l2_leaf_reg": 7},
    {"iterations": 500, "depth": 8, "learning_rate": 0.05, "l2_leaf_reg": 7},
    {"iterations": 800, "depth": 8, "learning_rate": 0.03, "l2_leaf_reg": 9},
]
RANDOM_STATE = 42


def choose_f1_threshold(y_true: pd.Series, probabilities: np.ndarray) -> tuple[float, float]:
    """valid 데이터에서 F1이 가장 높은 이탈 판정 임계값을 선택합니다."""
    # test는 임계값 선택에 사용하지 않도록 valid 예측 확률만 비교합니다.
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
    model: CatBoostClassifier,
    split_df: pd.DataFrame,
    threshold: float,
) -> str:
    """한 분할의 분류 성능을 계산하고, 화면 및 결과 파일에 쓸 문자열을 반환합니다."""
    X = split_df[FEATURE_COLUMNS]
    y = split_df[TARGET]

    # 예측 확률을 얻고 valid에서 선택한 임계값으로 이탈 여부를 판정합니다.
    positive_class_index = list(model.classes_).index(1)
    probabilities = model.predict_proba(X)[:, positive_class_index]
    predictions = (probabilities >= threshold).astype(int)

    # 정확도 외에 소수 클래스인 이탈 고객 성능을 나타내는 지표를 함께 기록합니다.
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
    # 고객별 피처와 사전에 정해진 train/valid/test 분할이 들어 있는 CSV를 읽습니다.
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, dtype={"user_id": "string"})

    # 필요한 열이 모두 있는지 검사합니다.
    required_columns = FEATURE_COLUMNS + [TARGET, SPLIT_COLUMN]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"데이터에 필요한 열이 없습니다: {missing_columns}")

    # 원본에서 만든 공통 분할을 유지하고 새로 섞거나 나누지 않습니다.
    train_df = df[df[SPLIT_COLUMN] == "train"].copy()
    valid_df = df[df[SPLIT_COLUMN] == "valid"].copy()
    test_df = df[df[SPLIT_COLUMN] == "test"].copy()
    if train_df.empty or valid_df.empty or test_df.empty:
        raise ValueError("train, valid, test 데이터가 모두 있는지 확인해 주세요.")

    # 입력에서는 고객 ID와 정답, 분할 표시를 제외하고 정의된 피처만 사용합니다.
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET]
    X_valid = valid_df[FEATURE_COLUMNS]
    y_valid = valid_df[TARGET]

    # CatBoost가 level_last를 범주형으로 직접 처리하도록 문자열 열을 지정합니다.
    # 이탈 클래스 비중이 낮아 Balanced 가중치로 학습 때 소수 클래스를 보정합니다.
    search_results = []
    best_model = None
    best_parameters = None
    best_valid_pr_auc = -1.0

    # 각 설정을 train으로 학습하고 valid PR-AUC를 기준으로 하나를 선택합니다.
    for candidate_parameters in PARAMETER_CANDIDATES:
        candidate = CatBoostClassifier(
            **candidate_parameters,
            loss_function="Logloss",
            eval_metric="PRAUC:type=Classic",
            auto_class_weights="Balanced",
            random_seed=RANDOM_STATE,
            allow_writing_files=False,
            thread_count=-1,
            verbose=False,
        )
        candidate.fit(
            X_train,
            y_train,
            cat_features=CATEGORICAL_FEATURES,
            eval_set=(X_valid, y_valid),
            early_stopping_rounds=100,
            use_best_model=True,
            verbose=False,
        )

        # 설정 선택용 지표는 sklearn의 PR-AUC로 계산해 다른 모델과 비교하기 쉽게 합니다.
        positive_class_index = list(candidate.classes_).index(1)
        valid_probabilities = candidate.predict_proba(X_valid)[:, positive_class_index]
        valid_pr_auc = average_precision_score(y_valid, valid_probabilities)
        search_line = (
            f"valid PR-AUC | iterations={candidate_parameters['iterations']}, "
            f"depth={candidate_parameters['depth']}, "
            f"learning_rate={candidate_parameters['learning_rate']}, "
            f"l2_leaf_reg={candidate_parameters['l2_leaf_reg']}: {valid_pr_auc:.4f} "
            f"(best_iteration={candidate.get_best_iteration()})"
        )
        search_results.append(search_line)
        print(search_line)

        if valid_pr_auc > best_valid_pr_auc:
            best_valid_pr_auc = float(valid_pr_auc)
            best_model = candidate
            best_parameters = candidate_parameters

    if best_model is None or best_parameters is None:
        raise RuntimeError("CatBoost 설정을 선택하지 못했습니다.")

    # valid에서 F1을 가장 높이는 임계값을 고르고, test에는 그대로 적용합니다.
    positive_class_index = list(best_model.classes_).index(1)
    best_valid_probabilities = best_model.predict_proba(X_valid)[:, positive_class_index]
    threshold, valid_best_f1 = choose_f1_threshold(y_valid, best_valid_probabilities)

    # 모델이 중요하게 사용한 피처 상위 10개를 결과에 남깁니다.
    feature_importances = best_model.get_feature_importance()
    top_features = sorted(
        zip(FEATURE_COLUMNS, feature_importances),
        key=lambda item: item[1],
        reverse=True,
    )[:10]

    # 데이터 출처, 선택 설정, 임계값 및 모델 선택 과정 요약을 작성합니다.
    summary = [
        f"데이터 파일: {DATA_PATH}",
        f"분할 크기: train={len(train_df):,}, valid={len(valid_df):,}, test={len(test_df):,}",
        f"학습 이탈 비율: {y_train.mean():.2%}",
        f"선택된 설정: {best_parameters}",
        f"선택 모델의 valid PR-AUC: {best_valid_pr_auc:.4f}",
        f"valid에서 선택한 임계값: {threshold:.6f} (F1={valid_best_f1:.4f})",
        "중요 피처 상위 10개 (CatBoost feature importance):",
    ]
    summary.extend(f"  {name}: {importance:.4f}" for name, importance in top_features)

    # valid 결과와 최종 test 결과를 각각 계산하고 텍스트 파일에 저장합니다.
    validation_report = evaluate_split("검증(valid)", best_model, valid_df, threshold)
    test_report = evaluate_split("최종 평가(test)", best_model, test_df, threshold)
    full_report = "\n".join(summary + ["하이퍼파라미터 비교:"] + search_results + [validation_report, test_report])
    print("\n" + full_report)
    RESULTS_PATH.write_text(full_report + "\n", encoding="utf-8")
    COLLECTED_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    COLLECTED_RESULTS_PATH.write_text(full_report + "\n", encoding="utf-8")

    # 모델과 재사용에 필요한 임계값, 설정, 피처 순서를 함께 저장합니다.
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
