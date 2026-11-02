"""교차 검증으로 CatBoost를 고르고 검증 데이터에서 업무별 임계값을 비교합니다.

원본 데이터 파일은 읽기만 합니다. 교차 검증과 설정 선택에는 train만,
임계값 선택에는 valid만, 최종 성능 보고에는 test만 사용합니다.
"""

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
from sklearn.model_selection import StratifiedKFold


SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR.parents[2]
DATA_PATH = DATASET_DIR / "data" / "processed" / "customer_features_20181101_v1_with_split.csv"
RESULT_PATH = SCRIPT_DIR / "catboost_cross_validation_threshold_v3_evaluation_results.txt"
COLLECTED_RESULT_PATH = DATASET_DIR / "result" / RESULT_PATH.name
MODEL_PATH = SCRIPT_DIR / "catboost_cross_validation_threshold_v3.joblib"

TARGET = "churn"
SPLIT_COLUMN = "split"
CATEGORICAL_FEATURES = ["level_last"]
EXCLUDED_COLUMNS = {"user_id", TARGET, SPLIT_COLUMN}
RANDOM_STATE = 42
FOLDS = 5

# 탐색 범위는 계산량을 고려해 기존 실험에서 괜찮았던 구간 중심으로 구성했습니다.
PARAMETER_CANDIDATES = [
    {"iterations": 800, "depth": 3, "learning_rate": 0.04, "l2_leaf_reg": 3, "random_strength": 0.5},
    {"iterations": 1000, "depth": 4, "learning_rate": 0.03, "l2_leaf_reg": 5, "random_strength": 1.0},
    {"iterations": 1000, "depth": 5, "learning_rate": 0.03, "l2_leaf_reg": 8, "random_strength": 1.0},
    {"iterations": 800, "depth": 6, "learning_rate": 0.02, "l2_leaf_reg": 10, "random_strength": 1.5},
]
CLASS_WEIGHT_OPTIONS = [None, "SqrtBalanced"]
RECALL_TARGETS = [0.50, 0.60, 0.70]
TOP_RISK_RATES = [0.05, 0.10]


def make_model(parameters: dict, class_weight: str | None, seed: int) -> CatBoostClassifier:
    """공통 CatBoost 설정을 적용해 모델을 만듭니다."""
    return CatBoostClassifier(
        **parameters,
        loss_function="Logloss",
        eval_metric="PRAUC:type=Classic",
        auto_class_weights=class_weight,
        random_seed=seed,
        allow_writing_files=False,
        thread_count=-1,
        verbose=False,
    )


def get_positive_probabilities(model: CatBoostClassifier, frame: pd.DataFrame) -> np.ndarray:
    """클래스 라벨 순서에 의존하지 않고 churn=1 확률을 가져옵니다."""
    positive_index = list(model.classes_).index(1)
    return model.predict_proba(frame)[:, positive_index]


def select_thresholds(y_valid: pd.Series, probabilities: np.ndarray) -> dict:
    """검증 데이터 점수만 이용해 F1, 재현율 목표, 상위 위험군 기준을 정합니다."""
    thresholds = np.unique(np.concatenate(([0.0, 0.5, 1.0], probabilities)))
    candidates = []
    for threshold in thresholds:
        prediction = (probabilities >= threshold).astype(int)
        candidates.append({
            "threshold": float(threshold),
            "precision": float(precision_score(y_valid, prediction, zero_division=0)),
            "recall": float(recall_score(y_valid, prediction, zero_division=0)),
            "f1": float(f1_score(y_valid, prediction, zero_division=0)),
        })

    choices = {"F1 최대": max(candidates, key=lambda item: item["f1"])}
    for target_recall in RECALL_TARGETS:
        eligible = [item for item in candidates if item["recall"] >= target_recall]
        if eligible:
            # 목표 재현율을 만족하는 중 정밀도가 높은 임계값을 선택합니다.
            choices[f"재현율 {target_recall:.0%} 이상"] = max(
                eligible, key=lambda item: (item["precision"], item["threshold"])
            )

    # 상위 위험군 기준은 예산/연락 가능 인원에 맞춰 선택할 수 있도록 두 시나리오를 기록합니다.
    for rate in TOP_RISK_RATES:
        threshold = float(np.quantile(probabilities, 1.0 - rate, method="higher"))
        prediction = (probabilities >= threshold).astype(int)
        choices[f"상위 위험 {rate:.0%}"] = {
            "threshold": threshold,
            "precision": float(precision_score(y_valid, prediction, zero_division=0)),
            "recall": float(recall_score(y_valid, prediction, zero_division=0)),
            "f1": float(f1_score(y_valid, prediction, zero_division=0)),
        }
    return choices


def evaluate(name: str, y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> str:
    """선택한 임계값에서 성능과 혼동 행렬을 문자열로 만듭니다."""
    prediction = (probabilities >= threshold).astype(int)
    return "\n".join([
        f"===== {name} (임계값={threshold:.6f}, 표본={len(y_true):,}) =====",
        f"Accuracy:  {accuracy_score(y_true, prediction):.4f}",
        f"Precision: {precision_score(y_true, prediction, zero_division=0):.4f}",
        f"Recall:    {recall_score(y_true, prediction, zero_division=0):.4f}",
        f"F1-score:  {f1_score(y_true, prediction, zero_division=0):.4f}",
        f"ROC-AUC:   {roc_auc_score(y_true, probabilities):.4f}",
        f"PR-AUC:    {average_precision_score(y_true, probabilities):.4f}",
        "분류 리포트:",
        classification_report(y_true, prediction, zero_division=0),
        "혼동 행렬 (행=실제, 열=예측):",
        str(confusion_matrix(y_true, prediction)),
    ])


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"입력 데이터 파일이 없습니다: {DATA_PATH}")
    data = pd.read_csv(DATA_PATH, dtype={"user_id": "string"})
    train = data[data[SPLIT_COLUMN] == "train"].copy()
    valid = data[data[SPLIT_COLUMN] == "valid"].copy()
    test = data[data[SPLIT_COLUMN] == "test"].copy()
    if train.empty or valid.empty or test.empty:
        raise ValueError("데이터에 train, valid, test 분할이 모두 있는지 확인해 주세요.")

    feature_columns = [column for column in data.columns if column not in EXCLUDED_COLUMNS]
    missing_categories = [column for column in CATEGORICAL_FEATURES if column not in feature_columns]
    if missing_categories:
        raise ValueError(f"범주형 피처를 찾을 수 없습니다: {missing_categories}")

    # StratifiedKFold는 train만 5개로 나누며 각 fold의 이탈 비율을 비슷하게 유지합니다.
    splitter = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_rows = []
    for weight in CLASS_WEIGHT_OPTIONS:
        for parameters in PARAMETER_CANDIDATES:
            fold_scores = []
            fold_iterations = []
            for fold_number, (fit_indices, holdout_indices) in enumerate(
                splitter.split(train[feature_columns], train[TARGET]), start=1
            ):
                fold_train = train.iloc[fit_indices]
                fold_holdout = train.iloc[holdout_indices]
                model = make_model(parameters, weight, RANDOM_STATE + fold_number)
                model.fit(
                    fold_train[feature_columns],
                    fold_train[TARGET],
                    cat_features=CATEGORICAL_FEATURES,
                    eval_set=(fold_holdout[feature_columns], fold_holdout[TARGET]),
                    early_stopping_rounds=100,
                    use_best_model=True,
                    verbose=False,
                )
                probabilities = get_positive_probabilities(model, fold_holdout[feature_columns])
                fold_scores.append(average_precision_score(fold_holdout[TARGET], probabilities))
                fold_iterations.append(max(1, model.get_best_iteration()))

            row = {
                "parameters": parameters,
                "class_weight": weight,
                "mean_pr_auc": float(np.mean(fold_scores)),
                "std_pr_auc": float(np.std(fold_scores)),
                "iterations": max(1, int(round(np.mean(fold_iterations)))),
                "fold_scores": fold_scores,
            }
            cv_rows.append(row)
            print(
                f"CV weight={weight}, params={parameters}, "
                f"mean PR-AUC={row['mean_pr_auc']:.4f} +/- {row['std_pr_auc']:.4f}"
            )

    winner = max(cv_rows, key=lambda row: row["mean_pr_auc"])
    final_parameters = dict(winner["parameters"])
    final_parameters["iterations"] = winner["iterations"]
    selected_model = make_model(final_parameters, winner["class_weight"], RANDOM_STATE)

    # train 전체로 한 번 학습하고 valid로 임계값만 정합니다. test는 이 과정에 포함하지 않습니다.
    selected_model.fit(
        train[feature_columns],
        train[TARGET],
        cat_features=CATEGORICAL_FEATURES,
        verbose=False,
    )
    valid_probabilities = get_positive_probabilities(selected_model, valid[feature_columns])
    threshold_choices = select_thresholds(valid[TARGET], valid_probabilities)
    test_probabilities = get_positive_probabilities(selected_model, test[feature_columns])

    lines = [
        "CatBoost 교차 검증 및 임계값 시나리오 비교",
        f"입력 데이터: {DATA_PATH}",
        f"표본 수: train={len(train):,}, valid={len(valid):,}, test={len(test):,}",
        f"학습 이탈 비율: {train[TARGET].mean():.2%}",
        f"사용 피처 수: {len(feature_columns)}",
        f"교차 검증: train에서 Stratified {FOLDS}-fold, 선택 기준은 평균 PR-AUC",
        f"선택 설정: {winner['parameters']}, class_weight={winner['class_weight']}, "
        f"최종 iterations={winner['iterations']}",
        f"선택 설정 CV PR-AUC: {winner['mean_pr_auc']:.4f} +/- {winner['std_pr_auc']:.4f}",
        "\n===== 전체 교차 검증 탐색 =====",
    ]
    for row in sorted(cv_rows, key=lambda item: item["mean_pr_auc"], reverse=True):
        lines.append(
            f"mean PR-AUC={row['mean_pr_auc']:.4f} +/- {row['std_pr_auc']:.4f}; "
            f"weight={row['class_weight']}; iterations={row['iterations']}; "
            f"params={row['parameters']}; folds={[round(score, 4) for score in row['fold_scores']]}"
        )

    lines.append("\n===== valid에서 선택한 임계값 시나리오 =====")
    for name, choice in threshold_choices.items():
        lines.append(
            f"{name}: threshold={choice['threshold']:.6f}, "
            f"precision={choice['precision']:.4f}, recall={choice['recall']:.4f}, f1={choice['f1']:.4f}"
        )
    lines.append("\n===== test 최종 성능 (각 임계값은 valid에서만 선택) =====")
    for name, choice in threshold_choices.items():
        lines.append(evaluate(name, test[TARGET], test_probabilities, choice["threshold"]))

    report = "\n".join(lines)
    print(report)
    COLLECTED_RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(report + "\n", encoding="utf-8")
    COLLECTED_RESULT_PATH.write_text(report + "\n", encoding="utf-8")
    joblib.dump(
        {
            "model": selected_model,
            "feature_columns": feature_columns,
            "categorical_features": CATEGORICAL_FEATURES,
            "parameters": final_parameters,
            "auto_class_weights": winner["class_weight"],
            "cv_mean_pr_auc": winner["mean_pr_auc"],
            "threshold_choices": threshold_choices,
        },
        MODEL_PATH,
    )
    print(f"\n결과 저장: {RESULT_PATH}")
    print(f"결과 모음 폴더 저장: {COLLECTED_RESULT_PATH}")
    print(f"모델 저장: {MODEL_PATH}")


if __name__ == "__main__":
    main()
