"""원본 고객 분할의 기존 CatBoost, XGBoost, LightGBM 예측을 결합합니다.

valid에서 제한된 조합 중 PR-AUC가 가장 높은 가중치를 고릅니다.
test는 마지막 성능 평가에만 사용합니다.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
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


SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR.parents[2]
DATA_PATH = DATASET_DIR / "data" / "processed" / "customer_features_20181101_v1_with_split.csv"
MODEL_PATHS = {
    "CatBoost": DATASET_DIR / "machine_learning" / "catboost" / "catboost_v1.joblib",
    "XGBoost": DATASET_DIR / "machine_learning" / "xgboost" / "xgboost_v1.joblib",
    "LightGBM": DATASET_DIR / "machine_learning" / "lightgbm" / "lightgbm_v1.joblib",
}
RESULT_PATH = SCRIPT_DIR / "ensemble_v1_evaluation_results.txt"
COLLECTED_RESULT_PATH = DATASET_DIR / "result" / RESULT_PATH.name
ENSEMBLE_PATH = SCRIPT_DIR / "ensemble_v1.joblib"

TARGET = "churn"
SPLIT_COLUMN = "split"
RECALL_TARGETS = [0.60, 0.70]

# 조합 수를 제한해 valid에 지나치게 맞추지 않도록 합니다.
WEIGHT_CANDIDATES = {
    "CatBoost only": {"CatBoost": 1.0},
    "XGBoost only": {"XGBoost": 1.0},
    "LightGBM only": {"LightGBM": 1.0},
    "CatBoost + XGBoost (50/50)": {"CatBoost": 0.5, "XGBoost": 0.5},
    "CatBoost + LightGBM (50/50)": {"CatBoost": 0.5, "LightGBM": 0.5},
    "XGBoost + LightGBM (50/50)": {"XGBoost": 0.5, "LightGBM": 0.5},
    "All models (equal weights)": {"CatBoost": 1 / 3, "XGBoost": 1 / 3, "LightGBM": 1 / 3},
}


def load_artifacts() -> dict:
    """기존 모델 파일을 읽고 필수 구조를 점검합니다."""
    artifacts = {}
    for name, path in MODEL_PATHS.items():
        if not path.exists():
            raise FileNotFoundError(f"기존 {name} 모델 파일이 없습니다: {path}")
        artifact = joblib.load(path)
        if not isinstance(artifact, dict) or "model" not in artifact:
            raise ValueError(f"{name} 모델 파일에 재사용 가능한 model 항목이 없습니다.")
        if "feature_columns" not in artifact:
            raise ValueError(f"{name} 모델 파일에 feature_columns가 없습니다.")
        artifacts[name] = artifact
    return artifacts


def model_probability(artifact: dict, frame: pd.DataFrame) -> np.ndarray:
    """각 기존 모델에 저장된 피처 순서대로 이탈 확률을 구합니다."""
    columns = artifact["feature_columns"]
    model = artifact["model"]
    positive_index = list(model.classes_).index(1)
    return model.predict_proba(frame[columns])[:, positive_index]


def choose_thresholds(y_valid: pd.Series, probabilities: np.ndarray) -> dict:
    """valid에서 F1 최대 및 재현율 목표를 만족하는 임계값을 고릅니다."""
    thresholds = np.unique(np.concatenate(([0.0, 0.5, 1.0], probabilities)))
    rows = []
    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        rows.append({
            "threshold": float(threshold),
            "precision": float(precision_score(y_valid, predictions, zero_division=0)),
            "recall": float(recall_score(y_valid, predictions, zero_division=0)),
            "f1": float(f1_score(y_valid, predictions, zero_division=0)),
        })

    choices = {"F1 최대": max(rows, key=lambda row: row["f1"])}
    for target in RECALL_TARGETS:
        possible = [row for row in rows if row["recall"] >= target]
        if possible:
            choices[f"재현율 {target:.0%} 이상"] = max(
                possible, key=lambda row: (row["precision"], row["threshold"])
            )
    return choices


def evaluate(name: str, y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> str:
    """test 지표와 혼동 행렬을 결과 문장으로 만듭니다."""
    predictions = (probabilities >= threshold).astype(int)
    return "\n".join([
        f"===== {name} (임계값={threshold:.6f}, 표본={len(y_true):,}) =====",
        f"Accuracy:  {accuracy_score(y_true, predictions):.4f}",
        f"Precision: {precision_score(y_true, predictions, zero_division=0):.4f}",
        f"Recall:    {recall_score(y_true, predictions, zero_division=0):.4f}",
        f"F1-score:  {f1_score(y_true, predictions, zero_division=0):.4f}",
        f"ROC-AUC:   {roc_auc_score(y_true, probabilities):.4f}",
        f"PR-AUC:    {average_precision_score(y_true, probabilities):.4f}",
        "분류 리포트:",
        classification_report(y_true, predictions, zero_division=0),
        "혼동 행렬 (행=실제, 열=예측):",
        str(confusion_matrix(y_true, predictions)),
    ])


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"원본 공통 분할 파일이 없습니다: {DATA_PATH}")
    data = pd.read_csv(DATA_PATH, dtype={"user_id": "string"})
    train = data[data[SPLIT_COLUMN] == "train"].copy()
    valid = data[data[SPLIT_COLUMN] == "valid"].copy()
    test = data[data[SPLIT_COLUMN] == "test"].copy()
    if train.empty or valid.empty or test.empty:
        raise ValueError("원본 데이터에 train, valid, test 분할이 모두 있어야 합니다.")

    artifacts = load_artifacts()
    probability_sets = {name: {} for name in ("train", "valid", "test")}
    split_frames = {"train": train, "valid": valid, "test": test}
    for model_name, artifact in artifacts.items():
        absent = [column for column in artifact["feature_columns"] if column not in data.columns]
        if absent:
            raise ValueError(f"원본 CSV에 {model_name} 입력 피처가 없습니다: {absent}")
        for split_name, frame in split_frames.items():
            probability_sets[split_name][model_name] = model_probability(artifact, frame)

    # 가중치는 valid PR-AUC 기준으로 제한된 조합 중 하나를 고릅니다.
    candidate_lines = []
    candidate_scores = {}
    for name, weights in WEIGHT_CANDIDATES.items():
        probabilities = sum(
            weight * probability_sets["valid"][model_name]
            for model_name, weight in weights.items()
        )
        score = float(average_precision_score(valid[TARGET], probabilities))
        candidate_scores[name] = score
        candidate_lines.append(f"{name}: valid PR-AUC={score:.4f}, weights={weights}")
    winner_name = max(candidate_scores, key=candidate_scores.get)
    winner_weights = WEIGHT_CANDIDATES[winner_name]

    valid_probabilities = sum(
        weight * probability_sets["valid"][model_name]
        for model_name, weight in winner_weights.items()
    )
    test_probabilities = sum(
        weight * probability_sets["test"][model_name]
        for model_name, weight in winner_weights.items()
    )
    threshold_choices = choose_thresholds(valid[TARGET], valid_probabilities)

    lines = [
        "원본 70/15/15 분할의 기존 모델 확률 앙상블",
        f"데이터: {DATA_PATH}",
        f"분할 크기: train={len(train):,}, valid={len(valid):,}, test={len(test):,}",
        f"학습 이탈 비율: {train[TARGET].mean():.2%}",
        "기존 CatBoost, XGBoost, LightGBM 모델 파일을 사용했습니다.",
        f"valid PR-AUC로 선택한 조합: {winner_name}",
        f"선택 조합 valid PR-AUC: {candidate_scores[winner_name]:.4f}",
        f"가중치: {winner_weights}",
        "\n===== 앙상블 조합 비교 (valid) =====",
        *candidate_lines,
        "\n===== 기존 단일 모델 test 기준 성능 =====",
    ]
    for model_name, artifact in artifacts.items():
        saved_threshold = float(artifact.get("decision_threshold", 0.5))
        lines.append(
            evaluate(
                f"기존 {model_name}",
                test[TARGET],
                probability_sets["test"][model_name],
                saved_threshold,
            )
        )
    lines.extend([
        "\n===== valid에서 선택한 임계값 =====",
    ])
    for name, choice in threshold_choices.items():
        lines.append(
            f"{name}: threshold={choice['threshold']:.6f}, "
            f"precision={choice['precision']:.4f}, recall={choice['recall']:.4f}, f1={choice['f1']:.4f}"
        )
    lines.append("\n===== 최종 test 평가 =====")
    for name, choice in threshold_choices.items():
        lines.append(evaluate(name, test[TARGET], test_probabilities, choice["threshold"]))

    report = "\n".join(lines)
    print(report)
    RESULT_PATH.write_text(report + "\n", encoding="utf-8")
    COLLECTED_RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    COLLECTED_RESULT_PATH.write_text(report + "\n", encoding="utf-8")
    joblib.dump(
        {
            "models": {name: artifacts[name]["model"] for name in winner_weights},
            "weights": winner_weights,
            "threshold_choices": threshold_choices,
            "feature_columns": {
                name: artifacts[name]["feature_columns"] for name in winner_weights
            },
        },
        ENSEMBLE_PATH,
    )
    print(f"\n결과 저장: {RESULT_PATH}")
    print(f"결과 모음 폴더 저장: {COLLECTED_RESULT_PATH}")
    print(f"앙상블 저장: {ENSEMBLE_PATH}")


if __name__ == "__main__":
    main()
