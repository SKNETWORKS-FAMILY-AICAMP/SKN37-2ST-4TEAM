"""원본 분할에서 재현율 우선 기준으로 CatBoost를 탐색합니다.

원본 CSV는 읽기만 합니다. 모델 및 임계값 선택에는 valid를 사용하고,
test는 모든 선택이 끝난 뒤 최종 비교에만 사용합니다.
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
    fbeta_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR.parents[2]
DATA_PATH = DATASET_DIR / "data" / "processed" / "customer_features_20181101_v1_with_split.csv"
RESULT_PATH = SCRIPT_DIR / "catboost_recall_optimization_v4_evaluation_results.txt"
COLLECTED_RESULT_PATH = DATASET_DIR / "result" / RESULT_PATH.name
MODEL_PATH = SCRIPT_DIR / "catboost_recall_optimization_v4.joblib"

TARGET = "churn"
SPLIT_COLUMN = "split"
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
BASE_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
ENGINEERED_NUMERIC_FEATURES = [
    "songs_per_active_day",
    "songs_per_session",
    "active_days_ratio_28d",
    "days_since_last_activity_ratio_28d",
    "recent_song_share",
    "previous_song_share",
    "weekly_song_momentum",
    "feedback_count",
    "thumbs_down_share",
    "ads_per_song",
    "errors_per_session",
    "help_per_session",
]
ENGINEERED_FEATURES = BASE_FEATURES + ENGINEERED_NUMERIC_FEATURES
RANDOM_STATE = 42

# 지나치게 깊은 트리 대신 작은 데이터에서도 과적합을 줄일 설정을 함께 비교합니다.
PARAMETER_CANDIDATES = [
    # 기존 CatBoost 결과를 같은 데이터에서 재현하기 위한 기준 설정입니다.
    {"iterations": 500, "depth": 4, "learning_rate": 0.05, "l2_leaf_reg": 3},
    {"iterations": 800, "depth": 3, "learning_rate": 0.04, "l2_leaf_reg": 5, "random_strength": 0.5},
    {"iterations": 800, "depth": 4, "learning_rate": 0.05, "l2_leaf_reg": 5, "random_strength": 1.0},
    {"iterations": 1000, "depth": 5, "learning_rate": 0.03, "l2_leaf_reg": 8, "random_strength": 1.0},
    {"iterations": 800, "depth": 6, "learning_rate": 0.03, "l2_leaf_reg": 10, "random_strength": 1.5},
]
CLASS_WEIGHT_OPTIONS = [None, "SqrtBalanced", "Balanced"]
RECALL_TARGETS = [0.70, 0.75, 0.80, 0.85, 0.90]


def add_derived_features(frame: pd.DataFrame) -> pd.DataFrame:
    """기존 고객 집계값만 이용해 활동률과 행동 비율 피처를 계산합니다."""
    result = frame.copy()

    def safe_ratio(numerator: str, denominator: str) -> pd.Series:
        # 분모가 0인 경우 해당 행동 비율을 0으로 두어 무한대가 생기지 않게 합니다.
        return (result[numerator] / result[denominator].replace(0, np.nan)).fillna(0.0)

    result["songs_per_active_day"] = safe_ratio("song_count", "active_days")
    result["songs_per_session"] = safe_ratio("song_count", "session_count")
    result["active_days_ratio_28d"] = result["active_days"] / 28.0
    result["days_since_last_activity_ratio_28d"] = result["days_since_last_activity"] / 28.0
    result["recent_song_share"] = safe_ratio("recent_7d_song_count", "song_count")
    result["previous_song_share"] = safe_ratio("previous_7d_song_count", "song_count")
    recent = result["recent_7d_song_count"]
    previous = result["previous_7d_song_count"]
    result["weekly_song_momentum"] = (recent - previous) / (recent + previous + 1.0)
    result["feedback_count"] = result["thumbs_up_count"] + result["thumbs_down_count"]
    result["thumbs_down_share"] = safe_ratio("thumbs_down_count", "feedback_count")
    result["ads_per_song"] = safe_ratio("ad_count", "song_count")
    result["errors_per_session"] = safe_ratio("error_count", "session_count")
    result["help_per_session"] = safe_ratio("help_count", "session_count")
    return result


def choose_thresholds(y_valid: pd.Series, probabilities: np.ndarray) -> dict:
    """검증 데이터에서 F1/F2 최대와 여러 재현율 목표별 임계값을 계산합니다."""
    thresholds = np.unique(np.concatenate(([0.0, 0.5, 1.0], probabilities)))
    rows = []
    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        rows.append({
            "threshold": float(threshold),
            "precision": float(precision_score(y_valid, predictions, zero_division=0)),
            "recall": float(recall_score(y_valid, predictions, zero_division=0)),
            "f1": float(f1_score(y_valid, predictions, zero_division=0)),
            "f2": float(fbeta_score(y_valid, predictions, beta=2, zero_division=0)),
        })
    choices = {"F1 최대": max(rows, key=lambda row: row["f1"])}
    choices["F2 최대 (재현율 우선)"] = max(rows, key=lambda row: row["f2"])
    for target in RECALL_TARGETS:
        eligible = [row for row in rows if row["recall"] >= target]
        if eligible:
            choices[f"재현율 {target:.0%} 이상"] = max(
                eligible, key=lambda row: (row["precision"], row["threshold"])
            )
    return choices


def evaluate(name: str, y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> tuple[str, dict]:
    """한 분할의 지표와 혼동 행렬을 계산합니다."""
    predictions = (probabilities >= threshold).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
    }
    report = "\n".join([
        f"===== {name} (임계값={threshold:.6f}, 표본={len(y_true):,}) =====",
        f"Accuracy:  {metrics['accuracy']:.4f}",
        f"Precision: {metrics['precision']:.4f}",
        f"Recall:    {metrics['recall']:.4f}",
        f"F1-score:  {metrics['f1']:.4f}",
        f"ROC-AUC:   {metrics['roc_auc']:.4f}",
        f"PR-AUC:    {metrics['pr_auc']:.4f}",
        "분류 리포트:",
        classification_report(y_true, predictions, zero_division=0),
        "혼동 행렬 (행=실제, 열=예측):",
        str(confusion_matrix(y_true, predictions)),
    ])
    return report, metrics


def fit_best_variant(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    feature_columns: list[str],
    variant_name: str,
) -> tuple[dict, list[str]]:
    """피처 버전별로 설정을 비교하고 valid F2 최대 기준으로 선택합니다."""
    best = None
    search_lines = []
    for class_weight in CLASS_WEIGHT_OPTIONS:
        for parameters in PARAMETER_CANDIDATES:
            model = CatBoostClassifier(
                **parameters,
                loss_function="Logloss",
                eval_metric="PRAUC:type=Classic",
                auto_class_weights=class_weight,
                random_seed=RANDOM_STATE,
                allow_writing_files=False,
                thread_count=-1,
                verbose=False,
            )
            model.fit(
                train[feature_columns],
                train[TARGET],
                cat_features=CATEGORICAL_FEATURES,
                eval_set=(valid[feature_columns], valid[TARGET]),
                early_stopping_rounds=100,
                use_best_model=True,
                verbose=False,
            )
            positive_index = list(model.classes_).index(1)
            probabilities = model.predict_proba(valid[feature_columns])[:, positive_index]
            pr_auc = float(average_precision_score(valid[TARGET], probabilities))
            threshold_choices = choose_thresholds(valid[TARGET], probabilities)
            f2_score_value = threshold_choices["F2 최대 (재현율 우선)"]["f2"]
            line = (
                f"{variant_name} | weight={class_weight}, params={parameters}: "
                f"valid PR-AUC={pr_auc:.4f}, valid F2={f2_score_value:.4f}, "
                f"best_iteration={model.get_best_iteration()}"
            )
            search_lines.append(line)
            print(line)
            if best is None or f2_score_value > best["valid_f2"]:
                best = {
                    "model": model,
                    "parameters": parameters,
                    "class_weight": class_weight,
                    "valid_pr_auc": pr_auc,
                    "valid_f2": f2_score_value,
                    "thresholds": threshold_choices,
                    "features": feature_columns,
                    "variant": variant_name,
                }
    if best is None:
        raise RuntimeError(f"{variant_name} 모델을 선택하지 못했습니다.")
    positive_index = list(best["model"].classes_).index(1)
    best["valid_probabilities"] = best["model"].predict_proba(valid[feature_columns])[:, positive_index]
    return best, search_lines


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"원본 데이터 파일을 찾을 수 없습니다: {DATA_PATH}")
    data = pd.read_csv(DATA_PATH, dtype={"user_id": "string"})
    required = BASE_FEATURES + [TARGET, SPLIT_COLUMN]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"원본 CSV에서 필요한 열이 없습니다: {missing}")

    base_data = data.copy()
    engineered_data = add_derived_features(data)
    train_base = base_data[base_data[SPLIT_COLUMN] == "train"].copy()
    valid_base = base_data[base_data[SPLIT_COLUMN] == "valid"].copy()
    test_base = base_data[base_data[SPLIT_COLUMN] == "test"].copy()
    train_engineered = engineered_data[engineered_data[SPLIT_COLUMN] == "train"].copy()
    valid_engineered = engineered_data[engineered_data[SPLIT_COLUMN] == "valid"].copy()
    test_engineered = engineered_data[engineered_data[SPLIT_COLUMN] == "test"].copy()
    if any(frame.empty for frame in (train_base, valid_base, test_base)):
        raise ValueError("원본 분할에 train, valid, test가 모두 있어야 합니다.")

    variants = [
        ("기본 피처", train_base, valid_base, test_base, BASE_FEATURES),
        ("파생 피처 추가", train_engineered, valid_engineered, test_engineered, ENGINEERED_FEATURES),
    ]
    best_by_variant = {}
    search_lines = []
    for name, train, valid, test, feature_columns in variants:
        best, candidate_lines = fit_best_variant(train, valid, feature_columns, name)
        positive_index = list(best["model"].classes_).index(1)
        best["test_probabilities"] = best["model"].predict_proba(test[feature_columns])[:, positive_index]
        best["test_labels"] = test[TARGET]
        best_by_variant[name] = best
        search_lines.extend(candidate_lines)

    # valid F2가 높은 변형을 저장 대상으로 정합니다. test 점수로 선택하지 않습니다.
    winner_name = max(best_by_variant, key=lambda name: best_by_variant[name]["valid_f2"])
    winner = best_by_variant[winner_name]
    importances = sorted(
        zip(winner["features"], winner["model"].get_feature_importance()),
        key=lambda item: item[1],
        reverse=True,
    )[:15]

    lines = [
        "원본 70/15/15 데이터 CatBoost 피처·가중치 개선 실험",
        f"입력 데이터: {DATA_PATH}",
        f"분할 크기: train={len(train_base):,}, valid={len(valid_base):,}, test={len(test_base):,}",
        f"학습 이탈 비율: {train_base[TARGET].mean():.2%}",
        "원본 CSV는 수정하지 않았습니다. 파생 피처는 메모리에서 계산했습니다.",
        f"저장 모델 선택 기준: valid F2 최대; 선택 변형={winner_name}",
        f"선택 설정: {winner['parameters']}, class_weight={winner['class_weight']}",
        f"선택 모델 valid PR-AUC={winner['valid_pr_auc']:.4f}",
        f"선택 모델 valid F2={winner['valid_f2']:.4f}, F2 임계값={winner['thresholds']['F2 최대 (재현율 우선)']['threshold']:.6f}",
        "선택 모델의 상위 피처 중요도:",
        *[f"  {name}: {importance:.4f}" for name, importance in importances],
        "\n===== 하이퍼파라미터·가중치 탐색 =====",
        *search_lines,
        "\n===== valid 임계값 선택 =====",
    ]
    for name, result in best_by_variant.items():
        lines.append(f"--- {name} | valid PR-AUC={result['valid_pr_auc']:.4f}, valid F2={result['valid_f2']:.4f} ---")
        for threshold_name, choice in result["thresholds"].items():
            lines.append(
                f"{threshold_name}: threshold={choice['threshold']:.6f}, "
                f"precision={choice['precision']:.4f}, recall={choice['recall']:.4f}, f1={choice['f1']:.4f}"
            )

    lines.append("\n===== test 최종 비교 (임계값은 valid에서 선택) =====")
    for name, result in best_by_variant.items():
        for threshold_name, choice in result["thresholds"].items():
            report, _ = evaluate(
                f"{name} | {threshold_name}",
                result["test_labels"],
                result["test_probabilities"],
                choice["threshold"],
            )
            lines.append(report)

    report_text = "\n".join(lines)
    print(report_text)
    COLLECTED_RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(report_text + "\n", encoding="utf-8")
    COLLECTED_RESULT_PATH.write_text(report_text + "\n", encoding="utf-8")
    joblib.dump(
        {
            "model": winner["model"],
            "variant": winner["variant"],
            "decision_thresholds": winner["thresholds"],
            "parameters": winner["parameters"],
            "auto_class_weights": winner["class_weight"],
            "feature_columns": winner["features"],
            "categorical_features": CATEGORICAL_FEATURES,
        },
        MODEL_PATH,
    )
    print(f"\n결과 저장: {RESULT_PATH}")
    print(f"result 폴더 사본: {COLLECTED_RESULT_PATH}")
    print(f"valid 기준 선택 모델: {MODEL_PATH}")


if __name__ == "__main__":
    main()
