"""원본 데이터는 유지하고 CatBoost 설정과 선택 시퀀스 피처를 비교합니다."""

from collections import Counter
from pathlib import Path
import json

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


# 이 실험의 파일은 전용 폴더에 두고, 입력 데이터 파일은 읽기만 합니다.
SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR.parents[2]
DATA_DIR = DATASET_DIR / "data" / "processed"
FEATURES_PATH = DATA_DIR / "customer_features_20181101_v1_with_split.csv"
SEQUENCES_PATH = DATA_DIR / "customer_sequences_20181101_v1.jsonl"
RESULTS_PATH = SCRIPT_DIR / "catboost_tuned_features_v2_evaluation_results.txt"
COLLECTED_RESULTS_PATH = DATASET_DIR / "result" / RESULTS_PATH.name
MODEL_PATH = SCRIPT_DIR / "catboost_tuned_features_v2.joblib"

# 기존 정형 CatBoost와 동일한 14개 기본 입력 변수입니다.
BASE_NUMERIC_FEATURES = [
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
BASE_CATEGORICAL_FEATURES = ["level_last"]
BASE_FEATURES = BASE_NUMERIC_FEATURES + BASE_CATEGORICAL_FEATURES
TARGET = "churn"
SPLIT_COLUMN = "split"

# 원본 시퀀스에서 추가할 소수의 변화·순서·시간 피처입니다.
SEQUENCE_NUMERIC_FEATURES = [
    "seq_length",
    "seq_unique_page_count",
    "seq_recent10_nextsong_ratio",
    "seq_previous10_nextsong_ratio",
    "seq_nextsong_ratio_change",
    "seq_recent10_thumbs_down_count",
    "seq_previous10_thumbs_down_count",
    "seq_thumbs_down_change",
    "seq_recent10_roll_advert_count",
    "seq_previous10_roll_advert_count",
    "seq_recent10_downgrade_count",
    "seq_recent10_playlist_add_count",
    "seq_recent10_unique_page_count",
    "seq_recent10_page_change_ratio",
    "seq_days_since_last_event",
    "seq_median_gap_minutes",
    "seq_max_gap_hours",
]
SEQUENCE_CATEGORICAL_FEATURES = ["seq_last_page"]
SEQUENCE_FEATURES = SEQUENCE_NUMERIC_FEATURES + SEQUENCE_CATEGORICAL_FEATURES
ENHANCED_FEATURES = BASE_FEATURES + SEQUENCE_FEATURES
ENHANCED_CATEGORICAL_FEATURES = BASE_CATEGORICAL_FEATURES + SEQUENCE_CATEGORICAL_FEATURES

# 관찰 기간이 끝나는 UTC 기준 시각. 이후 행동이 들어가면 정보 누수로 보고 중단합니다.
CUTOFF_TS_MS = int(pd.Timestamp("2018-11-01T00:00:00Z").timestamp() * 1000)
MILLISECONDS_PER_DAY = 24 * 60 * 60 * 1000
RANDOM_STATE = 42

# 기존의 여섯 설정보다 넓게 탐색하되, 검증 데이터로만 선택합니다.
PARAMETER_CANDIDATES = [
    {"iterations": 1200, "depth": 3, "learning_rate": 0.05, "l2_leaf_reg": 3, "random_strength": 0.5},
    {"iterations": 1600, "depth": 4, "learning_rate": 0.03, "l2_leaf_reg": 5, "random_strength": 1.0},
    {"iterations": 1200, "depth": 5, "learning_rate": 0.03, "l2_leaf_reg": 8, "random_strength": 1.0},
    {"iterations": 1000, "depth": 6, "learning_rate": 0.02, "l2_leaf_reg": 10, "random_strength": 1.5},
]
CLASS_WEIGHT_OPTIONS = ["Balanced", "SqrtBalanced"]


def extract_selected_sequence_features(record: dict) -> dict:
    """최대 50개 행동에서 최근-이전 변화와 시간 간격만 추출합니다."""
    events = record.get("events") or []
    pages = [str(event.get("page", "Unknown")) for event in events]
    timestamps = [int(event["ts"]) for event in events if event.get("ts") is not None]

    # 원본 설명서의 관찰 종료 시각 이후 이벤트는 피처에서 허용하지 않습니다.
    if any(timestamp >= CUTOFF_TS_MS for timestamp in timestamps):
        raise ValueError(f"관찰 종료 이후 이벤트가 있습니다: user_id={record.get('user_id')}")

    recent10 = pages[-10:]
    previous10 = pages[-20:-10]
    recent_counts = Counter(recent10)
    previous_counts = Counter(previous10)
    recent_transitions = list(zip(recent10[:-1], recent10[1:]))

    # 최근 10개와 그 이전 10개에서 곡 재생 비율이 어떻게 변했는지 계산합니다.
    recent_song_ratio = recent_counts["NextSong"] / len(recent10) if recent10 else 0.0
    previous_song_ratio = previous_counts["NextSong"] / len(previous10) if previous10 else 0.0

    row = {
        "user_id": str(record["user_id"]),
        "sequence_split": record.get("split"),
        "sequence_churn": record.get("churn"),
        "seq_length": len(pages),
        "seq_unique_page_count": len(set(pages)),
        "seq_recent10_nextsong_ratio": recent_song_ratio,
        "seq_previous10_nextsong_ratio": previous_song_ratio,
        "seq_nextsong_ratio_change": recent_song_ratio - previous_song_ratio,
        "seq_recent10_thumbs_down_count": recent_counts["Thumbs Down"],
        "seq_previous10_thumbs_down_count": previous_counts["Thumbs Down"],
        "seq_thumbs_down_change": recent_counts["Thumbs Down"] - previous_counts["Thumbs Down"],
        "seq_recent10_roll_advert_count": recent_counts["Roll Advert"],
        "seq_previous10_roll_advert_count": previous_counts["Roll Advert"],
        "seq_recent10_downgrade_count": recent_counts["Downgrade"] + recent_counts["Submit Downgrade"],
        "seq_recent10_playlist_add_count": recent_counts["Add to Playlist"],
        "seq_recent10_unique_page_count": len(set(recent10)),
        "seq_recent10_page_change_ratio": (
            sum(previous != current for previous, current in recent_transitions) / len(recent_transitions)
            if recent_transitions
            else 0.0
        ),
        "seq_last_page": pages[-1] if pages else "NoEvent",
    }

    # 최근 행동 시각의 경과와 이벤트 간격을 요약합니다.
    if timestamps:
        row["seq_days_since_last_event"] = max(
            0.0, (CUTOFF_TS_MS - max(timestamps)) / MILLISECONDS_PER_DAY
        )
        positive_gaps = np.diff(timestamps) / (1000 * 60)
        positive_gaps = positive_gaps[positive_gaps > 0]
        row["seq_median_gap_minutes"] = float(np.median(positive_gaps)) if len(positive_gaps) else 0.0
        row["seq_max_gap_hours"] = float(np.max(positive_gaps) / 60) if len(positive_gaps) else 0.0
    else:
        row["seq_days_since_last_event"] = 28.0
        row["seq_median_gap_minutes"] = 0.0
        row["seq_max_gap_hours"] = 0.0
    return row


def load_data() -> pd.DataFrame:
    """원본 정형 데이터와 선택 시퀀스 피처를 user_id로 결합합니다."""
    if not FEATURES_PATH.exists() or not SEQUENCES_PATH.exists():
        raise FileNotFoundError("입력 CSV 또는 JSONL 파일을 찾을 수 없습니다.")

    # CSV와 JSONL은 읽기만 하고 덮어쓰지 않습니다.
    features = pd.read_csv(FEATURES_PATH, dtype={"user_id": "string"})
    sequence_rows = []
    with SEQUENCES_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                sequence_rows.append(extract_selected_sequence_features(json.loads(line)))
    sequence_df = pd.DataFrame(sequence_rows)
    sequence_df["user_id"] = sequence_df["user_id"].astype("string")

    if sequence_df["user_id"].duplicated().any():
        raise ValueError("시퀀스 JSONL에 중복 user_id가 있습니다.")

    # 행 순서가 아니라 사용자 ID로 결합하고 split/정답이 같은지 확인합니다.
    data = features.merge(sequence_df, on="user_id", how="left", validate="one_to_one")
    if data["seq_length"].isna().any():
        raise ValueError("시퀀스가 없는 고객이 있습니다.")
    if not (data[SPLIT_COLUMN].astype(str) == data["sequence_split"].astype(str)).all():
        raise ValueError("정형 데이터와 시퀀스의 split이 다릅니다.")
    if not (data[TARGET].astype(int) == data["sequence_churn"].astype(int)).all():
        raise ValueError("정형 데이터와 시퀀스의 churn 라벨이 다릅니다.")
    return data.drop(columns=["sequence_split", "sequence_churn"])


def fit_best_variant(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_columns: list[str],
    categorical_features: list[str],
    variant_name: str,
) -> tuple[CatBoostClassifier, dict, str, float, list[str]]:
    """피처 버전별 설정과 클래스 가중치를 valid PR-AUC로 선택합니다."""
    best_model = None
    best_parameters = None
    best_weight = None
    best_pr_auc = -1.0
    search_lines = []
    y_train = train_df[TARGET]
    y_valid = valid_df[TARGET]

    for class_weight in CLASS_WEIGHT_OPTIONS:
        for parameters in PARAMETER_CANDIDATES:
            candidate = CatBoostClassifier(
                **parameters,
                loss_function="Logloss",
                eval_metric="PRAUC:type=Classic",
                auto_class_weights=class_weight,
                random_seed=RANDOM_STATE,
                allow_writing_files=False,
                thread_count=-1,
                verbose=False,
            )
            candidate.fit(
                train_df[feature_columns],
                y_train,
                cat_features=categorical_features,
                eval_set=(valid_df[feature_columns], y_valid),
                early_stopping_rounds=100,
                use_best_model=True,
                verbose=False,
            )
            positive_index = list(candidate.classes_).index(1)
            probabilities = candidate.predict_proba(valid_df[feature_columns])[:, positive_index]
            pr_auc = average_precision_score(y_valid, probabilities)
            line = (
                f"{variant_name} | weight={class_weight}, params={parameters}: "
                f"valid PR-AUC={pr_auc:.4f}, best_iteration={candidate.get_best_iteration()}"
            )
            search_lines.append(line)
            print(line)

            if pr_auc > best_pr_auc:
                best_model = candidate
                best_parameters = parameters
                best_weight = class_weight
                best_pr_auc = float(pr_auc)

    if best_model is None or best_parameters is None or best_weight is None:
        raise RuntimeError(f"{variant_name} 모델 선택에 실패했습니다.")
    return best_model, best_parameters, best_weight, best_pr_auc, search_lines


def choose_f1_threshold(y_true: pd.Series, probabilities: np.ndarray) -> tuple[float, float]:
    """valid에서 F1이 가장 높은 확률 임계값을 찾습니다."""
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
    variant_name: str,
    split_name: str,
    model: CatBoostClassifier,
    split_df: pd.DataFrame,
    feature_columns: list[str],
    threshold: float,
) -> tuple[str, dict]:
    """분할의 주요 성능 지표와 혼동 행렬을 계산합니다."""
    X = split_df[feature_columns]
    y = split_df[TARGET]
    positive_index = list(model.classes_).index(1)
    probabilities = model.predict_proba(X)[:, positive_index]
    predictions = (probabilities >= threshold).astype(int)
    metrics = {
        "accuracy": accuracy_score(y, predictions),
        "precision": precision_score(y, predictions, zero_division=0),
        "recall": recall_score(y, predictions, zero_division=0),
        "f1": f1_score(y, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y, probabilities),
        "pr_auc": average_precision_score(y, probabilities),
    }
    lines = [
        f"===== {variant_name} | {split_name} ({len(split_df):,}명) =====",
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
    return "\n".join(lines), metrics


def main() -> None:
    # 입력은 원본 파일에서만 읽고, 기존 데이터 파일은 변경하지 않습니다.
    data = load_data()
    required_columns = BASE_FEATURES + SEQUENCE_FEATURES + [TARGET, SPLIT_COLUMN]
    missing = [column for column in required_columns if column not in data.columns]
    if missing:
        raise ValueError(f"필요한 피처가 없습니다: {missing}")

    # 공통 고객 단위 분할을 유지합니다.
    train_df = data[data[SPLIT_COLUMN] == "train"].copy()
    valid_df = data[data[SPLIT_COLUMN] == "valid"].copy()
    test_df = data[data[SPLIT_COLUMN] == "test"].copy()
    if train_df.empty or valid_df.empty or test_df.empty:
        raise ValueError("train, valid, test 데이터가 모두 있는지 확인해 주세요.")

    # 기본 14개 피처 모델과 선택 시퀀스 피처 추가 모델을 같은 절차로 비교합니다.
    variants = [
        ("기본 피처", BASE_FEATURES, BASE_CATEGORICAL_FEATURES),
        ("선택 시퀀스 피처", ENHANCED_FEATURES, ENHANCED_CATEGORICAL_FEATURES),
    ]
    experiments = {}
    search_lines = []
    for name, feature_columns, categorical_features in variants:
        model, parameters, class_weight, valid_pr_auc, candidate_lines = fit_best_variant(
            train_df,
            valid_df,
            feature_columns,
            categorical_features,
            name,
        )
        positive_index = list(model.classes_).index(1)
        probabilities = model.predict_proba(valid_df[feature_columns])[:, positive_index]
        threshold, valid_f1 = choose_f1_threshold(valid_df[TARGET], probabilities)
        valid_report, _ = evaluate_split(
            name, "검증(valid)", model, valid_df, feature_columns, threshold
        )
        test_report, test_metrics = evaluate_split(
            name, "최종 평가(test)", model, test_df, feature_columns, threshold
        )
        experiments[name] = {
            "model": model,
            "features": feature_columns,
            "categorical_features": categorical_features,
            "parameters": parameters,
            "class_weight": class_weight,
            "valid_pr_auc": valid_pr_auc,
            "threshold": threshold,
            "valid_f1": valid_f1,
            "valid_report": valid_report,
            "test_report": test_report,
            "test_metrics": test_metrics,
        }
        search_lines.extend(candidate_lines)

    # 최종 저장 모델은 test가 아니라 valid PR-AUC가 더 높은 버전으로 정합니다.
    winner_name = max(experiments, key=lambda name: experiments[name]["valid_pr_auc"])
    winner = experiments[winner_name]
    winner_model = winner["model"]
    importance_values = winner_model.get_feature_importance()
    importance = sorted(
        zip(winner["features"], importance_values),
        key=lambda item: item[1],
        reverse=True,
    )[:15]

    # test 지표 차이를 보여 선택 시퀀스 피처의 실질적인 효과를 확인합니다.
    base_test = experiments["기본 피처"]["test_metrics"]
    enhanced_test = experiments["선택 시퀀스 피처"]["test_metrics"]
    comparison = [
        "===== test 변화: 선택 시퀀스 피처 - 기본 피처 =====",
        f"PR-AUC 변화: {enhanced_test['pr_auc'] - base_test['pr_auc']:+.4f}",
        f"F1 변화:     {enhanced_test['f1'] - base_test['f1']:+.4f}",
        f"Recall 변화: {enhanced_test['recall'] - base_test['recall']:+.4f}",
        f"Precision 변화: {enhanced_test['precision'] - base_test['precision']:+.4f}",
    ]
    summary = [
        f"정형 입력: {FEATURES_PATH}",
        f"시퀀스 입력: {SEQUENCES_PATH}",
        f"분할 크기: train={len(train_df):,}, valid={len(valid_df):,}, test={len(test_df):,}",
        f"학습 이탈 비율: {train_df[TARGET].mean():.2%}",
        f"기본 피처 수: {len(BASE_FEATURES)}; 선택 시퀀스 추가 후: {len(ENHANCED_FEATURES)}",
        "원본 데이터 파일은 수정하지 않았습니다.",
        f"valid PR-AUC 기준 저장 모델: {winner_name}",
    ]
    for name, experiment in experiments.items():
        summary.extend(
            [
                f"{name} 설정: params={experiment['parameters']}, class_weight={experiment['class_weight']}",
                f"{name} valid PR-AUC={experiment['valid_pr_auc']:.4f}, 임계값={experiment['threshold']:.6f}, valid F1={experiment['valid_f1']:.4f}",
            ]
        )
    summary.append("선택된 저장 모델의 상위 피처 중요도:")
    summary.extend(f"  {name}: {score:.4f}" for name, score in importance)

    reports = []
    for name, experiment in experiments.items():
        reports.extend([experiment["valid_report"], experiment["test_report"]])
    full_report = "\n".join(summary + ["하이퍼파라미터/가중치 비교:"] + search_lines + comparison + reports)
    print(full_report)

    # 결과 TXT는 모델 폴더와 중앙 result 폴더에 저장합니다.
    COLLECTED_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(full_report + "\n", encoding="utf-8")
    COLLECTED_RESULTS_PATH.write_text(full_report + "\n", encoding="utf-8")

    # valid 기준으로 선택한 모델과 임계값, 피처 구성을 함께 보관합니다.
    joblib.dump(
        {
            "model": winner_model,
            "model_variant": winner_name,
            "decision_threshold": winner["threshold"],
            "best_parameters": winner["parameters"],
            "auto_class_weights": winner["class_weight"],
            "feature_columns": winner["features"],
            "categorical_features": winner["categorical_features"],
        },
        MODEL_PATH,
    )
    print(f"\n결과 저장: {RESULTS_PATH}")
    print(f"결과 모음 폴더에도 저장: {COLLECTED_RESULTS_PATH}")
    print(f"valid 기준 선택 모델 저장: {MODEL_PATH}")


if __name__ == "__main__":
    main()
