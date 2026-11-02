"""행동 시퀀스에서 추가 피처를 만들어 기존 CatBoost와 비교합니다."""

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re

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


# 이 개선 실험의 코드, 모델, 결과를 각각 전용 폴더에 저장합니다.
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parents[2] / "data" / "processed"
FEATURES_PATH = DATA_DIR / "customer_features_20181101_v1_with_split.csv"
SEQUENCES_PATH = DATA_DIR / "customer_sequences_20181101_v1.jsonl"
RESULTS_PATH = SCRIPT_DIR / "catboost_sequence_features_evaluation_results.txt"
MODEL_PATH = SCRIPT_DIR / "catboost_sequence_features_v1.joblib"
COLLECTED_RESULTS_PATH = SCRIPT_DIR.parents[2] / "result" / "catboost_sequence_features_evaluation_results.txt"

# 정형 데이터에 이미 존재하는 피처입니다.
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

# JSONL에서 확인한 관찰 기간 내 행동 종류입니다.
PAGE_TYPES = [
    "About",
    "Add Friend",
    "Add to Playlist",
    "Downgrade",
    "Error",
    "Help",
    "Home",
    "Logout",
    "NextSong",
    "Roll Advert",
    "Save Settings",
    "Settings",
    "Submit Downgrade",
    "Submit Upgrade",
    "Thumbs Down",
    "Thumbs Up",
    "Upgrade",
]

# 예측 기준일은 관찰 종료 시각(UTC)입니다. 이후 이벤트가 피처에 섞이는 것을 막습니다.
CUTOFF_TS_MS = int(pd.Timestamp("2018-11-01T00:00:00Z").timestamp() * 1000)
MILLISECONDS_PER_DAY = 24 * 60 * 60 * 1000

# 기존 CatBoost와 개선 CatBoost가 같은 후보 설정을 비교하도록 둡니다.
PARAMETER_CANDIDATES = [
    {"iterations": 500, "depth": 4, "learning_rate": 0.05, "l2_leaf_reg": 3},
    {"iterations": 800, "depth": 4, "learning_rate": 0.03, "l2_leaf_reg": 5},
    {"iterations": 500, "depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 5},
    {"iterations": 800, "depth": 6, "learning_rate": 0.03, "l2_leaf_reg": 7},
    {"iterations": 500, "depth": 8, "learning_rate": 0.05, "l2_leaf_reg": 7},
    {"iterations": 800, "depth": 8, "learning_rate": 0.03, "l2_leaf_reg": 9},
]
RANDOM_STATE = 42


def page_slug(page: str) -> str:
    """행동 이름을 피처 열 이름에 넣기 좋은 영문 표기로 바꿉니다."""
    return re.sub(r"[^a-z0-9]+", "_", page.lower()).strip("_")


def extract_sequence_features(record: dict) -> dict:
    """고객의 최근 최대 50개 행동에서 횟수, 순서, 간격 요약 피처를 만듭니다."""
    events = record.get("events") or []
    pages = [str(event.get("page", "Unknown")) for event in events]
    timestamps = [int(event["ts"]) for event in events if event.get("ts") is not None]

    # 관찰 기간 종료 이후의 이벤트가 있으면 미래 정보 누수 가능성이 있으므로 중단합니다.
    if any(timestamp >= CUTOFF_TS_MS for timestamp in timestamps):
        raise ValueError(f"관찰 종료 이후 이벤트가 시퀀스에 있습니다: user_id={record.get('user_id')}")

    row = {
        "user_id": str(record["user_id"]),
        "sequence_split": record.get("split"),
        "sequence_churn": record.get("churn"),
        "seq_length": len(pages),
        "seq_unique_page_count": len(set(pages)),
        "seq_first_page": pages[0] if pages else "NoEvent",
        "seq_last_page": pages[-1] if pages else "NoEvent",
        "seq_most_common_page": Counter(pages).most_common(1)[0][0] if pages else "NoEvent",
    }

    # 전체 시퀀스, 마지막 10개 행동, 그 이전 행동의 종류별 횟수를 계산합니다.
    recent_pages = pages[-10:]
    earlier_pages = pages[:-10] if len(pages) > 10 else []
    total_counts = Counter(pages)
    recent_counts = Counter(recent_pages)
    earlier_counts = Counter(earlier_pages)
    known_pages = set(PAGE_TYPES)

    for page in PAGE_TYPES:
        slug = page_slug(page)
        row[f"seq_{slug}_count"] = total_counts[page]
        row[f"seq_recent10_{slug}_count"] = recent_counts[page]
        row[f"seq_earlier_{slug}_count"] = earlier_counts[page]
    row["seq_unknown_page_count"] = sum(page not in known_pages for page in pages)
    row["seq_recent10_nextsong_ratio"] = (
        recent_counts["NextSong"] / len(recent_pages) if recent_pages else 0.0
    )

    # 이어진 행동의 전환 종류와 반복 재생 흐름을 요약합니다.
    transitions = list(zip(pages[:-1], pages[1:]))
    row["seq_unique_transition_count"] = len(set(transitions))
    row["seq_repeated_page_ratio"] = (
        sum(previous == current for previous, current in transitions) / len(transitions)
        if transitions
        else 0.0
    )
    row["seq_max_nextsong_streak"] = 0
    current_streak = 0
    for page in pages:
        current_streak = current_streak + 1 if page == "NextSong" else 0
        row["seq_max_nextsong_streak"] = max(row["seq_max_nextsong_streak"], current_streak)
    for destination in ["Thumbs Down", "Thumbs Up", "Roll Advert", "Add to Playlist", "Downgrade"]:
        row[f"seq_nextsong_to_{page_slug(destination)}"] = sum(
            previous == "NextSong" and current == destination
            for previous, current in transitions
        )

    # 타임스탬프에서 활동 간격과 마지막 시퀀스 이벤트의 경과 시간을 구합니다.
    if timestamps:
        row["seq_days_since_last_event"] = max(
            0.0, (CUTOFF_TS_MS - max(timestamps)) / MILLISECONDS_PER_DAY
        )
        row["seq_span_hours"] = (max(timestamps) - min(timestamps)) / (1000 * 60 * 60)
        active_dates = {
            datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc).date()
            for timestamp in timestamps
        }
        row["seq_active_utc_dates"] = len(active_dates)
        gaps_minutes = np.diff(timestamps) / (1000 * 60)
        positive_gaps = gaps_minutes[gaps_minutes > 0]
        row["seq_median_gap_minutes"] = float(np.median(positive_gaps)) if len(positive_gaps) else 0.0
        row["seq_p90_gap_minutes"] = float(np.percentile(positive_gaps, 90)) if len(positive_gaps) else 0.0
        row["seq_max_gap_hours"] = float(np.max(positive_gaps) / 60) if len(positive_gaps) else 0.0
        row["seq_gaps_over_24h"] = int(np.sum(positive_gaps > 24 * 60))
    else:
        row["seq_days_since_last_event"] = 28.0
        row["seq_span_hours"] = 0.0
        row["seq_active_utc_dates"] = 0
        row["seq_median_gap_minutes"] = 0.0
        row["seq_p90_gap_minutes"] = 0.0
        row["seq_max_gap_hours"] = 0.0
        row["seq_gaps_over_24h"] = 0

    return row


def load_and_join_data() -> tuple[pd.DataFrame, list[str], list[str]]:
    """정형 데이터와 시퀀스 피처를 user_id로 결합하고 분할/라벨 일치를 확인합니다."""
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"정형 피처 파일을 찾을 수 없습니다: {FEATURES_PATH}")
    if not SEQUENCES_PATH.exists():
        raise FileNotFoundError(f"시퀀스 파일을 찾을 수 없습니다: {SEQUENCES_PATH}")

    # 정형 데이터에서 고객 ID를 문자열로 읽어 시퀀스 파일과 안전하게 연결합니다.
    features = pd.read_csv(FEATURES_PATH, dtype={"user_id": "string"})
    sequence_rows = []
    with SEQUENCES_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                sequence_rows.append(extract_sequence_features(json.loads(line)))
    sequence_features = pd.DataFrame(sequence_rows)
    sequence_features["user_id"] = sequence_features["user_id"].astype("string")

    # 시퀀스 JSONL의 고객 중복은 잘못된 일대일 결합을 만들 수 있어 검사합니다.
    if sequence_features["user_id"].duplicated().any():
        raise ValueError("시퀀스 파일에 중복 user_id가 있습니다.")

    # 같은 고객의 split과 정답이 두 원본에서 동일한지 확인한 뒤 입력에서 메타데이터를 제거합니다.
    joined = features.merge(sequence_features, on="user_id", how="left", validate="one_to_one")
    if joined["seq_length"].isna().any():
        raise ValueError("시퀀스 파일에서 찾지 못한 고객이 있습니다.")
    split_matches = joined[SPLIT_COLUMN].astype(str) == joined["sequence_split"].astype(str)
    label_matches = joined[TARGET].astype(int) == joined["sequence_churn"].astype(int)
    if not split_matches.all() or not label_matches.all():
        raise ValueError("정형 데이터와 시퀀스 데이터의 split 또는 churn 라벨이 일치하지 않습니다.")
    joined = joined.drop(columns=["sequence_split", "sequence_churn"])

    # CatBoost에 범주형으로 전달할 원래 요금제 및 시퀀스 페이지 이름 열을 지정합니다.
    sequence_categorical_features = [
        "seq_first_page",
        "seq_last_page",
        "seq_most_common_page",
    ]
    categorical_features = BASE_CATEGORICAL_FEATURES + sequence_categorical_features
    sequence_numeric_features = [
        column
        for column in joined.columns
        if column.startswith("seq_") and column not in sequence_categorical_features
    ]
    enhanced_features = BASE_FEATURES + sequence_numeric_features + sequence_categorical_features
    return joined, enhanced_features, categorical_features


def fit_best_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    categorical_features: list[str],
    experiment_name: str,
) -> tuple[CatBoostClassifier, dict, float, list[str]]:
    """고정 후보 설정을 train에서 학습하고 valid PR-AUC가 높은 모델을 고릅니다."""
    best_model = None
    best_parameters = None
    best_valid_pr_auc = -1.0
    search_results = []

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
            cat_features=categorical_features,
            eval_set=(X_valid, y_valid),
            early_stopping_rounds=100,
            use_best_model=True,
            verbose=False,
        )

        # 같은 외부 PR-AUC 지표로 기본 피처 실험과 시퀀스 피처 실험을 비교합니다.
        positive_class_index = list(candidate.classes_).index(1)
        probabilities = candidate.predict_proba(X_valid)[:, positive_class_index]
        valid_pr_auc = average_precision_score(y_valid, probabilities)
        line = (
            f"{experiment_name} | iterations={candidate_parameters['iterations']}, "
            f"depth={candidate_parameters['depth']}, "
            f"learning_rate={candidate_parameters['learning_rate']}, "
            f"l2_leaf_reg={candidate_parameters['l2_leaf_reg']}: "
            f"valid PR-AUC={valid_pr_auc:.4f}, "
            f"best_iteration={candidate.get_best_iteration()}"
        )
        search_results.append(line)
        print(line)

        if valid_pr_auc > best_valid_pr_auc:
            best_model = candidate
            best_parameters = candidate_parameters
            best_valid_pr_auc = float(valid_pr_auc)

    if best_model is None or best_parameters is None:
        raise RuntimeError(f"{experiment_name} 설정 선택에 실패했습니다.")
    return best_model, best_parameters, best_valid_pr_auc, search_results


def choose_f1_threshold(y_true: pd.Series, probabilities: np.ndarray) -> tuple[float, float]:
    """valid 확률로 F1을 가장 높이는 판정 임계값을 선택합니다."""
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
    experiment_name: str,
    split_name: str,
    model: CatBoostClassifier,
    split_df: pd.DataFrame,
    feature_columns: list[str],
    threshold: float,
) -> tuple[str, dict]:
    """분류 지표와 혼동 행렬을 계산해 텍스트 보고서로 반환합니다."""
    X = split_df[feature_columns]
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
        f"===== {experiment_name} | {split_name} ({len(split_df):,}명) =====",
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
    # 기본 14개 피처와 시퀀스에서 만든 피처를 고객별로 결합합니다.
    data, enhanced_features, enhanced_categorical_features = load_and_join_data()
    base_categorical_features = BASE_CATEGORICAL_FEATURES
    train_df = data[data[SPLIT_COLUMN] == "train"].copy()
    valid_df = data[data[SPLIT_COLUMN] == "valid"].copy()
    test_df = data[data[SPLIT_COLUMN] == "test"].copy()
    if train_df.empty or valid_df.empty or test_df.empty:
        raise ValueError("train, valid, test 데이터가 모두 있는지 확인해 주세요.")

    y_train = train_df[TARGET]
    y_valid = valid_df[TARGET]
    base_search_results = []
    enhanced_search_results = []
    reports = []
    test_metrics = {}

    # 기존 CatBoost와 같은 알고리즘/분할/설정 후보로 기본 피처 기준선을 다시 계산합니다.
    base_model, base_parameters, base_valid_pr_auc, search = fit_best_catboost(
        train_df[BASE_FEATURES],
        y_train,
        valid_df[BASE_FEATURES],
        y_valid,
        base_categorical_features,
        "기본 피처",
    )
    base_search_results.extend(search)
    base_valid_probabilities = base_model.predict_proba(valid_df[BASE_FEATURES])[:, 1]
    base_threshold, base_valid_f1 = choose_f1_threshold(y_valid, base_valid_probabilities)
    base_valid_report, _ = evaluate_split(
        "기본 피처", "검증(valid)", base_model, valid_df, BASE_FEATURES, base_threshold
    )
    base_test_report, base_test_metrics = evaluate_split(
        "기본 피처", "최종 평가(test)", base_model, test_df, BASE_FEATURES, base_threshold
    )
    reports.extend([base_valid_report, base_test_report])
    test_metrics["기본 피처"] = base_test_metrics

    # 같은 방식으로 시퀀스 피처를 추가한 모델을 학습하고 valid에서 선택합니다.
    enhanced_model, enhanced_parameters, enhanced_valid_pr_auc, search = fit_best_catboost(
        train_df[enhanced_features],
        y_train,
        valid_df[enhanced_features],
        y_valid,
        enhanced_categorical_features,
        "시퀀스 피처 추가",
    )
    enhanced_search_results.extend(search)
    enhanced_valid_probabilities = enhanced_model.predict_proba(valid_df[enhanced_features])[:, 1]
    enhanced_threshold, enhanced_valid_f1 = choose_f1_threshold(y_valid, enhanced_valid_probabilities)
    enhanced_valid_report, _ = evaluate_split(
        "시퀀스 피처 추가",
        "검증(valid)",
        enhanced_model,
        valid_df,
        enhanced_features,
        enhanced_threshold,
    )
    enhanced_test_report, enhanced_test_metrics = evaluate_split(
        "시퀀스 피처 추가",
        "최종 평가(test)",
        enhanced_model,
        test_df,
        enhanced_features,
        enhanced_threshold,
    )
    reports.extend([enhanced_valid_report, enhanced_test_report])
    test_metrics["시퀀스 피처 추가"] = enhanced_test_metrics

    # 테스트 점수의 차이를 표시해 새 피처의 변화량을 읽기 쉽게 합니다.
    comparison_lines = [
        "===== test 비교: 시퀀스 피처 추가 - 기본 피처 =====",
        f"PR-AUC 변화: {enhanced_test_metrics['pr_auc'] - base_test_metrics['pr_auc']:+.4f}",
        f"F1 변화:     {enhanced_test_metrics['f1'] - base_test_metrics['f1']:+.4f}",
        f"Recall 변화: {enhanced_test_metrics['recall'] - base_test_metrics['recall']:+.4f}",
        f"Precision 변화: {enhanced_test_metrics['precision'] - base_test_metrics['precision']:+.4f}",
    ]

    # 실험 조건, valid 선택값, test 비교, 상세 지표를 한 TXT에 저장합니다.
    summary = [
        f"정형 데이터: {FEATURES_PATH}",
        f"행동 시퀀스: {SEQUENCES_PATH}",
        f"분할 크기: train={len(train_df):,}, valid={len(valid_df):,}, test={len(test_df):,}",
        f"기본 피처 수: {len(BASE_FEATURES)}",
        f"시퀀스 피처 추가 후 피처 수: {len(enhanced_features)}",
        "시퀀스에서 행동 종류별 전체/최근 10개/이전 행동 횟수, 전환, 반복, 활동 간격, 마지막 행동 피처를 만들었습니다.",
        f"기본 모델 valid PR-AUC={base_valid_pr_auc:.4f}, 선택 설정={base_parameters}, 임계값={base_threshold:.6f}, valid F1={base_valid_f1:.4f}",
        f"개선 모델 valid PR-AUC={enhanced_valid_pr_auc:.4f}, 선택 설정={enhanced_parameters}, 임계값={enhanced_threshold:.6f}, valid F1={enhanced_valid_f1:.4f}",
    ]
    full_report = "\n".join(
        summary
        + ["\n하이퍼파라미터 탐색 결과:"]
        + base_search_results
        + enhanced_search_results
        + comparison_lines
        + reports
    )
    print(full_report)
    RESULTS_PATH.write_text(full_report + "\n", encoding="utf-8")
    COLLECTED_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    COLLECTED_RESULTS_PATH.write_text(full_report + "\n", encoding="utf-8")

    # 재사용할 개선 모델과 임계값, 설정, 피처 이름을 함께 보관합니다.
    joblib.dump(
        {
            "model": enhanced_model,
            "decision_threshold": enhanced_threshold,
            "best_parameters": enhanced_parameters,
            "feature_columns": enhanced_features,
            "categorical_features": enhanced_categorical_features,
        },
        MODEL_PATH,
    )
    print(f"\n비교 결과 저장: {RESULTS_PATH}")
    print(f"결과 모음 폴더에도 저장: {COLLECTED_RESULTS_PATH}")
    print(f"시퀀스 피처 CatBoost 저장: {MODEL_PATH}")


if __name__ == "__main__":
    main()
