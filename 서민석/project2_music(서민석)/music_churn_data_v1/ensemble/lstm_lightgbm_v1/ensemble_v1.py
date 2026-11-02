"""기존 v1 LSTM 시퀀스 모델과 LightGBM 정형 모델을 late-fusion 합니다.

원본 5_모델링.ipynb처럼 두 모델의 이탈 확률을 가중 평균합니다.
가중치는 valid ROC-AUC로 고르고, test는 마지막 평가에만 사용합니다.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset


# 결과·앙상블 폴더에서 v1 데이터셋 루트까지 경로를 계산합니다.
SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR.parents[1]
PROCESSED_DIR = DATASET_DIR / "data" / "processed"
FEATURES_PATH = PROCESSED_DIR / "customer_features_20181101_v1_with_split.csv"
SEQUENCES_PATH = PROCESSED_DIR / "customer_sequences_20181101_v1.jsonl"
LSTM_MODEL_PATH = DATASET_DIR / "deep_learning" / "lstm" / "lstm_v1.pt"
LIGHTGBM_MODEL_PATH = DATASET_DIR / "machine_learning" / "lightgbm" / "lightgbm_v1.joblib"

RESULT_PATH = SCRIPT_DIR / "lstm_lightgbm_ensemble_v1_evaluation_results.txt"
CENTRAL_RESULT_PATH = DATASET_DIR / "result" / RESULT_PATH.name
PREDICTIONS_PATH = SCRIPT_DIR / "lstm_lightgbm_ensemble_v1_test_predictions.csv"
CONFIG_PATH = SCRIPT_DIR / "lstm_lightgbm_ensemble_v1_config.json"
ENSEMBLE_PATH = SCRIPT_DIR / "lstm_lightgbm_ensemble_v1.joblib"

ID_COLUMN = "user_id"
TARGET = "churn"
SPLIT_COLUMN = "split"
PAD_ID = 0
UNKNOWN_ID = 1
MAX_BATCH_SIZE = 256

# 노트북과 같은 LightGBM 70~90% 가중치 탐색 범위를 사용합니다.
LIGHTGBM_WEIGHTS = np.linspace(0.70, 0.90, 21)
RECALL_TARGETS = (0.70, 0.80)


class SequenceDataset(Dataset):
    """고객별 시퀀스와 실제 길이를 배치에 전달합니다."""

    def __init__(self, token_matrix: np.ndarray, lengths: np.ndarray):
        self.tokens = torch.as_tensor(token_matrix, dtype=torch.long)
        # pack_padded_sequence는 0 길이를 받지 않아 빈 시퀀스만 길이 1로 처리합니다.
        self.lengths = torch.as_tensor(np.maximum(lengths, 1), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, index: int):
        return self.tokens[index], self.lengths[index]


class ChurnLSTM(nn.Module):
    """기존 lstm_v1.pt와 같은 이름·구조로 LSTM 체크포인트를 불러옵니다."""

    def __init__(self, vocabulary_size: int, embedding_dim: int = 32,
                 hidden_dim: int = 64, dropout: float = 0.30):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=PAD_ID)
        self.dropout = nn.Dropout(dropout)
        self.sequence = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(tokens)
        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.sequence(packed)
        return self.output(self.dropout(hidden[-1])).squeeze(-1)


def load_components():
    """저장된 LSTM 및 LightGBM을 읽고 파일 누락을 먼저 확인합니다."""
    missing = [
        path for path in (FEATURES_PATH, SEQUENCES_PATH, LSTM_MODEL_PATH, LIGHTGBM_MODEL_PATH)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError("필요한 데이터/모델 파일이 없습니다:\n" + "\n".join(map(str, missing)))

    # 체크포인트는 프로젝트에서 직접 저장한 신뢰 가능한 로컬 파일입니다.
    try:
        lstm_artifact = torch.load(LSTM_MODEL_PATH, map_location="cpu", weights_only=True)
    except TypeError:  # weights_only 옵션이 없는 구버전 PyTorch 지원
        lstm_artifact = torch.load(LSTM_MODEL_PATH, map_location="cpu")
    if not isinstance(lstm_artifact, dict) or "state_dict" not in lstm_artifact:
        raise ValueError("LSTM 체크포인트에 state_dict가 없습니다. lstm_v1.pt를 확인하세요.")
    if "vocabulary" not in lstm_artifact:
        raise ValueError("LSTM 체크포인트에 학습 vocabulary가 없습니다.")

    lightgbm_artifact = joblib.load(LIGHTGBM_MODEL_PATH)
    if not isinstance(lightgbm_artifact, dict) or "model" not in lightgbm_artifact:
        raise ValueError("LightGBM 파일에 model 항목이 없습니다.")
    if "feature_columns" not in lightgbm_artifact:
        raise ValueError("LightGBM 파일에 feature_columns가 없습니다.")
    return lstm_artifact, lightgbm_artifact


def load_aligned_data() -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """정형 데이터와 시퀀스의 user_id, split, churn 라벨을 교차 확인합니다."""
    features = pd.read_csv(FEATURES_PATH, dtype={ID_COLUMN: "string"})
    required = {ID_COLUMN, TARGET, SPLIT_COLUMN}
    missing = required - set(features.columns)
    if missing:
        raise ValueError(f"정형 CSV에 필수 열이 없습니다: {sorted(missing)}")
    if features[ID_COLUMN].duplicated().any():
        raise ValueError("정형 CSV에 중복 user_id가 있습니다.")

    records = {}
    cutoff_ms = int(pd.Timestamp("2018-11-01T00:00:00Z").timestamp() * 1000)
    with SEQUENCES_PATH.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            user_id = str(row[ID_COLUMN])
            if user_id in records:
                raise ValueError(f"JSONL에 중복 user_id가 있습니다: {user_id}")
            events = row.get("events") or []
            pages = []
            for event in events:
                if event.get("ts") is not None and int(event["ts"]) >= cutoff_ms:
                    raise ValueError(f"관찰 종료 이후 이벤트: JSONL line {line_number}")
                pages.append(str(event.get("page") or "<UNK>"))
            records[user_id] = {
                "split": str(row.get(SPLIT_COLUMN)),
                TARGET: int(row[TARGET]),
                "pages": pages,
            }

    feature_ids = set(features[ID_COLUMN].astype(str))
    if feature_ids != set(records):
        raise ValueError("정형 CSV와 JSONL의 고객 ID 집합이 일치하지 않습니다.")
    pages_by_id = {}
    for row in features[[ID_COLUMN, TARGET, SPLIT_COLUMN]].itertuples(index=False):
        user_id, label, split = str(row[0]), int(row[1]), str(row[2])
        sequence = records[user_id]
        if sequence[TARGET] != label or sequence["split"] != split:
            raise ValueError(f"user_id={user_id}의 정답 또는 split이 일치하지 않습니다.")
        pages_by_id[user_id] = sequence["pages"]
    return features, pages_by_id


def encode_sequences(user_ids: list[str], pages_by_id: dict[str, list[str]],
                     vocabulary: dict[str, int], max_length: int) -> tuple[np.ndarray, np.ndarray]:
    """학습 때 저장한 vocabulary로 행동을 정수 ID로 인코딩합니다."""
    matrix = np.full((len(user_ids), max_length), PAD_ID, dtype=np.int64)
    lengths = np.zeros(len(user_ids), dtype=np.int64)
    for row_index, user_id in enumerate(user_ids):
        pages = pages_by_id[user_id][-max_length:]
        tokens = [vocabulary.get(page, UNKNOWN_ID) for page in pages]
        lengths[row_index] = len(tokens)
        if tokens:
            matrix[row_index, :len(tokens)] = tokens
    return matrix, lengths


def lstm_probabilities(model: ChurnLSTM, token_matrix: np.ndarray,
                       lengths: np.ndarray, device: torch.device) -> np.ndarray:
    """한 split의 LSTM 이탈 확률을 순서를 유지해 계산합니다."""
    loader = DataLoader(SequenceDataset(token_matrix, lengths), batch_size=MAX_BATCH_SIZE)
    model.eval()
    output = []
    with torch.no_grad():
        for tokens, actual_lengths in loader:
            logits = model(tokens.to(device), actual_lengths.to(device))
            output.extend(torch.sigmoid(logits).cpu().numpy().tolist())
    return np.asarray(output, dtype=np.float64)


def choose_thresholds(y_true: np.ndarray, probabilities: np.ndarray) -> dict:
    """valid에서 F2 최대 임계값 및 재현율 목표별 임계값을 고릅니다."""
    thresholds = np.unique(np.concatenate(([0.0, 0.5, 1.0], probabilities)))
    rows = []
    for threshold in thresholds:
        prediction = (probabilities >= threshold).astype(int)
        rows.append({
            "threshold": float(threshold),
            "precision": float(precision_score(y_true, prediction, zero_division=0)),
            "recall": float(recall_score(y_true, prediction, zero_division=0)),
            "f2": float(fbeta_score(y_true, prediction, beta=2, zero_division=0)),
        })
    choices = {"F2 maximum (recall-oriented)": max(rows, key=lambda row: row["f2"])}
    for target in RECALL_TARGETS:
        eligible = [row for row in rows if row["recall"] >= target]
        if eligible:
            choices[f"Recall at least {target:.0%}"] = max(
                eligible, key=lambda row: (row["precision"], row["threshold"])
            )
    return choices


def evaluate(y_true: np.ndarray, probabilities: np.ndarray,
             threshold: float, title: str) -> str:
    """threshold에 따른 분류 지표와 threshold와 무관한 AUC 지표를 출력합니다."""
    prediction = (probabilities >= threshold).astype(int)
    return "\n".join([
        f"===== {title} | threshold={threshold:.6f} | n={len(y_true):,} =====",
        f"Accuracy:  {accuracy_score(y_true, prediction):.4f}",
        f"Precision: {precision_score(y_true, prediction, zero_division=0):.4f}",
        f"Recall:    {recall_score(y_true, prediction, zero_division=0):.4f}",
        f"F2-score:  {fbeta_score(y_true, prediction, beta=2, zero_division=0):.4f}",
        f"ROC-AUC:   {roc_auc_score(y_true, probabilities):.4f}",
        f"PR-AUC:    {average_precision_score(y_true, probabilities):.4f}",
        "Confusion matrix (rows=true, columns=predicted):\n" + str(confusion_matrix(y_true, prediction)),
    ])


def main() -> None:
    """기존 두 모델을 결합하고 valid 선택 후 test를 한 번 평가합니다."""
    lstm_artifact, lightgbm_artifact = load_components()
    features, pages_by_id = load_aligned_data()
    frames = {name: features[features[SPLIT_COLUMN] == name].copy()
              for name in ("train", "valid", "test")}
    if any(frame.empty for frame in frames.values()):
        raise ValueError("기존 v1 데이터에 train, valid, test 분할이 모두 필요합니다.")

    vocabulary = lstm_artifact["vocabulary"]
    max_length = int(lstm_artifact.get("max_sequence_length", 50))
    if vocabulary.get("<PAD>") != PAD_ID or vocabulary.get("<UNK>") != UNKNOWN_ID:
        raise ValueError("LSTM vocabulary의 PAD/UNK ID가 예상한 구조와 다릅니다.")
    model = ChurnLSTM(vocabulary_size=len(vocabulary)).to(
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    model.load_state_dict(lstm_artifact["state_dict"])
    device = next(model.parameters()).device
    model.eval()

    lgbm_bundle = lightgbm_artifact["model"]
    feature_columns = lightgbm_artifact["feature_columns"]
    absent = [column for column in feature_columns if column not in features.columns]
    if absent:
        raise ValueError(f"LightGBM에서 필요한 입력 피처가 없습니다: {absent}")

    probabilities = {name: {"LSTM": None, "LightGBM": None} for name in frames}
    labels = {name: frame[TARGET].astype(int).to_numpy() for name, frame in frames.items()}
    for split_name, frame in frames.items():
        ids = frame[ID_COLUMN].astype(str).tolist()
        token_matrix, lengths = encode_sequences(ids, pages_by_id, vocabulary, max_length)
        probabilities[split_name]["LSTM"] = lstm_probabilities(
            model, token_matrix, lengths, device
        )
        positive_index = list(lgbm_bundle.classes_).index(1)
        probabilities[split_name]["LightGBM"] = lgbm_bundle.predict_proba(
            frame[feature_columns]
        )[:, positive_index]

    # 노트북과 같은 0.70~0.90 범위에서 valid ROC-AUC가 가장 높은 비율을 선택합니다.
    weight_rows = []
    for lightgbm_weight in LIGHTGBM_WEIGHTS:
        lstm_weight = 1.0 - float(lightgbm_weight)
        valid_mix = (
            float(lightgbm_weight) * probabilities["valid"]["LightGBM"]
            + lstm_weight * probabilities["valid"]["LSTM"]
        )
        weight_rows.append({
            "lightgbm_weight": float(lightgbm_weight),
            "lstm_weight": lstm_weight,
            "valid_roc_auc": float(roc_auc_score(labels["valid"], valid_mix)),
            "valid_pr_auc": float(average_precision_score(labels["valid"], valid_mix)),
        })
    winner = max(weight_rows, key=lambda row: row["valid_roc_auc"])
    lgbm_weight, lstm_weight = winner["lightgbm_weight"], winner["lstm_weight"]

    mixed = {}
    for split_name in frames:
        mixed[split_name] = (
            lgbm_weight * probabilities[split_name]["LightGBM"]
            + lstm_weight * probabilities[split_name]["LSTM"]
        )
    thresholds = choose_thresholds(labels["valid"], mixed["valid"])
    # 단독 모델 비교도 저장 당시 임계값 대신 각 모델의 valid F2 기준으로 맞춥니다.
    lgbm_thresholds = choose_thresholds(
        labels["valid"], probabilities["valid"]["LightGBM"]
    )
    lstm_thresholds = choose_thresholds(
        labels["valid"], probabilities["valid"]["LSTM"]
    )

    lines = [
        "LSTM + LightGBM late-fusion ensemble, original v1 split",
        f"Feature data: {FEATURES_PATH}", f"Sequence data: {SEQUENCES_PATH}",
        f"LSTM checkpoint: {LSTM_MODEL_PATH}", f"LightGBM artifact: {LIGHTGBM_MODEL_PATH}",
        f"Split sizes: train={len(frames['train']):,}, valid={len(frames['valid']):,}, test={len(frames['test']):,}",
        f"Train churn rate: {labels['train'].mean():.2%}",
        "Combination: weighted average of LSTM sequence and LightGBM tabular probabilities",
        "Weight range: LightGBM 0.70-0.90 (21 candidates), LSTM 0.30-0.10",
        "Weight selection: highest validation ROC-AUC; test data was not used to select weights or thresholds.",
        f"Selected weights: LightGBM={lgbm_weight:.2f}, LSTM={lstm_weight:.2f}",
        f"Selected validation ROC-AUC: {winner['valid_roc_auc']:.4f}",
        f"Selected validation PR-AUC: {winner['valid_pr_auc']:.4f}",
        "\n===== Base and ensemble AUC comparison =====",
    ]
    for split_name in ("valid", "test"):
        lines.append(f"--- {split_name.upper()} ---")
        for model_name in ("LightGBM", "LSTM"):
            score = probabilities[split_name][model_name]
            lines.append(
                f"{model_name}: ROC-AUC={roc_auc_score(labels[split_name], score):.4f}, "
                f"PR-AUC={average_precision_score(labels[split_name], score):.4f}"
            )
        lines.append(
            f"Ensemble: ROC-AUC={roc_auc_score(labels[split_name], mixed[split_name]):.4f}, "
            f"PR-AUC={average_precision_score(labels[split_name], mixed[split_name]):.4f}"
        )
    lines.extend(["\n===== Weight search on valid ====="])
    lines.extend(
        f"LightGBM={row['lightgbm_weight']:.2f}, LSTM={row['lstm_weight']:.2f}: "
        f"valid ROC-AUC={row['valid_roc_auc']:.4f}, PR-AUC={row['valid_pr_auc']:.4f}"
        for row in weight_rows
    )
    lines.append("\n===== Thresholds selected on valid (F2/recall focus) =====")
    for name, choice in thresholds.items():
        lines.append(
            f"{name}: threshold={choice['threshold']:.6f}, precision={choice['precision']:.4f}, "
            f"recall={choice['recall']:.4f}, F2={choice['f2']:.4f}"
        )
    lines.append("\n===== Final test evaluation =====")
    for name, choice in thresholds.items():
        lines.append(evaluate(labels["test"], mixed["test"], choice["threshold"], name))
    lines.append("\n===== Base model test evaluation at validation F2 thresholds =====")
    lines.append(evaluate(
        labels["test"], probabilities["test"]["LightGBM"],
        lgbm_thresholds["F2 maximum (recall-oriented)"]["threshold"],
        "LightGBM only (validation F2 threshold)"
    ))
    lines.append(evaluate(
        labels["test"], probabilities["test"]["LSTM"],
        lstm_thresholds["F2 maximum (recall-oriented)"]["threshold"],
        "LSTM only (validation F2 threshold)"
    ))

    report = "\n".join(lines) + "\n"
    print(report)

    # test에서 고른 값이 아니라 valid에서 결정된 가중치와 임계값을 저장합니다.
    config = {
        "method": "late fusion weighted probability average",
        "weight_selection_metric": "validation ROC-AUC",
        "threshold_selection_metric": "validation F2",
        "lightgbm_weight": lgbm_weight,
        "lstm_weight": lstm_weight,
        "valid_roc_auc": winner["valid_roc_auc"],
        "valid_pr_auc": winner["valid_pr_auc"],
        "threshold_choices": thresholds,
        "component_paths": {
            "lightgbm": str(LIGHTGBM_MODEL_PATH),
            "lstm": str(LSTM_MODEL_PATH),
        },
        "split_sizes": {name: len(frame) for name, frame in frames.items()},
    }
    CONFIG_PATH.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    RESULT_PATH.write_text(report, encoding="utf-8")
    CENTRAL_RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CENTRAL_RESULT_PATH.write_text(report, encoding="utf-8")

    prediction_frame = pd.DataFrame({
        ID_COLUMN: frames["test"][ID_COLUMN].astype(str).to_numpy(),
        "churn_true": labels["test"],
        "lightgbm_probability": probabilities["test"]["LightGBM"],
        "lstm_probability": probabilities["test"]["LSTM"],
        "ensemble_probability": mixed["test"],
    })
    for name, choice in thresholds.items():
        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("%", "pct")
        prediction_frame[f"predicted_{safe_name}"] = (
            mixed["test"] >= choice["threshold"]
        ).astype(int)
    prediction_frame.to_csv(PREDICTIONS_PATH, index=False, encoding="utf-8-sig")

    # 묶음 파일에 두 학습 모델과 결합 가중치·운영 임계값을 함께 넣습니다.
    joblib.dump({
        "lightgbm_artifact": lightgbm_artifact,
        "lstm_state_dict": lstm_artifact["state_dict"],
        "lstm_vocabulary": vocabulary,
        "lstm_max_sequence_length": max_length,
        "weights": {"LightGBM": lgbm_weight, "LSTM": lstm_weight},
        "threshold_choices": thresholds,
        "feature_columns": feature_columns,
    }, ENSEMBLE_PATH)
    print(f"\nEnsemble artifact: {ENSEMBLE_PATH}")
    print(f"Evaluation results: {RESULT_PATH}")
    print(f"Collected result: {CENTRAL_RESULT_PATH}")
    print(f"Test predictions: {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
