"""원본 v1 공통 분할의 행동 시퀀스로 GRU 이탈 분류기를 학습합니다."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset


SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR.parents[1]
PROCESSED_DIR = DATASET_DIR / "data" / "processed"
FEATURES_PATH = PROCESSED_DIR / "customer_features_20181101_v1_with_split.csv"
SEQUENCES_PATH = PROCESSED_DIR / "customer_sequences_20181101_v1.jsonl"
MODEL_PATH = SCRIPT_DIR / "gru_v1.pt"
CONFIG_PATH = SCRIPT_DIR / "gru_v1_config.json"
RESULT_PATH = SCRIPT_DIR / "gru_v1_evaluation_results.txt"
CENTRAL_RESULT_PATH = DATASET_DIR / "result" / RESULT_PATH.name
PREDICTIONS_PATH = SCRIPT_DIR / "gru_v1_test_predictions.csv"

TARGET = "churn"
SPLIT_COLUMN = "split"
ID_COLUMN = "user_id"
MAX_SEQUENCE_LENGTH = 50
PAD_ID = 0
UNKNOWN_ID = 1
RANDOM_STATE = 42
BATCH_SIZE = 128
MAX_EPOCHS = 50
PATIENCE = 8
RECALL_TARGETS = [0.70, 0.80]


def set_seeds(seed: int = RANDOM_STATE) -> None:
    """난수 시드를 고정합니다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SequenceDataset(Dataset):
    """시퀀스, 실제 길이, 이탈 정답을 묶는 PyTorch Dataset입니다."""

    def __init__(self, sequences: np.ndarray, lengths: np.ndarray, labels: np.ndarray):
        self.sequences = torch.as_tensor(sequences, dtype=torch.long)
        # pack_padded_sequence는 길이 0을 받을 수 없으므로 빈 시퀀스는 PAD 1칸으로 둡니다.
        self.lengths = torch.as_tensor(np.maximum(lengths, 1), dtype=torch.long)
        self.labels = torch.as_tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.sequences[index], self.lengths[index], self.labels[index]


class ChurnGRU(nn.Module):
    """페이지 행동 ID 흐름을 학습하는 임베딩-GRU 이진 분류기입니다."""

    def __init__(
        self,
        vocabulary_size: int,
        embedding_dim: int = 32,
        hidden_dim: int = 64,
        dropout: float = 0.30,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            vocabulary_size, embedding_dim, padding_idx=PAD_ID
        )
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, sequences: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(sequences)
        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.gru(packed)
        return self.output(self.dropout(hidden[-1])).squeeze(-1)


def load_aligned_sequences() -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """정형 테이블과 시퀀스의 ID, 라벨, split이 맞는지 검사합니다."""
    if not FEATURES_PATH.exists() or not SEQUENCES_PATH.exists():
        raise FileNotFoundError(
            f"입력 파일을 찾을 수 없습니다:\n{FEATURES_PATH}\n{SEQUENCES_PATH}"
        )
    features = pd.read_csv(FEATURES_PATH, dtype={ID_COLUMN: "string"})
    required = [ID_COLUMN, TARGET, SPLIT_COLUMN]
    missing = [column for column in required if column not in features.columns]
    if missing:
        raise ValueError(f"정형 CSV에 필요한 컬럼이 없습니다: {missing}")
    if features[ID_COLUMN].duplicated().any():
        raise ValueError("정형 CSV에 중복 user_id가 있습니다.")

    sequence_records = {}
    cutoff_ms = int(pd.Timestamp("2018-11-01T00:00:00Z").timestamp() * 1000)
    with SEQUENCES_PATH.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            user_id = str(record[ID_COLUMN])
            if user_id in sequence_records:
                raise ValueError(f"JSONL에 중복 user_id가 있습니다: {user_id}")
            events = record.get("events") or []
            pages = []
            for event in events:
                if event.get("ts") is not None and int(event["ts"]) >= cutoff_ms:
                    raise ValueError(
                        f"관찰 종료 이후 이벤트 발견: line={line_number}, user_id={user_id}"
                    )
                pages.append(str(event.get("page") or "<UNK>"))
            sequence_records[user_id] = {
                "split": str(record.get(SPLIT_COLUMN)),
                TARGET: int(record[TARGET]),
                "pages": pages[-MAX_SEQUENCE_LENGTH:],
            }

    feature_ids = set(features[ID_COLUMN].astype(str))
    sequence_ids = set(sequence_records)
    if feature_ids != sequence_ids:
        raise ValueError(
            f"정형/시퀀스 ID가 다릅니다: 정형만 {len(feature_ids - sequence_ids)}, "
            f"시퀀스만 {len(sequence_ids - feature_ids)}"
        )

    pages_by_id = {}
    for row in features[[ID_COLUMN, TARGET, SPLIT_COLUMN]].itertuples(index=False):
        user_id = str(row[0])
        label = int(row[1])
        split = str(row[2])
        record = sequence_records[user_id]
        if record[TARGET] != label or record["split"] != split:
            raise ValueError(f"user_id={user_id}의 라벨 또는 split이 일치하지 않습니다.")
        pages_by_id[user_id] = record["pages"]
    return features, pages_by_id


def build_vocabulary(train_ids: list[str], pages_by_id: dict[str, list[str]]) -> dict[str, int]:
    """행동 사전은 train 시퀀스에서만 만들고 valid/test의 새 행동은 UNK로 둡니다."""
    training_pages = sorted({
        page
        for user_id in train_ids
        for page in pages_by_id[user_id]
        if page not in {"<PAD>", "<UNK>"}
    })
    vocabulary = {"<PAD>": PAD_ID, "<UNK>": UNKNOWN_ID}
    vocabulary.update({
        page: index + 2 for index, page in enumerate(training_pages)
    })
    return vocabulary


def encode_sequences(
    user_ids: list[str],
    pages_by_id: dict[str, list[str]],
    vocabulary: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    """시간순 토큰을 정수 ID로 바꾸고 반복 없이 오른쪽 PAD 처리합니다."""
    token_matrix = np.full(
        (len(user_ids), MAX_SEQUENCE_LENGTH), PAD_ID, dtype=np.int64
    )
    lengths = np.zeros(len(user_ids), dtype=np.int64)
    for row_index, user_id in enumerate(user_ids):
        pages = pages_by_id[user_id][-MAX_SEQUENCE_LENGTH:]
        tokens = [vocabulary.get(page, UNKNOWN_ID) for page in pages]
        lengths[row_index] = len(tokens)
        if tokens:
            token_matrix[row_index, :len(tokens)] = tokens
    return token_matrix, lengths


def predict_probabilities(
    model: ChurnGRU,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """모델의 sigmoid 예측 확률과 라벨을 반환합니다."""
    model.eval()
    probabilities = []
    labels_all = []
    with torch.no_grad():
        for sequences, lengths, labels in loader:
            logits = model(sequences.to(device), lengths.to(device))
            probabilities.extend(torch.sigmoid(logits).cpu().numpy().tolist())
            labels_all.extend(labels.numpy().tolist())
    return np.asarray(probabilities), np.asarray(labels_all, dtype=np.int64)


def choose_thresholds(y_true: np.ndarray, probabilities: np.ndarray) -> dict:
    """valid에서 F2 우선 기준과 재현율 목표별 임계값을 선택합니다."""
    candidate_thresholds = np.unique(
        np.concatenate(([0.0, 0.5, 1.0], probabilities))
    )
    rows = []
    for threshold in candidate_thresholds:
        predictions = (probabilities >= threshold).astype(int)
        rows.append({
            "threshold": float(threshold),
            "precision": float(precision_score(y_true, predictions, zero_division=0)),
            "recall": float(recall_score(y_true, predictions, zero_division=0)),
            "f1": float(f1_score(y_true, predictions, zero_division=0)),
            "f2": float(fbeta_score(y_true, predictions, beta=2, zero_division=0)),
        })
    choices = {
        "F2 최대 (재현율 우선)": max(rows, key=lambda row: row["f2"]),
        "F1 최대": max(rows, key=lambda row: row["f1"]),
    }
    for target in RECALL_TARGETS:
        eligible = [row for row in rows if row["recall"] >= target]
        if eligible:
            choices[f"재현율 {target:.0%} 이상"] = max(
                eligible, key=lambda row: (row["precision"], row["threshold"])
            )
    return choices


def evaluate(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    scenario: str,
) -> str:
    """test 임계값별 분류 지표와 혼동 행렬을 만듭니다."""
    predictions = (probabilities >= threshold).astype(int)
    return "\n".join([
        f"===== {scenario} | threshold={threshold:.6f} | n={len(y_true):,} =====",
        f"Accuracy:  {accuracy_score(y_true, predictions):.4f}",
        f"Precision: {precision_score(y_true, predictions, zero_division=0):.4f}",
        f"Recall:    {recall_score(y_true, predictions, zero_division=0):.4f}",
        f"F1-score:  {f1_score(y_true, predictions, zero_division=0):.4f}",
        f"F2-score:  {fbeta_score(y_true, predictions, beta=2, zero_division=0):.4f}",
        f"ROC-AUC:   {roc_auc_score(y_true, probabilities):.4f}",
        f"PR-AUC:    {average_precision_score(y_true, probabilities):.4f}",
        "Classification report:",
        classification_report(y_true, predictions, zero_division=0),
        "Confusion matrix (rows=true, columns=predicted):",
        str(confusion_matrix(y_true, predictions)),
    ])


def main() -> None:
    set_seeds()
    features, pages_by_id = load_aligned_sequences()
    train = features[features[SPLIT_COLUMN] == "train"].copy()
    valid = features[features[SPLIT_COLUMN] == "valid"].copy()
    test = features[features[SPLIT_COLUMN] == "test"].copy()
    if any(frame.empty for frame in (train, valid, test)):
        raise ValueError("기존 split에 train, valid, test가 모두 있어야 합니다.")

    train_ids = train[ID_COLUMN].astype(str).tolist()
    valid_ids = valid[ID_COLUMN].astype(str).tolist()
    test_ids = test[ID_COLUMN].astype(str).tolist()
    vocabulary = build_vocabulary(train_ids, pages_by_id)
    x_train, length_train = encode_sequences(train_ids, pages_by_id, vocabulary)
    x_valid, length_valid = encode_sequences(valid_ids, pages_by_id, vocabulary)
    x_test, length_test = encode_sequences(test_ids, pages_by_id, vocabulary)
    y_train = train[TARGET].astype(np.int64).to_numpy()
    y_valid = valid[TARGET].astype(np.int64).to_numpy()
    y_test = test[TARGET].astype(np.int64).to_numpy()

    train_dataset = SequenceDataset(x_train, length_train, y_train)
    valid_dataset = SequenceDataset(x_valid, length_valid, y_valid)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(RANDOM_STATE),
    )
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChurnGRU(vocabulary_size=len(vocabulary)).to(device)
    positive_count = max(int(y_train.sum()), 1)
    negative_count = int(len(y_train) - positive_count)
    positive_weight_value = negative_count / positive_count
    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(positive_weight_value, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_state = None
    best_epoch = 0
    best_valid_f2 = -1.0
    best_valid_ap = -1.0
    patience_left = PATIENCE
    history = []

    # 최적 epoch는 매 epoch valid에서 임계값을 조정한 F2로 고릅니다.
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        losses = []
        for sequences, lengths, labels in train_loader:
            sequences = sequences.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(sequences, lengths)
            loss = loss_function(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        valid_probabilities, valid_labels = predict_probabilities(
            model, valid_loader, device
        )
        valid_thresholds = choose_thresholds(valid_labels, valid_probabilities)
        valid_f2 = valid_thresholds["F2 최대 (재현율 우선)"]["f2"]
        valid_ap = float(average_precision_score(valid_labels, valid_probabilities))
        history.append(
            f"epoch={epoch:03d}, train_loss={np.mean(losses):.4f}, "
            f"valid_F2={valid_f2:.4f}, valid_PR-AUC={valid_ap:.4f}"
        )
        print(history[-1])

        if valid_f2 > best_valid_f2:
            best_valid_f2 = valid_f2
            best_valid_ap = valid_ap
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            patience_left = PATIENCE
        else:
            patience_left -= 1
            if patience_left == 0:
                break

    if best_state is None:
        raise RuntimeError("GRU 최적 상태를 저장하지 못했습니다.")
    model.load_state_dict(best_state)
    model.to(device)
    valid_probabilities, valid_labels = predict_probabilities(model, valid_loader, device)
    threshold_choices = choose_thresholds(valid_labels, valid_probabilities)
    decision_threshold = threshold_choices["F2 최대 (재현율 우선)"]["threshold"]

    # 임계값과 모델을 확정한 뒤에만 test를 평가합니다.
    test_dataset = SequenceDataset(x_test, length_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_probabilities, test_labels = predict_probabilities(model, test_loader, device)
    if not np.array_equal(test_labels, y_test):
        raise RuntimeError("시퀀스 test 라벨 순서가 정형 test 라벨과 다릅니다.")

    lines = [
        "GRU churn prediction, original v1 split",
        f"Tabular split/labels source: {FEATURES_PATH}",
        f"Sequence source: {SEQUENCES_PATH}",
        f"Split sizes: train={len(train):,}, valid={len(valid):,}, test={len(test):,}",
        f"Train churn rate: {y_train.mean():.2%}",
        f"Sequence max length: {MAX_SEQUENCE_LENGTH}",
        f"Train-only vocabulary size (including PAD/UNK): {len(vocabulary):,}",
        f"Train-only positive class weight: {positive_weight_value:.4f}",
        f"Best epoch by valid F2: {best_epoch}",
        f"Best valid F2: {best_valid_f2:.4f}",
        f"Selected epoch valid PR-AUC: {best_valid_ap:.4f}",
        f"Default decision threshold (valid F2): {decision_threshold:.6f}",
        "Original data and split assignments were not modified.",
        "\n===== Training history =====",
        *history,
        "\n===== Thresholds selected on valid =====",
    ]
    for name, choice in threshold_choices.items():
        lines.append(
            f"{name}: threshold={choice['threshold']:.6f}, "
            f"precision={choice['precision']:.4f}, recall={choice['recall']:.4f}, "
            f"F1={choice['f1']:.4f}, F2={choice['f2']:.4f}"
        )
    lines.append("\n===== Final test evaluation =====")
    for name, choice in threshold_choices.items():
        lines.append(
            evaluate(test_labels, test_probabilities, choice["threshold"], name)
        )
    report = "\n".join(lines)
    print(report)

    torch.save(
        {
            "state_dict": {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            },
            "vocabulary": vocabulary,
            "max_sequence_length": MAX_SEQUENCE_LENGTH,
            "embedding_dim": 32,
            "hidden_dim": 64,
            "dropout": 0.30,
            "decision_threshold": float(decision_threshold),
            "threshold_selection_metric": "valid F2",
        },
        MODEL_PATH,
    )
    CONFIG_PATH.write_text(
        json.dumps(
            {
                "best_epoch": best_epoch,
                "best_valid_f2": best_valid_f2,
                "best_valid_pr_auc": best_valid_ap,
                "positive_class_weight": positive_weight_value,
                "decision_threshold": decision_threshold,
                "threshold_selection_metric": "valid F2",
                "threshold_choices": threshold_choices,
                "max_sequence_length": MAX_SEQUENCE_LENGTH,
                "vocabulary_size": len(vocabulary),
                "split_sizes": {
                    "train": len(train),
                    "valid": len(valid),
                    "test": len(test),
                },
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    RESULT_PATH.write_text(report + "\n", encoding="utf-8")
    CENTRAL_RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CENTRAL_RESULT_PATH.write_text(report + "\n", encoding="utf-8")
    pd.DataFrame({
        ID_COLUMN: test[ID_COLUMN].astype(str).to_numpy(),
        "churn_true": test_labels,
        "churn_probability": test_probabilities,
    }).to_csv(PREDICTIONS_PATH, index=False, encoding="utf-8-sig")
    print(f"\nModel weights: {MODEL_PATH}")
    print(f"Evaluation report: {RESULT_PATH}")
    print(f"Central result copy: {CENTRAL_RESULT_PATH}")


if __name__ == "__main__":
    main()
