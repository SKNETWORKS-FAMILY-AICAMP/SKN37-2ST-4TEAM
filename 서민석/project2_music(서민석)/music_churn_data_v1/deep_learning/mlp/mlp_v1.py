"""원본 v1 공통 분할의 정형 피처로 MLP 이탈 분류기를 학습합니다."""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset


SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR.parents[1]
DATA_PATH = DATASET_DIR / "data" / "processed" / "customer_features_20181101_v1_with_split.csv"
MODEL_PATH = SCRIPT_DIR / "mlp_v1.pt"
PREPROCESSOR_PATH = SCRIPT_DIR / "mlp_v1_preprocessor.joblib"
CONFIG_PATH = SCRIPT_DIR / "mlp_v1_config.json"
RESULT_PATH = SCRIPT_DIR / "mlp_v1_evaluation_results.txt"
CENTRAL_RESULT_PATH = DATASET_DIR / "result" / RESULT_PATH.name
PREDICTIONS_PATH = SCRIPT_DIR / "mlp_v1_test_predictions.csv"

TARGET = "churn"
SPLIT_COLUMN = "split"
ID_COLUMN = "user_id"
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
RANDOM_STATE = 42
BATCH_SIZE = 256
MAX_EPOCHS = 100
PATIENCE = 10
RECALL_TARGETS = [0.70, 0.80]


def set_seeds(seed: int = RANDOM_STATE) -> None:
    """실험 결과를 반복할 수 있도록 난수 시드를 고정합니다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ChurnMLP(nn.Module):
    """정형 피처를 입력받는 다층 퍼셉트론 이진 분류기입니다."""

    def __init__(self, input_dim: int, dropout: float = 0.30):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs).squeeze(-1)


def get_probabilities(
    model: ChurnMLP,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """DataLoader의 이탈 확률과 실제 라벨을 반환합니다."""
    model.eval()
    probabilities = []
    labels_all = []
    with torch.no_grad():
        for inputs, labels in loader:
            logits = model(inputs.to(device))
            probabilities.extend(torch.sigmoid(logits).cpu().numpy().tolist())
            labels_all.extend(labels.numpy().tolist())
    return np.asarray(probabilities), np.asarray(labels_all, dtype=np.int64)


def choose_thresholds(y_true: np.ndarray, probabilities: np.ndarray) -> dict:
    """valid에서 F2/F1 및 재현율 목표별 임계값을 선택합니다."""
    thresholds = np.unique(np.concatenate(([0.0, 0.5, 1.0], probabilities)))
    rows = []
    for threshold in thresholds:
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
    name: str,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> str:
    """한 임계값의 성능 지표와 혼동 행렬을 텍스트로 반환합니다."""
    predictions = (probabilities >= threshold).astype(int)
    return "\n".join([
        f"===== {name} | threshold={threshold:.6f} | n={len(y_true):,} =====",
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
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"정형 데이터 파일이 없습니다: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, dtype={ID_COLUMN: "string"})
    required = [ID_COLUMN, TARGET, SPLIT_COLUMN] + FEATURE_COLUMNS
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"CSV에 필요한 컬럼이 없습니다: {missing}")

    train = df[df[SPLIT_COLUMN] == "train"].copy()
    valid = df[df[SPLIT_COLUMN] == "valid"].copy()
    test = df[df[SPLIT_COLUMN] == "test"].copy()
    if any(frame.empty for frame in (train, valid, test)):
        raise ValueError("train, valid, test 분할이 모두 필요합니다.")

    # 결측 대체와 정규화는 train에만 맞추고, valid/test에는 같은 변환을 적용합니다.
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])
    x_train = preprocessor.fit_transform(train[FEATURE_COLUMNS]).astype(np.float32)
    x_valid = preprocessor.transform(valid[FEATURE_COLUMNS]).astype(np.float32)
    x_test = preprocessor.transform(test[FEATURE_COLUMNS]).astype(np.float32)
    y_train = train[TARGET].astype(np.int64).to_numpy()
    y_valid = valid[TARGET].astype(np.int64).to_numpy()
    y_test = test[TARGET].astype(np.int64).to_numpy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChurnMLP(input_dim=x_train.shape[1]).to(device)
    train_dataset = TensorDataset(
        torch.from_numpy(x_train), torch.from_numpy(y_train.astype(np.float32))
    )
    valid_dataset = TensorDataset(
        torch.from_numpy(x_valid), torch.from_numpy(y_valid.astype(np.float32))
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(RANDOM_STATE),
    )
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    positives = max(int(y_train.sum()), 1)
    negatives = int(len(y_train) - positives)
    pos_weight = torch.tensor(negatives / positives, dtype=torch.float32, device=device)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_state = None
    best_valid_f2 = -1.0
    best_valid_ap = -1.0
    best_epoch = 0
    patience_left = PATIENCE
    epoch_lines = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        losses = []
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = loss_function(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        valid_probabilities, valid_labels = get_probabilities(model, valid_loader, device)
        valid_ap = float(average_precision_score(valid_labels, valid_probabilities))
        epoch_thresholds = choose_thresholds(valid_labels, valid_probabilities)
        valid_f2 = epoch_thresholds["F2 최대 (재현율 우선)"]["f2"]
        line = (
            f"epoch={epoch:03d}, train_loss={np.mean(losses):.4f}, "
            f"valid_F2={valid_f2:.4f}, valid_PR-AUC={valid_ap:.4f}"
        )
        epoch_lines.append(line)
        print(line)

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
        raise RuntimeError("MLP 학습에서 최적 모델 상태를 저장하지 못했습니다.")
    model.load_state_dict(best_state)
    model.to(device)
    valid_probabilities, valid_labels = get_probabilities(model, valid_loader, device)
    thresholds = choose_thresholds(valid_labels, valid_probabilities)

    # 임계값이 확정된 뒤 test를 한 번 예측하여 최종 평가합니다.
    test_dataset = TensorDataset(
        torch.from_numpy(x_test), torch.from_numpy(y_test.astype(np.float32))
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_probabilities, test_labels = get_probabilities(model, test_loader, device)

    report_lines = [
        "MLP churn prediction, original v1 split",
        f"Data: {DATA_PATH}",
        f"Split sizes: train={len(train):,}, valid={len(valid):,}, test={len(test):,}",
        f"Train churn rate: {y_train.mean():.2%}",
        f"Input features: {len(FEATURE_COLUMNS)} raw fields, {x_train.shape[1]} encoded dimensions",
        f"Train-only positive class weight: {negatives / positives:.4f}",
        f"Best epoch by valid F2: {best_epoch}",
        f"Best valid F2: {best_valid_f2:.4f}",
        f"Selected epoch valid PR-AUC: {best_valid_ap:.4f}",
        "Original dataset and split assignments were not modified.",
        "\n===== Training history =====",
        *epoch_lines,
        "\n===== Thresholds selected on valid =====",
    ]
    for name, choice in thresholds.items():
        report_lines.append(
            f"{name}: threshold={choice['threshold']:.6f}, "
            f"precision={choice['precision']:.4f}, recall={choice['recall']:.4f}, "
            f"F1={choice['f1']:.4f}, F2={choice['f2']:.4f}"
        )
    decision_threshold = thresholds["F2 최대 (재현율 우선)"]["threshold"]
    report_lines.append(
        f"Recommended operating threshold (valid F2): {decision_threshold:.6f}"
    )
    report_lines.append("\n===== Final test evaluation =====")
    for name, choice in thresholds.items():
        report_lines.append(
            evaluate(name, test_labels, test_probabilities, choice["threshold"])
        )
    report = "\n".join(report_lines)
    print(report)

    # 모델 가중치, train-only 전처리기, 임계값을 함께 저장합니다.
    torch.save(
        {
            "state_dict": {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            },
            "input_dim": int(x_train.shape[1]),
            "hidden_dims": [64, 32],
            "dropout": 0.30,
            "decision_threshold": float(decision_threshold),
            "threshold_selection_metric": "valid F2",
        },
        MODEL_PATH,
    )
    joblib.dump(
        {
            "preprocessor": preprocessor,
            "feature_columns": FEATURE_COLUMNS,
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
        },
        PREPROCESSOR_PATH,
    )
    CONFIG_PATH.write_text(
        json.dumps(
            {
                "best_epoch": best_epoch,
                "best_valid_f2": best_valid_f2,
                "best_valid_pr_auc": best_valid_ap,
                "positive_class_weight": negatives / positives,
                "threshold_choices": thresholds,
                "decision_threshold": float(decision_threshold),
                "threshold_selection_metric": "valid F2",
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
    print(f"Preprocessor: {PREPROCESSOR_PATH}")
    print(f"Evaluation report: {RESULT_PATH}")
    print(f"Central result copy: {CENTRAL_RESULT_PATH}")
    print(f"Test predictions: {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
