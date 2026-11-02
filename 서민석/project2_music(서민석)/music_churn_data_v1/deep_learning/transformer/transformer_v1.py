"""공통 학습·평가 코드를 통해 시퀀스 딥러닝 모델을 실행합니다."""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report,
    confusion_matrix, f1_score, fbeta_score, precision_score, recall_score,
    roc_auc_score,
)
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
FEATURES = DATA_DIR / "customer_features_20181101_v1_with_split.csv"
SEQUENCES = DATA_DIR / "customer_sequences_20181101_v1.jsonl"
MAX_LEN, PAD, UNK = 50, 0, 1
SEED, BATCH, EPOCHS, PATIENCE = 42, 128, 50, 8


def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def read_data():
    """v1 정형 파일과 시퀀스 JSONL의 고객·라벨·분할을 맞춰 읽습니다."""
    if not FEATURES.exists() or not SEQUENCES.exists():
        raise FileNotFoundError(f"입력 데이터를 찾을 수 없습니다: {FEATURES}, {SEQUENCES}")
    frame = pd.read_csv(FEATURES, dtype={"user_id": "string"})
    records = {}
    cutoff = int(pd.Timestamp("2018-11-01T00:00:00Z").timestamp() * 1000)
    with SEQUENCES.open(encoding="utf-8") as file:
        for line_num, line in enumerate(file, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            uid = str(row["user_id"])
            if uid in records:
                raise ValueError(f"중복 시퀀스 user_id: {uid}")
            events = row.get("events") or []
            for event in events:
                if event.get("ts") is not None and int(event["ts"]) >= cutoff:
                    raise ValueError(f"관찰 기간 이후 이벤트: JSONL line {line_num}")
            records[uid] = {
                "split": str(row.get("split")), "churn": int(row["churn"]),
                "pages": [str(e.get("page") or "<UNK>") for e in events][-MAX_LEN:],
            }
    if frame["user_id"].duplicated().any() or set(frame.user_id.astype(str)) != set(records):
        raise ValueError("정형 데이터와 시퀀스 데이터의 고객 ID가 일치하지 않습니다.")
    pages = {}
    for row in frame[["user_id", "churn", "split"]].itertuples(index=False):
        uid, churn, split = str(row[0]), int(row[1]), str(row[2])
        rec = records[uid]
        if (rec["churn"], rec["split"]) != (churn, split):
            raise ValueError(f"{uid}의 churn 또는 split이 일치하지 않습니다.")
        pages[uid] = rec["pages"]
    return frame, pages


def encode(ids, pages, vocab):
    values = np.full((len(ids), MAX_LEN), PAD, dtype=np.int64)
    lengths = np.zeros(len(ids), dtype=np.int64)
    for i, uid in enumerate(ids):
        tokens = [vocab.get(page, UNK) for page in pages[uid][-MAX_LEN:]]
        lengths[i] = len(tokens)
        if tokens:
            values[i, :len(tokens)] = tokens
    return values, lengths


class SequenceDataset(Dataset):
    def __init__(self, x, lengths, y):
        self.x = torch.as_tensor(x, dtype=torch.long)
        self.lengths = torch.as_tensor(np.maximum(lengths, 1), dtype=torch.long)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.lengths[index], self.y[index]


class ChurnSequenceModel(nn.Module):
    """RNN, LSTM, Transformer, 또는 이벤트 연결 그래프 모델입니다."""
    def __init__(self, kind, vocab_size, embedding_dim=32, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.kind = kind
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD)
        self.dropout = nn.Dropout(dropout)
        if kind in {"rnn", "lstm"}:
            recurrent = nn.RNN if kind == "rnn" else nn.LSTM
            self.sequence = recurrent(
                embedding_dim, hidden_dim, num_layers=2, batch_first=True,
                dropout=dropout, nonlinearity="tanh" if kind == "rnn" else "tanh",
            ) if kind == "rnn" else recurrent(
                embedding_dim, hidden_dim, num_layers=2, batch_first=True, dropout=dropout
            )
            output_dim = hidden_dim
        elif kind == "transformer":
            self.position = nn.Embedding(MAX_LEN, embedding_dim)
            layer = nn.TransformerEncoderLayer(
                d_model=embedding_dim, nhead=4, dim_feedforward=128,
                dropout=dropout, batch_first=True, norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=2)
            output_dim = embedding_dim
        elif kind == "gnn":
            # 각 이벤트를 그래프 노드로, 시간상 이웃한 이벤트를 양방향 간선으로 둡니다.
            self.graph1 = nn.Linear(embedding_dim, hidden_dim)
            self.graph2 = nn.Linear(hidden_dim, hidden_dim)
            output_dim = hidden_dim
        else:
            raise ValueError(f"지원하지 않는 모델: {kind}")
        self.output = nn.Linear(output_dim, 1)

    def forward(self, tokens, lengths):
        embedded = self.embedding(tokens)
        lengths = lengths.clamp(min=1, max=tokens.shape[1])
        if self.kind in {"rnn", "lstm"}:
            packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
            _, state = self.sequence(packed)
            hidden = state[0] if self.kind == "lstm" else state
            pooled = hidden[-1]
        elif self.kind == "transformer":
            positions = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
            # 빈 시퀀스도 첫 PAD 위치 하나를 유효 위치로 두어 attention이 NaN이 되지 않게 합니다.
            valid = positions < lengths.unsqueeze(1)
            encoded = self.encoder(
                embedded + self.position(positions), src_key_padding_mask=~valid
            )
            pooled = (encoded * valid.unsqueeze(-1)).sum(1) / lengths.unsqueeze(1)
        else:
            batch, steps, _ = embedded.shape
            valid = torch.arange(steps, device=tokens.device).unsqueeze(0) < lengths.unsqueeze(1)
            adjacency = torch.zeros(batch, steps, steps, device=tokens.device)
            adjacency.diagonal(dim1=1, dim2=2).fill_(1.0)
            if steps > 1:
                edge_valid = valid[:, 1:] & valid[:, :-1]
                idx = torch.arange(steps - 1, device=tokens.device)
                adjacency[:, idx, idx + 1] = edge_valid.float()
                adjacency[:, idx + 1, idx] = edge_valid.float()
            degree = adjacency.sum(-1).clamp(min=1.0)
            normalized = adjacency / degree.unsqueeze(-1)
            node_state = torch.relu(self.graph1(torch.bmm(normalized, embedded)))
            node_state = torch.relu(self.graph2(torch.bmm(normalized, node_state)))
            node_state = node_state * valid.unsqueeze(-1)
            pooled = node_state.sum(1) / lengths.unsqueeze(1)
        return self.output(self.dropout(pooled)).squeeze(-1)


def probabilities(model, loader, device):
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for x, lengths, y in loader:
            probs.extend(torch.sigmoid(model(x.to(device), lengths.to(device))).cpu().numpy())
            labels.extend(y.numpy())
    return np.asarray(probs), np.asarray(labels, dtype=np.int64)


def threshold_options(y, probs):
    rows = []
    for threshold in np.unique(np.r_[0.0, probs, 0.5, 1.0]):
        pred = (probs >= threshold).astype(int)
        rows.append({
            "threshold": float(threshold),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "f2": float(fbeta_score(y, pred, beta=2, zero_division=0)),
        })
    choices = {"F2 maximum (recall-oriented)": max(rows, key=lambda r: r["f2"]),
               "F1 maximum": max(rows, key=lambda r: r["f1"])}
    for target in (0.70, 0.80):
        eligible = [r for r in rows if r["recall"] >= target]
        if eligible:
            choices[f"Recall at least {target:.0%}"] = max(
                eligible, key=lambda r: (r["precision"], r["threshold"])
            )
    return choices


def test_report(y, probs, threshold, title):
    pred = (probs >= threshold).astype(int)
    return "\n".join([
        f"===== {title}, threshold={threshold:.6f} =====",
        f"Accuracy: {accuracy_score(y, pred):.4f}",
        f"Precision: {precision_score(y, pred, zero_division=0):.4f}",
        f"Recall: {recall_score(y, pred, zero_division=0):.4f}",
        f"F1-score: {f1_score(y, pred, zero_division=0):.4f}",
        f"F2-score: {fbeta_score(y, pred, beta=2, zero_division=0):.4f}",
        f"ROC-AUC: {roc_auc_score(y, probs):.4f}",
        f"PR-AUC: {average_precision_score(y, probs):.4f}",
        "Classification report:\n" + classification_report(y, pred, zero_division=0),
        "Confusion matrix (rows=true, columns=predicted):\n" + str(confusion_matrix(y, pred)),
    ])


def run_model(kind, model_dir):
    """지정 모델을 공통 split로 학습하고 모델·설정·결과를 저장합니다."""
    seed_everything()
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    frame, pages = read_data()
    splits = {name: frame[frame.split == name].copy() for name in ("train", "valid", "test")}
    if any(part.empty for part in splits.values()):
        raise ValueError("기존 데이터에 train, valid, test split이 모두 필요합니다.")
    ids = {name: part.user_id.astype(str).tolist() for name, part in splits.items()}
    train_pages = {page for uid in ids["train"] for page in pages[uid] if page not in {"<PAD>", "<UNK>"}}
    vocab = {"<PAD>": PAD, "<UNK>": UNK}
    vocab.update({page: i + 2 for i, page in enumerate(sorted(train_pages))})
    arrays = {name: encode(ids[name], pages, vocab) for name in ids}
    labels = {name: splits[name].churn.astype(np.int64).to_numpy() for name in splits}
    datasets = {name: SequenceDataset(*arrays[name], labels[name]) for name in arrays}
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=BATCH, shuffle=True,
                            generator=torch.Generator().manual_seed(SEED)),
        "valid": DataLoader(datasets["valid"], batch_size=BATCH, shuffle=False),
        "test": DataLoader(datasets["test"], batch_size=BATCH, shuffle=False),
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChurnSequenceModel(kind, len(vocab)).to(device)
    positive = max(int(labels["train"].sum()), 1)
    pos_weight = (len(labels["train"]) - positive) / positive
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_state, best_f2, best_epoch, patience_left, history = None, -1.0, 0, PATIENCE, []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for x, lengths, y in loaders["train"]:
            x, lengths, y = x.to(device), lengths.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x, lengths), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        valid_probs, valid_y = probabilities(model, loaders["valid"], device)
        choices = threshold_options(valid_y, valid_probs)
        score = choices["F2 maximum (recall-oriented)"]["f2"]
        line = f"epoch={epoch:03d}, train_loss={np.mean(losses):.4f}, valid_F2={score:.4f}"
        print(line)
        history.append(line)
        if score > best_f2:
            best_f2, best_epoch = score, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = PATIENCE
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
    if best_state is None:
        raise RuntimeError("검증 F2 기준 최적 모델이 없습니다.")
    model.load_state_dict(best_state)
    model.to(device)
    valid_probs, valid_y = probabilities(model, loaders["valid"], device)
    choices = threshold_options(valid_y, valid_probs)
    test_probs, test_y = probabilities(model, loaders["test"], device)
    if not np.array_equal(test_y, labels["test"]):
        raise ValueError("test 라벨 순서가 일치하지 않습니다.")
    threshold = choices["F2 maximum (recall-oriented)"]["threshold"]
    prefix = f"{kind}_v1"
    report_lines = [
        f"{kind.upper()} churn prediction, original v1 split",
        f"Features: {FEATURES}", f"Sequences: {SEQUENCES}",
        f"Split sizes: train={len(splits['train']):,}, valid={len(splits['valid']):,}, test={len(splits['test']):,}",
        f"Train churn rate: {labels['train'].mean():.2%}",
        f"Max sequence length: {MAX_LEN}; train-only vocabulary size: {len(vocab)}",
        f"Positive class weight: {pos_weight:.4f}", f"Best epoch by valid F2: {best_epoch}",
        f"Best valid F2: {best_f2:.4f}", f"Selected threshold by valid F2: {threshold:.6f}",
        "Test data was used only for final evaluation.", "\n===== Training history =====", *history,
        "\n===== Thresholds selected on valid =====",
    ]
    for name, row in choices.items():
        report_lines.append(f"{name}: threshold={row['threshold']:.6f}, precision={row['precision']:.4f}, recall={row['recall']:.4f}, F1={row['f1']:.4f}, F2={row['f2']:.4f}")
    report_lines.append("\n===== Final test evaluation =====")
    for name, row in choices.items():
        report_lines.append(test_report(test_y, test_probs, row["threshold"], name))
    report = "\n".join(report_lines) + "\n"
    torch.save({"state_dict": best_state, "vocabulary": vocab, "kind": kind,
                "max_sequence_length": MAX_LEN, "decision_threshold": float(threshold),
                "threshold_selection_metric": "valid F2"}, model_dir / f"{prefix}.pt")
    config = {"model": kind, "best_epoch": best_epoch, "best_valid_f2": best_f2,
              "positive_class_weight": pos_weight, "decision_threshold": threshold,
              "threshold_selection_metric": "valid F2", "threshold_choices": choices,
              "max_sequence_length": MAX_LEN, "vocabulary_size": len(vocab),
              "split_sizes": {name: len(part) for name, part in splits.items()}}
    (model_dir / f"{prefix}_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result_path = model_dir / f"{prefix}_evaluation_results.txt"
    result_path.write_text(report, encoding="utf-8")
    central = ROOT / "result" / result_path.name
    central.parent.mkdir(parents=True, exist_ok=True)
    central.write_text(report, encoding="utf-8")
    pd.DataFrame({"user_id": splits["test"].user_id.astype(str).to_numpy(),
                  "churn_true": test_y, "churn_probability": test_probs}).to_csv(
                      model_dir / f"{prefix}_test_predictions.csv", index=False, encoding="utf-8-sig")
    print(f"Model: {model_dir / f'{prefix}.pt'}\nResults: {result_path}\nCentral result: {central}")



if __name__ == '__main__':
    # 현재 폴더에 해당 모델의 체크포인트, 설정, 결과를 저장합니다.
    run_model('transformer', Path(__file__).resolve().parent)
