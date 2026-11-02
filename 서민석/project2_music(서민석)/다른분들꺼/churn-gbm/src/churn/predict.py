"""저장된 최종 모델로 새 고객 예측.

    python -m churn predict --input 새고객.csv --output 예측결과.csv

입력 CSV에는 model_card.json의 input_columns(피처 14개)가 있어야 합니다. user_id가 있으면 결과에 함께 남깁니다.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import Config
from .utils import log, read_json


def load_model(cfg: Config):
    card = read_json(cfg.paths.models_dir / "model_card.json")
    model = joblib.load(cfg.paths.models_dir / "best_model.joblib")
    return model, card


def predict_frame(model, card: dict, df: pd.DataFrame, threshold: float | None = None) -> pd.DataFrame:
    cols = card["input_columns"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"입력에 필요한 열이 없습니다: {missing}")
    X = df[cols].copy()
    cats = set(card["preprocessing"]["categories"])
    for c in cols:
        if c in cats:
            X[c] = pd.Series([np.nan if pd.isna(v) else str(v) for v in X[c].tolist()], index=X.index, dtype=object)
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce").astype(float)
    thr = card["threshold"] if threshold is None else threshold
    out = pd.DataFrame(index=df.index)
    if "user_id" in df.columns:
        out["user_id"] = df["user_id"].astype(str)
    out["churn_prob"] = model.predict_proba(X)[:, 1]
    out["churn_pred"] = (out["churn_prob"] >= thr).astype(int)
    out["rank"] = out["churn_prob"].rank(ascending=False, method="first").astype(int)
    return out


def run_predict(cfg: Config, input_path: str, output_path: str | None) -> Path:
    model, card = load_model(cfg)
    df = pd.read_csv(input_path, dtype={"user_id": str}, encoding="utf-8-sig")
    out = predict_frame(model, card, df).sort_values("rank")
    path = Path(output_path) if output_path else Path(input_path).with_name(Path(input_path).stem + "_예측.csv")
    out.to_csv(path, index=False, encoding="utf-8-sig")
    log.info(f"[predict] {len(out):,}명 예측 → {path} (임계값 {card['threshold']:.3f}, "
             f"이탈 예상 {int(out['churn_pred'].sum()):,}명)")
    return path
