"""팀 공통 피처 표 읽기와 점검.

- user_id는 문자열로 읽습니다.
- 팀 공통 분할(split 열)을 그대로 씁니다. 다시 나누지 않습니다.
- 값은 바꾸지 않습니다 (0인 행동 횟수, 음수 song_count_change는 정보로 유지).
- 파일목록_및_해시.json이 있으면 sha256을 비교해 어떤 데이터 버전으로 학습했는지 기록합니다.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .utils import log, read_json, sha256

SPLITS = ("train", "valid", "test")


@dataclass
class Dataset:
    df: pd.DataFrame
    numeric: list[str]
    categorical: list[str]
    target: str
    id_col: str
    info: dict

    @property
    def features(self) -> list[str]:
        return self.numeric + self.categorical

    def X(self, split: str) -> pd.DataFrame:
        return self.df.loc[self.df["split"] == split, self.features].reset_index(drop=True)

    def y(self, split: str) -> np.ndarray:
        return self.df.loc[self.df["split"] == split, self.target].to_numpy().astype(int)

    def ids(self, split: str) -> np.ndarray:
        return self.df.loc[self.df["split"] == split, self.id_col].to_numpy(dtype=str)


def load_dataset(cfg: Config) -> Dataset:
    d = cfg["data"]
    path = cfg.paths.data
    if not path.exists():
        raise FileNotFoundError(f"데이터 파일이 없습니다: {path}\n"
                                "configs/config.yaml의 paths.data 경로를 확인하세요.")
    df = pd.read_csv(path, dtype={d["id_col"]: str}, encoding="utf-8-sig")
    num, cat = list(d["numeric"]), list(d["categorical"])
    need = [d["id_col"], d["target"], d["split_col"], *num, *cat]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"데이터에 필요한 열이 없습니다: {missing}")
    df = df.rename(columns={d["split_col"]: "split"})

    # 점검: 분할 값, 정답 값, ID 중복, 결측
    bad_split = set(df["split"].unique()) - set(SPLITS)
    if bad_split:
        raise ValueError(f"split 열에 예상하지 못한 값이 있습니다: {bad_split}")
    if not set(df[d["target"]].unique()) <= {0, 1}:
        raise ValueError("정답 열은 0 또는 1이어야 합니다.")
    if df[d["id_col"]].duplicated().any():
        raise ValueError("user_id가 중복된 행이 있습니다.")
    for c in num:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    for c in cat:
        df[c] = pd.Series([np.nan if pd.isna(v) else str(v) for v in df[c].tolist()], index=df.index, dtype=object)

    info = {
        "file": path.name,
        "sha256": sha256(path),
        "n_rows": int(len(df)),
        "n_features": len(num) + len(cat),
        "missing_cells": int(df[num + cat].isna().sum().sum()),
        "split": {s: {"n": int((df["split"] == s).sum()), "churn": int(df.loc[df["split"] == s, d["target"]].sum()),
                      "rate": float(df.loc[df["split"] == s, d["target"]].mean())} for s in SPLITS},
        "churn_rate": float(df[d["target"]].mean()),
    }
    info["hash_check"] = _check_hash(cfg.paths.hash_list, path, info["sha256"])
    log.info(f"[data] {path.name}: {len(df):,}명, 이탈률 {info['churn_rate']:.2%}, "
             f"분할 { {s: v['n'] for s, v in info['split'].items()} }, 해시 {info['hash_check']}")
    return Dataset(df=df, numeric=num, categorical=cat, target=d["target"], id_col=d["id_col"], info=info)


def _check_hash(hash_list: Path, path: Path, digest: str) -> str:
    if not hash_list or not Path(hash_list).is_file():
        return "목록 없음"
    try:
        items = read_json(hash_list)
    except Exception:
        return "목록 읽기 실패"
    for it in items:
        if Path(str(it.get("파일", ""))).name == path.name:
            return "일치" if it.get("sha256") == digest else "불일치 (데이터가 바뀌었을 수 있음)"
    return "목록에 파일 없음"
