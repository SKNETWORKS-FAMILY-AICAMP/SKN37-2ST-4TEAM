"""공통 도구: 로그, 시드, JSON, 한글 그래프 폰트, 마크다운 표."""
from __future__ import annotations

import hashlib
import json
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd


def get_logger() -> logging.Logger:
    logger = logging.getLogger("churn")
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


log = get_logger()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def has_module(name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(name) is not None


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_default)


def read_json(path: Path):
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def setup_matplotlib() -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager, rcParams

    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR", "Noto Sans KR",
                 "Noto Sans CJK JP", "Noto Sans CJK SC"]:
        if name in installed:
            rcParams["font.family"] = name
            break
    rcParams["axes.unicode_minus"] = False
    rcParams["figure.dpi"] = 110
    rcParams["savefig.bbox"] = "tight"


def df_to_markdown(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    """tabulate 없이 마크다운 표 만들기."""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        cells = []
        for v in row.tolist():
            if isinstance(v, (float, np.floating)):
                cells.append("-" if np.isnan(v) else format(v, floatfmt))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
