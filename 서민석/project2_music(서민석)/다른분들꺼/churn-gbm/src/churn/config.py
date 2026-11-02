"""설정 파일 로드와 경로 관리."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"


@dataclass
class Paths:
    root: Path
    data: Path
    hash_list: Path
    models_dir: Path
    reports_dir: Path

    @property
    def figures_dir(self) -> Path:
        return self.reports_dir / "figures"

    @property
    def eda_dir(self) -> Path:
        return self.reports_dir / "eda"

    @property
    def predictions_dir(self) -> Path:
        return self.models_dir / "predictions"

    def ensure(self) -> None:
        for p in [self.models_dir, self.predictions_dir, self.reports_dir, self.figures_dir, self.eda_dir]:
            p.mkdir(parents=True, exist_ok=True)


class Config(dict):
    """dict처럼 쓰되 paths 속성을 함께 제공합니다."""

    def __init__(self, data: dict[str, Any], root: Path):
        super().__init__(copy.deepcopy(data))
        p = self["paths"]

        def resolve(v: str) -> Path:
            path = Path(v)
            return path if path.is_absolute() else (root / path)

        self.paths = Paths(root=root, data=resolve(p["data"]), hash_list=resolve(p.get("hash_list", "")),
                           models_dir=resolve(p["models_dir"]), reports_dir=resolve(p["reports_dir"]))

    @property
    def features(self) -> list[str]:
        return list(self["data"]["numeric"]) + list(self["data"]["categorical"])


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG
    path = path.resolve()
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if overrides:
        data = deep_update(data, overrides)
    return Config(data, path.parent.parent)  # configs/ 의 상위 폴더 = 프로젝트 폴더


def deep_update(base: dict, upd: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out
