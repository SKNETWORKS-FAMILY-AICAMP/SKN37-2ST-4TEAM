"""gradient boosting 모델 목록과 하이퍼파라미터 탐색 범위.

다섯 모델 모두 '약한 트리를 하나씩 더해 가며 이전 트리의 오차(gradient)를 줄이는' 같은 원리이고,
트리를 만드는 방식과 속도, 규제 방법이 다릅니다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

from .preprocess import make_preprocessor
from .utils import has_module, log


@dataclass
class ModelSpec:
    name: str
    title: str
    package: str
    default: Callable[[int], object]          # random_state -> 기본값 모델
    tuned_base: Callable[[int], object]       # random_state -> 탐색용 모델 (조기 종료 등 고정 설정)
    space: Callable[[float], dict]            # neg/pos 비율 -> 탐색 범위
    why: str
    overfit_controls: str

    def available(self) -> bool:
        return has_module(self.package)


def _hist_default(rs):
    return HistGradientBoostingClassifier(random_state=rs)


def _hist_tuned(rs):
    # 학습셋의 10%를 떼어 내부 조기 종료에 사용 (검증·테스트셋은 쓰지 않음)
    return HistGradientBoostingClassifier(random_state=rs, early_stopping=True, validation_fraction=0.1,
                                          n_iter_no_change=20, scoring="loss")


def _gbm_default(rs):
    return GradientBoostingClassifier(random_state=rs)


def _gbm_tuned(rs):
    return GradientBoostingClassifier(random_state=rs, validation_fraction=0.1, n_iter_no_change=20)


def _lgbm(rs, **kw):
    from lightgbm import LGBMClassifier

    return LGBMClassifier(objective="binary", random_state=rs, n_jobs=1, verbose=-1, **kw)


def _cat(rs, **kw):
    from catboost import CatBoostClassifier

    # allow_writing_files=False: 학습 기록 폴더(catboost_info)를 만들지 않음
    return CatBoostClassifier(random_seed=rs, thread_count=1, verbose=0, allow_writing_files=False, **kw)


def _xgb(rs, **kw):
    from xgboost import XGBClassifier

    return XGBClassifier(random_state=rs, n_jobs=1, tree_method="hist", eval_metric="logloss", **kw)


SPECS: dict[str, ModelSpec] = {
    "hist_gb": ModelSpec(
        name="hist_gb", title="HistGradientBoosting", package="sklearn",
        default=_hist_default, tuned_base=_hist_tuned,
        space=lambda ratio: {
            "learning_rate": loguniform(0.01, 0.3),
            "max_iter": randint(100, 800),
            "max_leaf_nodes": randint(8, 64),
            "max_depth": [None, 3, 5, 8],
            "min_samples_leaf": randint(20, 300),
            "l2_regularization": loguniform(1e-4, 10),
            "class_weight": [None, "balanced"],
        },
        why="값을 구간(히스토그램)으로 묶어 빠르게 학습하는 sklearn 내장 부스팅. 추가 설치가 필요 없음",
        overfit_controls="학습률, 잎 수·깊이, 잎 최소 샘플 수, L2 규제, 내부 조기 종료",
    ),
    "gbm": ModelSpec(
        name="gbm", title="GradientBoosting", package="sklearn",
        default=_gbm_default, tuned_base=_gbm_tuned,
        space=lambda ratio: {
            "n_estimators": randint(100, 600),
            "learning_rate": loguniform(0.01, 0.2),
            "max_depth": randint(2, 6),
            "min_samples_leaf": randint(20, 300),
            "subsample": uniform(0.6, 0.4),
            "max_features": [None, "sqrt", 0.7],
        },
        why="가장 기본적인 gradient boosting. 다른 부스팅 모델의 기준점",
        overfit_controls="학습률, 트리 깊이, 잎 최소 샘플 수, 행 샘플링(subsample), 열 샘플링, 내부 조기 종료",
    ),
    "lightgbm": ModelSpec(
        name="lightgbm", title="LightGBM", package="lightgbm",
        default=lambda rs: _lgbm(rs), tuned_base=lambda rs: _lgbm(rs, subsample_freq=1),
        space=lambda ratio: {
            "n_estimators": randint(100, 800),
            "learning_rate": loguniform(0.01, 0.2),
            "num_leaves": randint(8, 64),
            "max_depth": [-1, 4, 6, 8],
            "min_child_samples": randint(20, 300),
            "subsample": uniform(0.6, 0.4),
            "colsample_bytree": uniform(0.6, 0.4),
            "reg_lambda": loguniform(1e-3, 10),
            "class_weight": [None, "balanced"],
        },
        why="잎 단위로 트리를 키워 빠르고 성능이 좋은 부스팅. 표 데이터에서 많이 쓰임",
        overfit_controls="학습률, 잎 수·깊이, 잎 최소 샘플 수, 행·열 샘플링, L2 규제",
    ),
    "xgboost": ModelSpec(
        name="xgboost", title="XGBoost", package="xgboost",
        default=lambda rs: _xgb(rs), tuned_base=lambda rs: _xgb(rs),
        space=lambda ratio: {
            "n_estimators": randint(100, 800),
            "learning_rate": loguniform(0.01, 0.2),
            "max_depth": randint(2, 8),
            "min_child_weight": loguniform(1, 30),
            "subsample": uniform(0.6, 0.4),
            "colsample_bytree": uniform(0.6, 0.4),
            "reg_lambda": loguniform(1e-2, 10),
            "scale_pos_weight": [1.0, float(ratio)],
        },
        why="규제 항을 목적 함수에 넣은 부스팅. LightGBM과 함께 많이 비교됨",
        overfit_controls="학습률, 트리 깊이, 최소 가중치 합, 행·열 샘플링, L2 규제",
    ),
    "catboost": ModelSpec(
        name="catboost", title="CatBoost", package="catboost",
        default=lambda rs: _cat(rs), tuned_base=lambda rs: _cat(rs),
        space=lambda ratio: {
            "iterations": randint(200, 900),
            "learning_rate": loguniform(0.01, 0.2),
            "depth": randint(3, 8),
            "l2_leaf_reg": loguniform(1, 30),
            "random_strength": loguniform(0.1, 10),
            "bagging_temperature": uniform(0, 1),
            # auto_class_weights=None은 CatBoost가 거부해서(25개 중 13개 실패) xgboost와 같은 방식으로 변경
            "scale_pos_weight": [1.0, float(ratio)],
        },
        why="좌우가 같은 대칭 트리와 순서 기반 부스팅(ordered boosting)으로 과적합을 줄인 부스팅. 기본값 성능이 좋은 편",
        overfit_controls="학습률, 트리 깊이, L2 규제, 무작위성(random_strength, bagging_temperature)",
    ),
}


def pipeline(estimator, numeric, categorical) -> Pipeline:
    return Pipeline([("prep", make_preprocessor(numeric, categorical)), ("model", estimator)])


def selected_specs(names: list[str]) -> list[ModelSpec]:
    out = []
    for n in names:
        spec = SPECS.get(n)
        if spec is None:
            log.info(f"[train] 알 수 없는 모델 이름이라 건너뜁니다: {n}")
        elif not spec.available():
            log.info(f"[train] {spec.title}: {spec.package} 패키지가 없어 건너뜁니다. (pip install {spec.package})")
        else:
            out.append(spec)
    return out
