"""2단계: 학습과 하이퍼파라미터 튜닝.

데이터 사용 원칙 (팀 공통 분할)
- train: 전처리 기준, 모델 학습, 교차검증 튜닝(RandomizedSearchCV)
- valid: 튜닝 전/후 성능 확인, (evaluate 단계) 임계값·최종 모델 선택
- test : 예측값만 저장. 성능은 evaluate 단계에서 한 번만 계산
"""
from __future__ import annotations

import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import FitFailedWarning
from sklearn.metrics import average_precision_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score

from .config import Config
from .data import SPLITS, load_dataset
from .models import pipeline, selected_specs
from .utils import log, set_seed, write_json


def run_train(cfg: Config) -> pd.DataFrame:
    cfg.paths.ensure()
    t = cfg["tuning"]
    rs = int(t["random_state"])
    set_seed(rs)
    ds = load_dataset(cfg)
    X = {s: ds.X(s) for s in SPLITS}
    y = {s: ds.y(s) for s in SPLITS}
    ratio = float((y["train"] == 0).sum() / max((y["train"] == 1).sum(), 1))
    specs = selected_specs(cfg["models"])
    if not specs:
        raise RuntimeError("학습할 수 있는 모델이 없습니다. configs/config.yaml의 models를 확인하세요.")

    rows, failed = [], []
    for spec in specs:
        try:
            rows.append(_train_one(cfg, spec, ds, X, y, ratio))
        except Exception as e:  # 한 모델이 실패해도 나머지는 계속 (라이브러리 버전 문제 등)
            msg = _root_cause(e)
            log.info(f"[train] {spec.title} 실패 → 건너뜁니다. {msg[:300]}")
            failed.append({"model": spec.name, "error": msg[:500]})
    if not rows:
        raise RuntimeError(f"모든 모델 학습이 실패했습니다: {failed}")

    tbl = pd.DataFrame(rows)
    tbl.to_csv(cfg.paths.reports_dir / "tuning_results.csv", index=False, encoding="utf-8-sig")
    write_json({"models": tbl["model"].tolist(), "failed": failed, "data": ds.info},
               cfg.paths.models_dir / "train_meta.json")
    log.info("[train] 완료\n" + tbl[["model", "default_cv_pr_auc", "cv_valid_pr_auc", "default_valid_pr_auc",
                                     "tuned_valid_pr_auc", "chosen"]]
             .to_string(index=False))
    return tbl


def _train_one(cfg, spec, ds, X, y, ratio) -> dict:
    t = cfg["tuning"]
    rs = int(t["random_state"])
    log.info(f"[train] {spec.title} 학습 시작")
    cv = StratifiedKFold(int(t["cv_folds"]), shuffle=True, random_state=rs)

    # 1) 기본값: 같은 교차검증 폴드로 점수를 내서 튜닝 결과와 공정하게 비교
    t0 = time.time()
    base_pipe = pipeline(spec.default(rs), ds.numeric, ds.categorical)
    base_cv = cross_val_score(base_pipe, X["train"], y["train"], cv=cv, scoring=t["scoring"],
                              n_jobs=int(t["n_jobs"]))
    base = base_pipe.fit(X["train"], y["train"])
    base_valid = _ap(y["valid"], base.predict_proba(X["valid"])[:, 1])
    base_train = _ap(y["train"], base.predict_proba(X["train"])[:, 1])

    # 2) 튜닝: 학습셋 안의 층화 K-겹 교차검증 (검증·테스트셋은 쓰지 않음)
    search = RandomizedSearchCV(
        pipeline(spec.tuned_base(rs), ds.numeric, ds.categorical),
        {f"model__{k}": v for k, v in spec.space(ratio).items()},
        n_iter=int(t["n_iter"]), scoring=t["scoring"], cv=cv,
        n_jobs=int(t["n_jobs"]), random_state=rs, refit=True, return_train_score=True, error_score=np.nan,
    )
    with warnings.catch_warnings(record=True) as caught:  # 일부 조합 실패를 조용히 넘기지 않도록 기록
        warnings.simplefilter("always", FitFailedWarning)
        search.fit(X["train"], y["train"])
    n_failed = int(pd.Series(search.cv_results_["mean_test_score"]).isna().sum())
    fail_reason = ""
    if n_failed:
        msgs = [str(w.message) for w in caught if issubclass(w.category, FitFailedWarning)]
        fail_reason = _root_cause(Exception(msgs[0])) if msgs else "원인 미상"
        log.info(f"[train] 주의: {spec.title} 탐색 {int(t['n_iter'])}개 조합 중 {n_failed}개 실패 → {fail_reason[:200]}")
    tuned = search.best_estimator_
    tuned_valid = _ap(y["valid"], tuned.predict_proba(X["valid"])[:, 1])
    sec = time.time() - t0

    # 기본값 vs 튜닝은 교차검증 점수로 결정 (학습셋 이탈자 약 700명 → 검증셋 약 150명보다 덜 흔들림)
    use_tuned = bool(np.isnan(base_cv.mean()) or search.best_score_ >= base_cv.mean())
    final = tuned if use_tuned else base
    joblib.dump(final, cfg.paths.models_dir / f"{spec.name}.joblib")
    np.savez_compressed(
        cfg.paths.predictions_dir / f"{spec.name}.npz",
        **{f"p_{s}": final.predict_proba(X[s])[:, 1] for s in SPLITS},
        **{f"y_{s}": y[s] for s in SPLITS}, **{f"id_{s}": ds.ids(s) for s in SPLITS},
    )
    res = pd.DataFrame(search.cv_results_)
    res.to_csv(cfg.paths.reports_dir / f"cv_results_{spec.name}.csv", index=False, encoding="utf-8-sig")
    bi = int(search.best_index_)
    log.info(f"[train] {spec.name}: CV PR-AUC 기본 {base_cv.mean():.4f} → 튜닝 {search.best_score_:.4f} "
             f"({'튜닝 채택' if use_tuned else '기본값 유지'}) | 검증 {base_valid:.4f} → {tuned_valid:.4f}, {sec:.0f}초")
    return {
        "model": spec.name, "title": spec.title, "why": spec.why, "overfit_controls": spec.overfit_controls,
        "default_train_pr_auc": base_train, "default_cv_pr_auc": float(base_cv.mean()),
        "default_valid_pr_auc": base_valid,
        "cv_train_pr_auc": float(res.loc[bi, "mean_train_score"]), "cv_valid_pr_auc": float(search.best_score_),
        "cv_std": float(res.loc[bi, "std_test_score"]),
        "tuned_valid_pr_auc": tuned_valid, "chosen": "tuned" if use_tuned else "default",
        "best_params": _fmt(search.best_params_), "n_candidates": int(t["n_iter"]), "sec": round(sec, 1),
        "failed_candidates": n_failed, "fail_reason": fail_reason[:300],
    }


def _root_cause(e: Exception) -> str:
    """교차검증 안에서 난 오류는 메시지 안쪽에 진짜 원인이 있으므로 그 줄을 꺼냄."""
    import re

    text = str(e)
    causes = re.findall(r"^\s*(\w+(?:Error|Exception): .+)$", text, flags=re.M)
    return causes[-1].strip() if causes else f"{type(e).__name__}: {' '.join(text.split())}"


def _ap(y, p) -> float:
    return float(average_precision_score(y, p))


def _fmt(params: dict) -> str:
    out = []
    for k, v in params.items():
        k = k.replace("model__", "")
        out.append(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}")
    return ", ".join(out)
