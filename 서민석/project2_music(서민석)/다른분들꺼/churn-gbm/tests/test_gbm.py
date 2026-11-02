"""pytest 테스트: 데이터 점검, 전처리 기준, 분할 유지, 전체 실행(작은 설정), 예측 함수.

실행: python -m pytest -q   (VS Code 테스트 탭에서도 실행 가능)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from churn.config import DEFAULT_CONFIG, load_config
from churn.data import load_dataset
from churn.models import pipeline
from churn.preprocess import describe_preprocessor, make_preprocessor


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("gbm")
    return load_config(DEFAULT_CONFIG, {
        "paths": {"models_dir": str(tmp / "models"), "reports_dir": str(tmp / "reports")},
        "models": ["hist_gb"],
        "tuning": {"n_iter": 2, "cv_folds": 2, "n_jobs": 1},
        "evaluation": {"learning_curve": False},
        "explain": {"n_repeats": 2, "pdp_top": 2, "shap_sample": 200},
    })


@pytest.fixture(scope="module")
def ds(cfg):
    return load_dataset(cfg)


def test_team_split_is_kept(ds):
    """팀 공통 분할과 정답 수가 README_DATA.md와 같아야 함."""
    s = ds.info["split"]
    assert (s["train"]["n"], s["valid"]["n"], s["test"]["n"]) == (12697, 2721, 2721)
    assert (s["train"]["churn"], s["valid"]["churn"], s["test"]["churn"]) == (724, 156, 155)
    assert ds.df["user_id"].is_unique
    assert len(ds.features) == 14 and "churn" not in ds.features and "user_id" not in ds.features


def test_values_not_changed(cfg, ds):
    """음수 이용량 변화와 0 값이 그대로 있어야 함 (팀 기준)."""
    raw = pd.read_csv(cfg.paths.data, dtype={"user_id": str}, encoding="utf-8-sig")
    assert (ds.df["song_count_change"] < 0).sum() == (raw["song_count_change"] < 0).sum()
    assert (ds.df["error_count"] == 0).sum() == (raw["error_count"] == 0).sum()


def test_preprocessor_fit_on_train_only(ds):
    prep = make_preprocessor(ds.numeric, ds.categorical).fit(ds.X("train"))
    info = describe_preprocessor(prep, ds.categorical)
    assert info["feature_order"] == ds.features
    assert info["categories"]["level_last"] == {"free": 0, "paid": 1}
    assert info["numeric_median"]["song_count"] == pytest.approx(ds.X("train")["song_count"].median())


def test_unseen_category_and_missing(ds):
    prep = make_preprocessor(ds.numeric, ds.categorical).fit(ds.X("train"))
    new = ds.X("valid").head(3).copy()
    new.loc[0, "level_last"] = "premium"
    new.loc[1, "song_count"] = np.nan
    out = prep.transform(new)
    assert out.loc[0, "level_last"] == -1
    assert np.isfinite(out.to_numpy()).all()


def test_scaling_does_not_matter_for_trees(ds):
    """트리 모델은 단조 변환(스케일링)에 결과가 같다는 전처리 결정의 근거."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    X, y = ds.X("train").head(3000), ds.y("train")[:3000]
    Xs = X.copy()
    Xs[ds.numeric] = Xs[ds.numeric] * 10 + 5
    a = pipeline(HistGradientBoostingClassifier(random_state=0, early_stopping=False, max_iter=30),
                 ds.numeric, ds.categorical).fit(X, y)
    b = pipeline(HistGradientBoostingClassifier(random_state=0, early_stopping=False, max_iter=30),
                 ds.numeric, ds.categorical).fit(Xs, y)
    assert np.allclose(a.predict_proba(X)[:, 1], b.predict_proba(Xs)[:, 1], atol=1e-6)


def test_end_to_end_and_predict(cfg, ds):
    from churn.eda import run_eda
    from churn.evaluate import run_evaluate
    from churn.explain import run_explain
    from churn.predict import load_model, predict_frame
    from churn.train import run_train

    run_eda(cfg)
    tbl = run_train(cfg)
    assert tbl["model"].tolist() == ["hist_gb"]
    card = run_evaluate(cfg)
    assert 0 < card["threshold"] < 1
    assert card["input_columns"] == ds.features
    assert card["data"]["hash_check"] == "일치"
    run_explain(cfg)
    assert (cfg.paths.reports_dir / "학습_결과서.md").exists()

    model, card = load_model(cfg)
    sample = ds.df[ds.df["split"] == "test"].head(5)
    out = predict_frame(model, card, sample)
    assert out["churn_prob"].between(0, 1).all()
    assert list(out["user_id"]) == list(sample["user_id"])
