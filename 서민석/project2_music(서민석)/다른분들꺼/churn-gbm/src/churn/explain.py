"""4단계: 최종 모델 해석.

- 순열 중요도 (검증셋): 피처 하나를 섞었을 때 PR-AUC가 얼마나 떨어지나 → 모델이 실제로 기대는 피처
- 모델 내장 중요도: 트리 분할에 쓰인 정도 (LightGBM·XGBoost·GradientBoosting)
- 부분 의존도(PDP): 피처 값이 커질 때 예측 확률이 평균적으로 어떻게 변하나 → 방향
- SHAP (설치돼 있으면): 고객별·피처별 기여도
주의: 모델이 어떻게 판단했는지에 대한 설명이며, 실제 해지 원인을 증명하지 않습니다.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import PartialDependenceDisplay, partial_dependence, permutation_importance

from .config import Config
from .data import load_dataset
from .eda import DESC
from .utils import df_to_markdown, has_module, log, read_json, setup_matplotlib, write_json


def run_explain(cfg: Config) -> dict:
    setup_matplotlib()
    import matplotlib.pyplot as plt

    ex = cfg["explain"]
    card = read_json(cfg.paths.models_dir / "model_card.json")
    name = card["model"]
    pipe = joblib.load(cfg.paths.models_dir / "best_model.joblib")
    ds = load_dataset(cfg)
    Xv, yv = ds.X("valid"), ds.y("valid")
    rs = int(cfg["tuning"]["random_state"])
    fd = cfg.paths.figures_dir
    log.info(f"[explain] 최종 모델 {name} 해석 중")

    # 1) 순열 중요도
    pi = permutation_importance(pipe, Xv, yv, scoring="average_precision", n_repeats=int(ex["n_repeats"]),
                                random_state=rs, n_jobs=int(cfg["tuning"]["n_jobs"]))
    imp = (pd.DataFrame({"feature": ds.features, "importance": pi.importances_mean, "std": pi.importances_std})
             .sort_values("importance", ascending=False).reset_index(drop=True))
    imp["desc"] = imp["feature"].map(DESC).fillna("")

    # 2) 모델 내장 중요도
    est = pipe.named_steps["model"]
    names = list(pipe.named_steps["prep"].get_feature_names_out())
    native = getattr(est, "feature_importances_", None)
    if native is not None:
        nat = pd.Series(np.asarray(native, dtype=float), index=names)
        imp["native_importance"] = imp["feature"].map(nat / nat.sum())

    # 방향: 다른 피처는 그대로 두고 이 피처만 1%→99% 분위로 바꿀 때 평균 예측 확률 변화
    effect = {}
    for f in ds.numeric:
        pdp = partial_dependence(pipe, Xv, [f], kind="average", grid_resolution=20, percentiles=(0.01, 0.99))
        avg = np.asarray(pdp["average"]).ravel()
        effect[f] = float(avg[-1] - avg[0])
    imp["pdp_effect"] = imp["feature"].map(effect)
    imp["direction"] = imp["pdp_effect"].map(
        lambda v: "" if pd.isna(v) else ("클수록 위험 ↑" if v > 0.01 else "클수록 위험 ↓" if v < -0.01 else "거의 영향 없음"))
    imp.to_csv(cfg.paths.reports_dir / "feature_importance.csv", index=False, encoding="utf-8-sig")

    top = imp[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top["feature"], top["importance"], xerr=top["std"], color="#4C78A8")
    ax.set(xlabel="검증 PR-AUC 감소량", title=f"순열 중요도 ({name})")
    fig.savefig(fd / "permutation_importance.png"); plt.close(fig)

    # 3) 부분 의존도 (상위 수치 피처)
    feats = [f for f in imp["feature"] if f in ds.numeric][: int(ex["pdp_top"])]
    ncol = 3
    nrow = int(np.ceil(len(feats) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 3.4 * nrow), squeeze=False)
    PartialDependenceDisplay.from_estimator(pipe, Xv, feats, ax=axes.ravel()[: len(feats)], kind="average",
                                            grid_resolution=40, percentiles=(0.01, 0.99))
    for ax in axes.ravel()[len(feats):]:
        ax.axis("off")
    fig.suptitle(f"부분 의존도: 피처 값에 따른 평균 예측 확률 ({name}, 검증셋)")
    fig.tight_layout()
    fig.savefig(fd / "partial_dependence.png"); plt.close(fig)

    # 4) SHAP
    shap_ok = False
    if has_module("shap"):
        try:
            import shap

            Xt = pipe.named_steps["prep"].transform(Xv)
            n = min(int(ex["shap_sample"]), len(Xt))
            Xs = Xt.sample(n, random_state=rs)
            explainer = shap.TreeExplainer(est)
            sv = explainer.shap_values(Xs)
            if isinstance(sv, list):
                sv = sv[1]
            sv = np.asarray(sv)
            if sv.ndim == 3:
                sv = sv[:, :, 1]
            shap.summary_plot(sv, Xs, show=False, max_display=len(names))
            plt.gcf().savefig(fd / "shap_summary.png", bbox_inches="tight"); plt.close("all")
            pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(sv).mean(axis=0)}) \
              .sort_values("mean_abs_shap", ascending=False) \
              .to_csv(cfg.paths.reports_dir / "shap_importance.csv", index=False, encoding="utf-8-sig")
            shap_ok = True
        except Exception as e:  # 모델·버전에 따라 지원 안 될 수 있음
            log.info(f"[explain] SHAP 생략 ({type(e).__name__}: {e})")

    write_json({"model": name, "shap": shap_ok}, cfg.paths.reports_dir / "explain_meta.json")
    _append_report(cfg, name, imp, shap_ok)
    log.info(f"[explain] 완료: 중요도 1위 {imp.loc[0, 'feature']}, SHAP {'사용' if shap_ok else '미사용'}")
    return {"model": name, "top": imp["feature"].head(5).tolist(), "shap": shap_ok}


def _append_report(cfg, name, imp, shap_ok):
    f = cfg.paths.reports_dir / "학습_결과서.md"
    text = f.read_text(encoding="utf-8") if f.exists() else "# 학습 결과서\n"
    text = text.split("\n## 10. 모델 해석")[0].rstrip()
    has_native = "native_importance" in imp
    cols = ["feature", "importance", "std"] + (["native_importance"] if has_native else []) + ["direction", "desc"]
    top3 = ", ".join(f"`{x}`" for x in imp["feature"].head(3))
    L = ["", "## 10. 모델 해석", f"- 대상: 최종 모델 `{name}`",
         "- 순열 중요도: 검증셋에서 피처 하나를 무작위로 섞었을 때 PR-AUC가 떨어지는 정도 (클수록 모델이 많이 의존)",
         *(["- native_importance: 모델 내장 중요도 (합이 1이 되도록 정규화)"] if has_native else []),
         "- direction: 이 피처만 하위 1% → 상위 99% 값으로 바꿨을 때 평균 예측 확률이 오르는지 내리는지 (부분 의존도)", "",
         df_to_markdown(imp[cols]), "",
         f"- 모델은 {top3} 순으로 크게 의존합니다.",
         "- EDA의 단변량 AUC(피처 하나만 볼 때)와 direction(다른 피처를 고정할 때)이 다를 수 있습니다. "
         "예를 들어 활동량이 같은 고객끼리 비교하면 좋아요가 많을수록 위험이 내려가는 식입니다.", "",
         "![pi](figures/permutation_importance.png)", "", "![pdp](figures/partial_dependence.png)", ""]
    if shap_ok:
        L += ["- SHAP: 점 하나가 고객 한 명. 오른쪽일수록 이탈 확률을 올린 쪽, 색이 붉을수록 피처 값이 큼", "",
              "![shap](figures/shap_summary.png)", ""]
    L += ["- 주의: 모델이 어떻게 판단했는지에 대한 설명이며 실제 해지 원인의 증명이 아닙니다.",
          "- 같은 기준일(2018-11-01)의 다른 고객으로 평가했으므로, 다른 달에서도 같은 성능을 보장하지 않습니다.", ""]
    f.write_text(text + "\n" + "\n".join(L), encoding="utf-8")
