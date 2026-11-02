"""3단계: 평가, 임계값, 최종 모델 선택, 학습 결과서.

순서
1) 모델마다 검증셋에서 임계값을 정함 (F-beta 최대 또는 목표 재현율)
2) 학습-검증 차이, 트리 개수별 곡선, 학습 곡선으로 과적합 점검
3) 검증 PR-AUC가 가장 높은 모델을 최종 모델로 선택
4) 테스트셋 성능은 여기서 처음이자 마지막으로 계산
"""
from __future__ import annotations

import shutil
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, fbeta_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve)

from .config import Config
from .data import SPLITS, load_dataset
from .models import SPECS
from .preprocess import describe_preprocessor
from .utils import df_to_markdown, has_module, log, read_json, setup_matplotlib, write_json


# ---------------------------------------------------------------
# 지표
# ---------------------------------------------------------------
def choose_threshold(y, p, strategy="max_fbeta", beta=2.0, target_recall=0.7) -> float:
    prec, rec, thr = precision_recall_curve(y, p)
    prec, rec = prec[:-1], rec[:-1]
    if strategy == "target_recall":
        ok = np.where(rec >= target_recall)[0]
        return float(thr[ok[-1]]) if len(ok) else float(thr[0])
    b2 = beta ** 2
    f = (1 + b2) * prec * rec / np.maximum(b2 * prec + rec, 1e-12)
    return float(thr[int(np.nanargmax(f))])


def top_k_capture(y, p, k: float) -> float:
    n = max(int(round(len(p) * k)), 1)
    top = np.argsort(-p, kind="mergesort")[:n]
    return float(y[top].sum() / max(y.sum(), 1))


def bootstrap_ci(y, p, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    """PR-AUC의 95% 부트스트랩 구간. 이탈자가 적은 셋에서 점수가 얼마나 흔들리는지 보여 줌."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    vals = []
    for _ in range(n_boot):
        b = rng.choice(idx, len(idx), replace=True)
        if y[b].sum() == 0:
            continue
        vals.append(average_precision_score(y[b], p[b]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def threshold_metrics(y, p, thr, beta) -> dict:
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(thr),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(fbeta_score(y, pred, beta=1, zero_division=0)),
        f"f{beta:g}": float(fbeta_score(y, pred, beta=beta, zero_division=0)),
        "flagged": int(tp + fp), "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


# ---------------------------------------------------------------
# 진입점
# ---------------------------------------------------------------
def run_evaluate(cfg: Config) -> dict:
    setup_matplotlib()
    cfg.paths.ensure()
    ev = cfg["evaluation"]
    beta, ks = float(ev["beta"]), [float(k) for k in ev["top_k_pct"]]
    meta = read_json(cfg.paths.models_dir / "train_meta.json")
    preds = {}
    for m in meta["models"]:
        f = cfg.paths.predictions_dir / f"{m}.npz"
        if f.exists():
            z = np.load(f, allow_pickle=False)
            preds[m] = {k: z[k] for k in z.files}
    if not preds:
        raise RuntimeError("예측 파일이 없습니다. train 단계를 먼저 실행하세요.")

    rows, thr = [], {}
    for m, d in preds.items():
        t = choose_threshold(d["y_valid"], d["p_valid"], ev["threshold_strategy"], beta, float(ev["target_recall"]))
        thr[m] = t
        r = {"model": m}
        for s in SPLITS:
            y, p = d[f"y_{s}"], d[f"p_{s}"]
            r[f"{s}_pr_auc"] = float(average_precision_score(y, p))
            r[f"{s}_roc_auc"] = float(roc_auc_score(y, p))
            r[f"{s}_brier"] = float(brier_score_loss(y, p))
        for s in ("valid", "test"):
            y, p = d[f"y_{s}"], d[f"p_{s}"]
            r[f"{s}_pr_auc_lo"], r[f"{s}_pr_auc_hi"] = bootstrap_ci(y, p, seed=int(cfg["tuning"]["random_state"]))
            for k, v in threshold_metrics(y, p, t, beta).items():
                r[f"{s}_{k}"] = v
            for k in ks:
                r[f"{s}_top{k * 100:g}_capture"] = top_k_capture(y, p, k)
        rows.append(r)
    res = pd.DataFrame(rows).sort_values(f"valid_{ev['selection_metric']}", ascending=False).reset_index(drop=True)
    res.to_csv(cfg.paths.reports_dir / "evaluation_all.csv", index=False, encoding="utf-8-sig")
    best = res.loc[0, "model"]

    gap = pd.DataFrame({
        "model": res["model"], "train_pr_auc": res["train_pr_auc"], "valid_pr_auc": res["valid_pr_auc"],
        "gap_pr_auc": res["train_pr_auc"] - res["valid_pr_auc"],
        "train_roc_auc": res["train_roc_auc"], "valid_roc_auc": res["valid_roc_auc"],
        "gap_roc_auc": res["train_roc_auc"] - res["valid_roc_auc"],
    })
    gap["status"] = np.where(gap["gap_pr_auc"] > float(ev["overfit_gap_warn"]), "주의 (과적합 경향)", "양호")
    gap.to_csv(cfg.paths.reports_dir / "overfitting_check.csv", index=False, encoding="utf-8-sig")

    # 최종 모델 저장 + 모델 카드 (전처리 기준, 피처 순서, 설정을 모델과 함께)
    ds = load_dataset(cfg)
    model = joblib.load(cfg.paths.models_dir / f"{best}.joblib")
    shutil.copyfile(cfg.paths.models_dir / f"{best}.joblib", cfg.paths.models_dir / "best_model.joblib")
    b = res.loc[0]
    card = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": best, "title": SPECS[best].title,
        "params": {k: _jsonable(v) for k, v in model.named_steps["model"].get_params().items()},
        "selection": f"검증셋 {ev['selection_metric']} 최대",
        "threshold": thr[best], "threshold_strategy": ev["threshold_strategy"], "beta": beta,
        "preprocessing": describe_preprocessor(model.named_steps["prep"], ds.categorical),
        "input_columns": ds.features, "target": ds.target,
        "data": ds.info,
        "valid": {c[6:]: _jsonable(b[c]) for c in res.columns if c.startswith("valid_")},
        "test": {c[5:]: _jsonable(b[c]) for c in res.columns if c.startswith("test_")},
        "versions": _versions(),
    }
    write_json(card, cfg.paths.models_dir / "model_card.json")
    _write_predictions(cfg, ds, preds, best, thr[best])

    _figures(cfg, ds, preds, res, gap, best, thr[best], beta)
    _report(cfg, ds, res, gap, card, ks)
    log.info(f"[evaluate] 최종 모델 {best} | 검증 PR-AUC {b['valid_pr_auc']:.4f} | 테스트 PR-AUC {b['test_pr_auc']:.4f}, "
             f"ROC-AUC {b['test_roc_auc']:.4f}, 재현율 {b['test_recall']:.3f}, 정밀도 {b['test_precision']:.3f} "
             f"(임계값 {thr[best]:.3f})")
    return card


def _jsonable(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if np.isnan(v) else float(v)
    if v is None or isinstance(v, (int, str, bool, list, dict)):
        return v
    return str(v)


def _versions() -> dict:
    import sklearn

    out = {"scikit-learn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__}
    for m in ("lightgbm", "xgboost", "catboost"):
        if has_module(m):
            out[m] = __import__(m).__version__
    return out


def _write_predictions(cfg, ds, preds, best, thr):
    frames = []
    for s in SPLITS:
        f = pd.DataFrame({"user_id": preds[best][f"id_{s}"], "split": s, "churn": preds[best][f"y_{s}"]})
        for m, d in preds.items():
            f[f"prob_{m}"] = d[f"p_{s}"]
        frames.append(f)
    out = pd.concat(frames, ignore_index=True)
    out["prob"] = out[f"prob_{best}"]
    out["pred"] = (out["prob"] >= thr).astype(int)
    out.to_csv(cfg.paths.models_dir / "predictions.csv", index=False, encoding="utf-8-sig")


# ---------------------------------------------------------------
# 그림
# ---------------------------------------------------------------
def _figures(cfg, ds, preds, res, gap, best, thr, beta):
    import matplotlib.pyplot as plt

    fd = cfg.paths.figures_dir
    order = res["model"].tolist()

    for s in ("valid", "test"):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
        for m in order:
            y, p = preds[m][f"y_{s}"], preds[m][f"p_{s}"]
            fpr, tpr, _ = roc_curve(y, p)
            pr, rc, _ = precision_recall_curve(y, p)
            lw = 2.4 if m == best else 1.2
            axes[0].plot(fpr, tpr, lw=lw, label=f"{m} ({roc_auc_score(y, p):.3f})")
            axes[1].plot(rc, pr, lw=lw, label=f"{m} ({average_precision_score(y, p):.3f})")
        axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
        axes[0].set(xlabel="거짓 양성 비율", ylabel="재현율", title=f"ROC ({s})")
        axes[1].axhline(preds[order[0]][f"y_{s}"].mean(), color="k", ls="--", lw=0.8)
        axes[1].set(xlabel="재현율", ylabel="정밀도", title=f"Precision-Recall ({s}, 점선=무작위)")
        for ax in axes:
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(fd / f"roc_pr_{s}.png"); plt.close(fig)

    d = preds[best]
    cm = confusion_matrix(d["y_test"], (d["p_test"] >= thr).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12)
    ax.set_xticks([0, 1], ["예측 유지", "예측 이탈"])
    ax.set_yticks([0, 1], ["실제 유지", "실제 이탈"])
    ax.set_title(f"혼동행렬 (테스트, {best}, 임계값 {thr:.3f})")
    fig.savefig(fd / "confusion_test.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(gap)), 3.8))
    x = np.arange(len(gap))
    ax.bar(x - 0.2, gap["train_pr_auc"], 0.4, label="학습")
    ax.bar(x + 0.2, gap["valid_pr_auc"], 0.4, label="검증")
    for i, g in enumerate(gap["gap_pr_auc"]):
        ax.text(i, max(gap["train_pr_auc"].iloc[i], gap["valid_pr_auc"].iloc[i]) + 0.01, f"차이 {g:+.3f}",
                ha="center", fontsize=8)
    ax.set_xticks(x, gap["model"])
    ax.set_ylim(0, min(1.05, gap[["train_pr_auc", "valid_pr_auc"]].values.max() + 0.12))
    ax.set_title("과적합 점검: 학습 vs 검증 PR-AUC")
    ax.legend()
    fig.savefig(fd / "overfit_gap.png"); plt.close(fig)

    tf = cfg.paths.reports_dir / "tuning_results.csv"
    if tf.exists():
        t = pd.read_csv(tf)
        fig, axes = plt.subplots(1, 2, figsize=(max(9, 2.6 * len(t)), 3.8))
        x = np.arange(len(t))
        for ax, (a_col, b_col, title) in zip(axes, [
                ("default_cv_pr_auc", "cv_valid_pr_auc", "교차검증 (학습셋, 채택 기준)"),
                ("default_valid_pr_auc", "tuned_valid_pr_auc", "검증셋")]):
            ax.bar(x - 0.2, t[a_col], 0.4, label="기본값", color="#BAB0AC")
            ax.bar(x + 0.2, t[b_col], 0.4, label="튜닝 후", color="#4C78A8")
            for i, (a, b2) in enumerate(zip(t[a_col], t[b_col])):
                ax.text(i - 0.2, a, f"{a:.3f}", ha="center", va="bottom", fontsize=8)
                ax.text(i + 0.2, b2, f"{b2:.3f}", ha="center", va="bottom", fontsize=8)
            ax.set_xticks(x, t["model"])
            vals = t[[a_col, b_col]].values
            ax.set_ylim(max(0, np.nanmin(vals) - 0.05), np.nanmax(vals) + 0.03)
            ax.set_title(f"PR-AUC: {title}", fontsize=10)
            ax.legend(fontsize=8)
        fig.suptitle("하이퍼파라미터 튜닝 전후")
        fig.tight_layout()
        fig.savefig(fd / "tuning_before_after.png"); plt.close(fig)

    pr, rc, th = precision_recall_curve(d["y_valid"], d["p_valid"])
    b2 = beta ** 2
    f = (1 + b2) * pr[:-1] * rc[:-1] / np.maximum(b2 * pr[:-1] + rc[:-1], 1e-12)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(th, pr[:-1], label="정밀도")
    ax.plot(th, rc[:-1], label="재현율")
    ax.plot(th, f, label=f"F{beta:g}")
    ax.axvline(thr, color="k", ls="--", lw=1, label=f"선택 {thr:.3f}")
    ax.set(xlabel="임계값", title=f"임계값에 따른 지표 (검증, {best})")
    ax.legend(fontsize=8)
    fig.savefig(fd / "threshold_curve.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    for m in order:
        y, p = preds[m]["y_test"], preds[m]["p_test"]
        o = np.argsort(-p, kind="mergesort")
        gain = np.concatenate([[0], np.cumsum(y[o]) / y.sum()])
        ax.plot(np.linspace(0, 1, len(gain)), gain, lw=2.4 if m == best else 1.1, label=m)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="무작위")
    ax.set(xlabel="관리 대상 비율 (확률 상위)", ylabel="잡아낸 이탈자 비율", title="누적 이득 곡선 (테스트)")
    ax.legend(fontsize=8)
    fig.savefig(fd / "cumulative_gain_test.png"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for s, ax in zip(("valid", "test"), axes):
        for m in order:
            y, p = preds[m][f"y_{s}"], preds[m][f"p_{s}"]
            fp_, mp_ = calibration_curve(y, p, n_bins=10, strategy="quantile")
            ax.plot(mp_, fp_, "o-", ms=3, lw=2 if m == best else 1, label=m)
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set(xlabel="예측 확률 (구간 평균)", ylabel="실제 이탈 비율", title=f"확률 보정 ({s})")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fd / "calibration.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.6))
    bins = np.linspace(0, 1, 41)
    ax.hist(d["p_test"][d["y_test"] == 0], bins=bins, alpha=0.6, density=True, label="실제 유지")
    ax.hist(d["p_test"][d["y_test"] == 1], bins=bins, alpha=0.6, density=True, label="실제 이탈")
    ax.axvline(thr, color="k", ls="--", lw=1)
    ax.set(xlabel="예측 이탈 확률", title=f"예측 확률 분포 (테스트, {best})")
    ax.legend()
    fig.savefig(fd / "score_distribution_test.png"); plt.close(fig)

    _iteration_curves(cfg, ds, order, best)
    if cfg["evaluation"].get("learning_curve", True):
        try:
            _learning_curve(cfg, ds, best)
        except Exception as e:
            log.info(f"[evaluate] 학습 곡선 생략: {type(e).__name__}: {e}")


def staged_scores(pipe, X, y, n_points: int = 30) -> pd.DataFrame:
    """트리를 하나씩 더할 때마다 PR-AUC가 어떻게 바뀌는지 (과적합이 시작되는 지점 확인)."""
    Xt = pipe.named_steps["prep"].transform(X)
    est = pipe.named_steps["model"]
    kind = type(est).__name__
    rows = []
    if hasattr(est, "staged_predict_proba"):  # sklearn 부스팅
        probs = list(est.staged_predict_proba(Xt))
        idx = np.unique(np.linspace(1, len(probs), min(n_points, len(probs))).astype(int))
        rows = [(i, average_precision_score(y, probs[i - 1][:, 1])) for i in idx]
    elif kind == "LGBMClassifier":
        n = int(est.booster_.current_iteration())
        for i in np.unique(np.linspace(1, n, min(n_points, n)).astype(int)):
            rows.append((i, average_precision_score(y, est.predict_proba(Xt, num_iteration=int(i))[:, 1])))
    elif kind == "CatBoostClassifier":
        n = int(est.tree_count_)
        for i in np.unique(np.linspace(1, n, min(n_points, n)).astype(int)):
            rows.append((i, average_precision_score(y, est.predict_proba(Xt, ntree_end=int(i))[:, 1])))
    elif kind == "XGBClassifier":
        n = int(est.get_booster().num_boosted_rounds())
        for i in np.unique(np.linspace(1, n, min(n_points, n)).astype(int)):
            rows.append((i, average_precision_score(y, est.predict_proba(Xt, iteration_range=(0, int(i)))[:, 1])))
    return pd.DataFrame(rows, columns=["n_trees", "pr_auc"])


def _iteration_curves(cfg, ds, order, best):
    import matplotlib.pyplot as plt

    n = len(order)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.6), squeeze=False)
    rows = []
    for ax, m in zip(axes[0], order):
        pipe = joblib.load(cfg.paths.models_dir / f"{m}.joblib")
        for s, ls in (("train", "-"), ("valid", "-")):
            try:
                c = staged_scores(pipe, ds.X(s), ds.y(s))
            except Exception as e:  # 라이브러리 버전에 따라 단계별 예측을 지원하지 않을 수 있음
                log.info(f"[evaluate] {m} 트리 개수별 곡선 생략: {type(e).__name__}: {e}")
                c = pd.DataFrame()
            if len(c):
                ax.plot(c["n_trees"], c["pr_auc"], ls, label="학습" if s == "train" else "검증")
                rows += [{"model": m, "split": s, **r} for r in c.to_dict("records")]
        ax.set(xlabel="트리 개수", ylabel="PR-AUC", title=m + (" (최종)" if m == best else ""))
        ax.legend(fontsize=8)
    fig.suptitle("트리 개수에 따른 학습/검증 PR-AUC — 두 선이 벌어지기 시작하면 과적합")
    fig.tight_layout()
    fig.savefig(cfg.paths.figures_dir / "iteration_curves.png"); plt.close(fig)
    pd.DataFrame(rows).to_csv(cfg.paths.reports_dir / "iteration_curves.csv", index=False)


def _learning_curve(cfg, ds, best):
    import matplotlib.pyplot as plt
    from sklearn.model_selection import StratifiedKFold, learning_curve

    t = cfg["tuning"]
    pipe = joblib.load(cfg.paths.models_dir / f"{best}.joblib")
    sizes, tr_s, cv_s = learning_curve(
        pipe, ds.X("train"), ds.y("train"), train_sizes=np.linspace(0.2, 1.0, 5),
        cv=StratifiedKFold(int(t["cv_folds"]), shuffle=True, random_state=int(t["random_state"])),
        scoring="average_precision", n_jobs=int(t["n_jobs"]), shuffle=True, random_state=int(t["random_state"]))
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for sc, lab in [(tr_s, "학습"), (cv_s, "교차검증")]:
        mu, sd = sc.mean(axis=1), sc.std(axis=1)
        ax.plot(sizes, mu, "o-", label=lab)
        ax.fill_between(sizes, mu - sd, mu + sd, alpha=0.15)
    ax.set(xlabel="학습 고객 수", ylabel="PR-AUC", title=f"학습 곡선 ({best})")
    ax.legend()
    fig.savefig(cfg.paths.figures_dir / "learning_curve.png"); plt.close(fig)
    pd.DataFrame({"train_size": sizes, "train_mean": tr_s.mean(1), "cv_mean": cv_s.mean(1), "cv_std": cv_s.std(1)}) \
        .to_csv(cfg.paths.reports_dir / "learning_curve.csv", index=False)


# ---------------------------------------------------------------
# 학습 결과서
# ---------------------------------------------------------------
def _report(cfg, ds, res, gap, card, ks):
    ev = cfg["evaluation"]
    beta = float(ev["beta"])
    info = ds.info
    tune = pd.read_csv(cfg.paths.reports_dir / "tuning_results.csv")
    eda_f = cfg.paths.eda_dir / "eda_summary.json"
    eda = read_json(eda_f) if eda_f.exists() else None
    uni_f = cfg.paths.eda_dir / "univariate_auc.csv"
    uni = pd.read_csv(uni_f) if uni_f.exists() else None
    base = info["split"]["valid"]["rate"]
    k1 = f"{ks[1] * 100:g}" if len(ks) > 1 else f"{ks[0] * 100:g}"

    L = [
        "# 학습 결과서 — gradient boosting",
        "",
        "자동 생성 문서입니다. 숫자는 실행할 때마다 다시 계산됩니다.",
        "",
        "## 1. 데이터와 문제",
        "- 문제: 최근 28일(2018-10-04 ~ 2018-11-01) 이용 행동으로 이후 14일(~ 2018-11-15) 안의 해지 예측",
        "- 정답: 예측 기간에 `Cancellation Confirmation`이 확인되면 1",
        f"- 데이터: `{info['file']}` — {info['n_rows']:,}명, 입력 피처 {info['n_features']}개, "
        f"결측 {info['missing_cells']}칸, 해시 확인 **{info['hash_check']}**",
        "- 분할: 팀 공통 분할을 그대로 사용 (다시 나누지 않음)",
        "",
        df_to_markdown(pd.DataFrame([{"split": s, "고객 수": v["n"], "이탈": v["churn"], "이탈률": f"{v['rate']:.2%}"}
                                     for s, v in info["split"].items()])),
        "",
        "## 2. EDA 요약 (학습셋)",
        f"- 이탈률 {info['split']['train']['rate']:.2%}의 불균형 데이터. 모두 '유지'로 찍어도 정확도 "
        f"{1 - info['split']['train']['rate']:.1%}라 정확도는 지표로 쓰지 않음",
        f"- 주 지표 PR-AUC (무작위 기준값 = 이탈률 약 {base:.3f}), 보조 ROC-AUC, 정밀도·재현율·F{beta:g}, 상위 k% 포착률",
    ]
    if uni is not None:
        L += ["- 피처 하나만으로 본 이탈 관련성 (단변량 AUC 상위 6개):", "",
              df_to_markdown(uni.head(6)[["feature", "auc", "direction", "churn_median", "stay_median"]], ".3f"), ""]
    if eda is not None:
        L.append(f"- 누수 의심 피처 (단변량 AUC ≥ 0.95 또는 ≤ 0.05): {', '.join(eda['leak_suspects']) or '없음'}")
        for c, rates in eda.get("categorical_churn_rate", {}).items():
            L.append(f"- `{c}`별 이탈률: " + ", ".join(f"{k} {v:.1%}" for k, v in rates.items()))
        if uni is not None and (uni.head(6)["direction"] == "높을수록 이탈").mean() > 0.5:
            L.append("- 이탈 고객이 오히려 활동량이 많습니다. 해지 확인 페이지까지 가려면 접속해야 하고, 오래 쓸수록 광고·싫어요 같은 "
                     "불만 경험도 쌓이기 때문으로 보입니다. 접속 없이 조용히 떠나는 고객은 이번 정답 정의(해지 확인)에서는 0으로 남습니다.")
        if eda["high_corr_pairs"]:
            L.append("- 강한 상관(|r| ≥ 0.85): " + ", ".join(f"{a}↔{b} {r:.2f}" for a, b, r in eda["high_corr_pairs"][:5])
                     + " → 트리 모델은 상관에 강해 그대로 사용")
    L += ["", "![dist](eda/dist_by_label.png)", "",
          "## 3. 전처리",
          "| 항목 | 처리 | 이유 |", "|---|---|---|",
          "| 결측 | 수치 중앙값, 범주 최빈값 (학습셋 기준) | 현재 결측 0건. 새 고객 입력에 대비해 파이프라인에 포함 |",
          "| 범주 인코딩 | `level_last` → 정수 (free=0, paid=1), 처음 보는 값은 -1 | 값이 2개라 원-핫과 정보량이 같고, 트리는 정수로 바로 분할 |",
          "| 스케일링 | 하지 않음 | 트리는 값의 순서로만 분할하므로 스케일링해도 결과가 같음 |",
          "| 이상치 | 자르지 않음 | 팀 기준(큰 값만으로 오류라 볼 근거 부족) + 트리는 극단값의 영향이 작음 |",
          "| 0, 음수 | 그대로 | 0은 '행동 없음', 음수 `song_count_change`는 '이용 감소'라는 정보 |",
          "| 불균형 | `class_weight` / `scale_pos_weight`를 튜닝 후보에 포함 | 가중치를 줄지 말지를 교차검증으로 결정 |",
          "",
          "전처리는 sklearn `Pipeline` 안에 넣어 교차검증 폴드마다 학습 폴드로만 다시 계산되고, 저장된 모델 파일에 함께 들어 있습니다.",
          "",
          "## 4. 모델과 하이퍼파라미터 튜닝",
          f"학습셋 안에서 층화 {cfg['tuning']['cv_folds']}-겹 교차검증 + RandomizedSearchCV ({cfg['tuning']['n_iter']}개 조합, "
          "PR-AUC 기준). 기본값도 같은 폴드로 교차검증해서, 튜닝 결과가 더 좋을 때만 채택했습니다.",
          "채택 기준을 검증셋이 아니라 교차검증으로 둔 이유: 학습셋 이탈자(약 700명)가 검증셋(약 150명)보다 많아 점수가 덜 흔들림.",
          ""]
    t = tune[["model", "default_cv_pr_auc", "cv_valid_pr_auc", "cv_std", "default_valid_pr_auc", "tuned_valid_pr_auc",
              "chosen", "sec"]].copy()
    t = t.rename(columns={"cv_valid_pr_auc": "tuned_cv_pr_auc"})
    t["sec"] = t["sec"].map(lambda v: f"{v:.0f}")
    L += [df_to_markdown(t), ""]
    meta = read_json(cfg.paths.models_dir / "train_meta.json")
    skipped = [m for m in cfg["models"] if m not in meta["models"] and m not in [f["model"] for f in meta.get("failed", [])]]
    if skipped:
        L += [f"- 설치되지 않아 건너뛴 모델: {', '.join(skipped)} (`pip install`로 설치 후 다시 실행하면 비교에 포함)"]
    for f in meta.get("failed", []):
        L += [f"- 학습 중 오류로 제외한 모델: `{f['model']}` — {f['error'][:200]}"]
    if "failed_candidates" in tune:
        for _, r in tune[tune["failed_candidates"].fillna(0) > 0].iterrows():
            L += [f"- 주의: `{r['model']}` 탐색 조합 {int(r['n_candidates'])}개 중 {int(r['failed_candidates'])}개가 오류로 "
                  f"제외됨 — {str(r['fail_reason'])[:200]}"]
    L += ["", "**선택된 하이퍼파라미터**", ""]
    L += [f"- `{r['model']}`: {r['best_params']}" for _, r in tune.iterrows()]
    L += ["", "**모델별 특징과 과적합 조절 수단**", ""]
    L += [f"- {r['title']}: {r['why']} / 조절: {r['overfit_controls']}" for _, r in tune.iterrows()]
    L += ["", "![tuning](figures/tuning_before_after.png)", "",
          "## 5. 과적합 점검",
          f"학습 PR-AUC − 검증 PR-AUC가 {ev['overfit_gap_warn']}보다 크면 '주의'로 표시합니다.",
          "", df_to_markdown(gap), "",
          "- 트리 개수별 곡선: 검증 곡선이 꺾이는 지점보다 트리가 훨씬 많으면 과적합. "
          "HistGradientBoosting·GradientBoosting은 학습셋의 10%로 내부 조기 종료를 써서 자동으로 멈춤",
          "- 학습 곡선: 데이터가 늘수록 학습·교차검증 점수 차이가 줄면 데이터 추가가 도움이 되는 상태",
          "", "![gap](figures/overfit_gap.png)", "", "![iter](figures/iteration_curves.png)", ""]
    if (cfg.paths.figures_dir / "learning_curve.png").exists():
        L += ["![lc](figures/learning_curve.png)", ""]

    res = res.copy()
    for s_ in ("valid", "test"):
        res[f"{s_}_pr_auc_95ci"] = [f"{lo:.3f} ~ {hi:.3f}" for lo, hi in zip(res[f"{s_}_pr_auc_lo"], res[f"{s_}_pr_auc_hi"])]
    vcols = ["model", "valid_pr_auc", "valid_pr_auc_95ci", "valid_roc_auc", "valid_brier", "valid_threshold",
             "valid_precision", "valid_recall", f"valid_f{beta:g}", f"valid_top{k1}_capture"]
    width = float((res["valid_pr_auc_hi"] - res["valid_pr_auc_lo"]).mean())
    diff = float(res["valid_pr_auc"].iloc[0] - res["valid_pr_auc"].iloc[1]) if len(res) > 1 else float("nan")
    L += ["## 6. 검증셋 비교와 최종 모델 선택", "",
          df_to_markdown(res[[c for c in vcols if c in res.columns]]), "",
          f"- 최종 모델: **{card['title']} (`{card['model']}`)** — 검증 PR-AUC가 가장 높음",
          *([f"- Brier 점수(확률이 실제 비율과 얼마나 맞는지, 작을수록 좋음)가 {res['valid_brier'].iloc[0]:.3f}로 다른 모델보다 큽니다. "
             "class_weight=balanced로 이탈자에 가중치를 줘서 확률이 위로 부풀려진 것입니다. 순위와 임계값 판단에는 문제없지만, "
             "'이탈 확률 몇 %'를 그대로 말하려면 확률 보정이 필요합니다 (아래 확률 보정 그림)."]
            if res["valid_brier"].iloc[0] > 1.5 * res["valid_brier"].min() else []),
          f"- 검증셋 이탈자가 {info['split']['valid']['churn']}명뿐이라 PR-AUC 95% 구간 폭이 평균 {width:.3f}입니다."
          + (f" 1·2위 차이({diff:.3f})가 이보다 훨씬 작으면 사실상 비슷한 성능으로 보고, 속도·설치 편의도 함께 고려할 수 있습니다."
             if len(res) > 1 else ""),
          "",
          "## 7. 임계값",
          f"- 검증셋에서 {'F' + format(beta, 'g') + '가 최대가 되는 값' if card['threshold_strategy'] == 'max_fbeta' else '목표 재현율을 지키는 가장 높은 값'}"
          f"으로 정함: **{card['threshold']:.3f}**",
          "- 0.5를 쓰지 않는 이유: 이탈률이 5.7%로 낮아 0.5 기준이면 대부분을 놓치고, 클래스 가중치를 쓰면 확률 크기가 달라짐",
          "", "![thr](figures/threshold_curve.png)", "",
          "## 8. 테스트셋 최종 성능 (한 번만 사용)", ""]
    tcols = ["model", "test_pr_auc", "test_pr_auc_95ci", "test_roc_auc", "test_precision", "test_recall", "test_f1", f"test_f{beta:g}"] + \
            [f"test_top{k * 100:g}_capture" for k in ks]
    L += [df_to_markdown(res[[c for c in tcols if c in res.columns]]), ""]
    te = card["test"]
    L += [f"- 최종 모델 테스트: PR-AUC {te['pr_auc']:.3f} (무작위 {info['split']['test']['rate']:.3f}), "
          f"ROC-AUC {te['roc_auc']:.3f}, 정밀도 {te['precision']:.3f}, 재현율 {te['recall']:.3f}",
          f"- 혼동행렬: 이탈자 {te['tp'] + te['fn']:,}명 중 {te['tp']:,}명을 잡고 {te['fn']:,}명을 놓침. "
          f"관리 대상 {te['flagged']:,}명 중 실제 이탈 {te['tp']:,}명"]
    for k in ks:
        cap = te.get(f"top{k * 100:g}_capture")
        if cap is not None:
            L.append(f"- 확률 상위 {k:.0%} 고객만 관리하면 이탈자의 {cap:.1%}를 잡음 (무작위 대비 {cap / k:.1f}배)")
    L += ["- 다른 모델의 테스트 점수는 참고용이며, 선택은 검증셋으로만 했습니다.", "",
          "![roc](figures/roc_pr_test.png)", "", "![cm](figures/confusion_test.png)", "",
          "![gain](figures/cumulative_gain_test.png)", "", "![cal](figures/calibration.png)", "",
          "## 9. 저장한 것",
          "- `models/best_model.joblib`: 최종 모델 (전처리 포함 파이프라인). `predict_proba(DataFrame)`으로 바로 예측",
          "- `models/model_card.json`: 피처 순서, 범주 인코딩 기준, 중앙값, 하이퍼파라미터, 임계값, 데이터 해시, 라이브러리 버전",
          "- `models/<모델>.joblib`: 비교한 모든 모델",
          "- `models/predictions.csv`: 고객별 예측 확률 (train/valid/test)",
          "- `reports/학습_결과서.html`: 이 문서를 그림까지 넣은 HTML 파일 하나로 (브라우저에서 열기, 인쇄 → PDF로 저장)",
          ""]
    (cfg.paths.reports_dir / "학습_결과서.md").write_text("\n".join(L), encoding="utf-8")
    log.info(f"[evaluate] 학습 결과서 저장: {cfg.paths.reports_dir / '학습_결과서.md'}")
