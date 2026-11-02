"""1단계: 탐색적 데이터 분석. 학습셋만 사용해 검증·테스트 정보가 판단에 섞이지 않게 합니다.

결과 (reports/eda/)
- class_balance.png      이탈 비율 → 평가 지표 선택 근거
- univariate_auc.csv     피처 하나만으로 낸 AUC → 이탈과 관련 큰 행동, 누수 의심 점검
- dist_by_label.png      상위 피처의 이탈/유지 분포
- churn_by_level.png     요금제별 이탈률
- corr_heatmap.png       스피어만 상관 → 중복 피처 확인
- distribution.csv       분위수·왜도·0 비율 → 스케일링·이상치 처리 판단 근거
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .config import Config
from .data import load_dataset
from .utils import log, setup_matplotlib, write_json

DESC = {
    "active_days": "관찰 기간 활동 날짜 수",
    "session_count": "관찰 기간 세션 수",
    "song_count": "관찰 기간 NextSong 수",
    "days_since_last_activity": "마지막 활동 후 기준일까지 일수",
    "recent_7d_song_count": "기준일 직전 7일 NextSong 수",
    "previous_7d_song_count": "그 앞 7일 NextSong 수",
    "song_count_change": "최근 7일 - 이전 7일 NextSong 수",
    "thumbs_up_count": "Thumbs Up 수",
    "thumbs_down_count": "Thumbs Down 수",
    "playlist_add_count": "Add to Playlist 수",
    "ad_count": "Roll Advert 수",
    "error_count": "Error 수",
    "help_count": "Help 수",
    "level_last": "기준일 직전 요금제 (free/paid)",
}


def run_eda(cfg: Config) -> dict:
    setup_matplotlib()
    import matplotlib.pyplot as plt

    cfg.paths.ensure()
    out = cfg.paths.eda_dir
    ds = load_dataset(cfg)
    tr = ds.df[ds.df["split"] == "train"].reset_index(drop=True)
    y = tr[ds.target].to_numpy()
    X = tr[ds.numeric].astype(float)
    log.info(f"[eda] 학습셋 {len(tr):,}명으로 분석합니다.")

    # 1) 클래스 비율
    counts = pd.Series(y).value_counts().reindex([0, 1], fill_value=0)
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ax.bar(["유지(0)", "이탈(1)"], counts.values, color=["#4C78A8", "#E45756"])
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}\n({v / len(tr):.1%})", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, counts.max() * 1.25)
    ax.set_title("학습셋 이탈 비율")
    fig.savefig(out / "class_balance.png"); plt.close(fig)

    # 2) 분포 (스케일링·이상치 판단 근거)
    dist = pd.DataFrame({
        "min": X.min(), "p50": X.median(), "p95": X.quantile(0.95), "p99": X.quantile(0.99), "max": X.max(),
        "skew": X.skew(), "zero_ratio": (X == 0).mean(), "missing": X.isna().sum(),
    })
    dist.index.name = "feature"
    dist.to_csv(out / "distribution.csv", encoding="utf-8-sig")

    # 3) 단변량 AUC
    rows = []
    for c in ds.numeric:
        auc = roc_auc_score(y, X[c].fillna(X[c].median()))
        rows.append({"feature": c, "auc": auc, "strength": abs(auc - 0.5),
                     "direction": "높을수록 이탈" if auc > 0.5 else "낮을수록 이탈",
                     "churn_median": X.loc[y == 1, c].median(), "stay_median": X.loc[y == 0, c].median(),
                     "desc": DESC.get(c, "")})
    uni = pd.DataFrame(rows).sort_values("strength", ascending=False).reset_index(drop=True)
    uni.to_csv(out / "univariate_auc.csv", index=False, encoding="utf-8-sig")
    leak = uni.loc[uni["strength"] >= 0.45, "feature"].tolist()

    # 4) 상위 피처 분포 (log1p 축, 음수가 있는 피처는 원래 값)
    top = uni["feature"].head(8).tolist()
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.4))
    for ax, c in zip(axes.ravel(), top):
        v = X[c]
        use_log = (v.min() >= 0)
        data = [np.log1p(v[y == k]) if use_log else v[y == k] for k in (0, 1)]
        ax.boxplot(data, showfliers=False)
        ax.set_xticks([1, 2], ["유지", "이탈"])
        ax.set_title(f"{c}" + ("\n(log1p)" if use_log else ""), fontsize=9)
    fig.suptitle("이탈 여부별 주요 피처 분포 (학습셋, 이상치 표시 생략)")
    fig.tight_layout()
    fig.savefig(out / "dist_by_label.png"); plt.close(fig)

    # 5) 범주별 이탈률
    cat_rows = {}
    for c in ds.categorical:
        t = tr.groupby(tr[c].fillna("(결측)"))[ds.target].agg(["mean", "size"])
        cat_rows[c] = t
    if cat_rows:
        fig, axes = plt.subplots(1, len(cat_rows), figsize=(4.2 * len(cat_rows), 3.2))
        for ax, (c, t) in zip(np.atleast_1d(axes), cat_rows.items()):
            ax.bar(t.index.astype(str), t["mean"], color="#72B7B2")
            for i, (m, n) in enumerate(zip(t["mean"], t["size"])):
                ax.text(i, m, f"{m:.1%}\n(n={n:,})", ha="center", va="bottom", fontsize=9)
            ax.axhline(y.mean(), color="gray", ls="--", lw=1)
            ax.set_ylim(0, t["mean"].max() * 1.35)
            ax.set_title(f"{c}별 이탈률 (점선=전체)")
        fig.savefig(out / "churn_by_level.png"); plt.close(fig)

    # 6) 상관
    corr = X.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(corr)), corr.columns, fontsize=8)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iat[i, j]:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if abs(corr.iat[i, j]) > 0.6 else "black")
    fig.colorbar(im, ax=ax, shrink=0.7)
    ax.set_title("스피어만 상관 (학습셋)")
    fig.savefig(out / "corr_heatmap.png"); plt.close(fig)
    pairs = [(a, b, corr.loc[a, b]) for i, a in enumerate(corr.columns) for b in corr.columns[i + 1:]
             if abs(corr.loc[a, b]) >= 0.85]
    pairs.sort(key=lambda t: -abs(t[2]))

    summary = {
        "n_train": int(len(tr)), "train_churn_rate": float(y.mean()),
        "leak_suspects": leak, "high_corr_pairs": [[a, b, float(r)] for a, b, r in pairs],
        "categorical_churn_rate": {c: {str(k): float(v) for k, v in t["mean"].items()} for c, t in cat_rows.items()},
    }
    write_json(summary, out / "eda_summary.json")
    log.info(f"[eda] 완료: 단변량 AUC 1위 {uni.loc[0, 'feature']} ({uni.loc[0, 'auc']:.3f}), "
             f"누수 의심 {leak or '없음'}, 강한 상관 쌍 {len(pairs)}개")
    return summary
