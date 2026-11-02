"""고객 이탈 대응 서비스의 모델·집계 로직.

확정 조합은 저장된 확률 비중과 판정 기준을 그대로 사용합니다.
상세 옵션의 단독·수동 결합은 검증 자료 기준 백분위 점수를 사용합니다.
고객 번호와 기준일에 맞는 시퀀스만 연결하며 앱에서 학습·재탐색하지 않습니다.
"""
from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix, fbeta_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from validation import prepare_customers, InputError, read_customers
import storage as snapshot_storage
from sequence_runtime import SequenceAdapter, SequenceInputError

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "customer_features_20181101_v1_with_split.csv"
MODEL_DIR = ROOT / "model" / "service"
STORE = ROOT / "service_data"

NUMERIC = ["active_days", "session_count", "song_count", "days_since_last_activity", "recent_7d_song_count",
           "previous_7d_song_count", "song_count_change", "thumbs_up_count", "thumbs_down_count",
           "playlist_add_count", "ad_count", "error_count", "help_count"]
FEATURES = NUMERIC + ["level_last"]
LEVEL_MAP = {"free": 0, "paid": 1}

ML_SPECS = {  # key: (화면 이름, 필요한 패키지)
    "xgboost": ("엑스지부스트 · XGBoost", "xgboost"),
    "lightgbm": ("라이트지비엠 · LightGBM", "lightgbm"),
    "catboost": ("캣부스트 · CatBoost", "catboost"),
    "hist_gb": ("히스토그램 부스팅 · sklearn", "sklearn"),
}
NN_SPECS = {
    "mlp_small": ("다층 신경망 · 32·16", (32, 16)),
    "mlp_large": ("다층 신경망 · 128·64·32", (128, 64, 32)),
}
MODE_NAMES = {"optimal": "최적 조합", "nn": "신경망 모델", "ml": "머신러닝 모델", "blend": "두 모델 결합"}

FEATURE_KO = {
    "active_days": "활동 날짜 수", "session_count": "세션 수", "song_count": "재생 곡 수",
    "days_since_last_activity": "마지막 활동 후 일수", "recent_7d_song_count": "최근 7일 재생",
    "previous_7d_song_count": "그 전 7일 재생", "song_count_change": "재생 수 변화",
    "thumbs_up_count": "좋아요", "thumbs_down_count": "싫어요", "playlist_add_count": "플레이리스트 추가",
    "ad_count": "광고 노출", "error_count": "오류", "help_count": "도움말 방문", "level_last": "요금제",
}

# (피처, 방향) → (이유, 추천 활동, 아이콘). high = 유지 고객보다 큼, low = 작음
RULES = {
    ("ad_count", "high"): ("광고 노출이 많아요", {"free": "광고 없는 프리미엄 7일 무료 체험",
                                              "paid": "광고 노출 설정 점검 + 사과 쿠폰"}, "📢"),
    ("thumbs_down_count", "high"): ("싫어요가 많아요 · 추천이 취향과 안 맞음", "취향 재설정 + 맞춤 플레이리스트 추천", "👎"),
    ("thumbs_up_count", "low"): ("좋아요가 적어요 · 만족한 곡이 적음", "취향 기반 신곡·인기 플레이리스트 추천", "💔"),
    ("playlist_add_count", "high"): ("플레이리스트를 많이 만들었어요 · 떠나면 손실이 큰 고객", "저장한 플레이리스트 상기 + 장기 이용 혜택", "🎵"),
    ("playlist_add_count", "low"): ("플레이리스트를 거의 안 만들어요 · 정착 전", "플레이리스트 만들기 안내", "🎵"),
    ("song_count", "high"): ("청취량이 많아요", "헤비 유저 만족도 조사 + VIP 혜택", "🎧"),
    ("song_count", "low"): ("청취량이 적어요", "맞춤 추천 알림으로 재방문 유도", "🎧"),
    ("session_count", "high"): ("접속이 잦아요 · 해지 전 탐색 가능성", "만족도 조사 후 불만 먼저 해결", "🔁"),
    ("session_count", "low"): ("접속이 드물어요", "재방문 유도 알림", "🔁"),
    ("active_days", "high"): ("활동일이 많아요 · 해지 전 탐색 가능성", "만족도 조사 후 불만 먼저 해결", "📅"),
    ("active_days", "low"): ("활동일이 적어요", "재방문 유도 알림", "📅"),
    ("days_since_last_activity", "high"): ("최근 접속하지 않았어요", "추천곡을 담은 재방문 푸시·이메일", "⏰"),
    ("days_since_last_activity", "low"): ("최근까지 접속 중이에요", "만족도 확인 메시지", "⏰"),
    ("recent_7d_song_count", "low"): ("최근 7일 청취가 적어요", "맞춤 추천 알림으로 재방문 유도", "📉"),
    ("recent_7d_song_count", "high"): ("최근 7일 청취가 많아요", "만족도 확인 메시지", "📈"),
    ("previous_7d_song_count", "high"): ("그 전 주 청취가 많았어요", "이용 패턴 변화 확인 + 맞춤 추천", "📊"),
    ("previous_7d_song_count", "low"): ("그 전 주 청취가 적었어요", "맞춤 추천 알림", "📊"),
    ("song_count_change", "low"): ("최근 이용량이 줄었어요", "이용 감소 고객 맞춤 콘텐츠 알림", "📉"),
    ("song_count_change", "high"): ("최근 이용량이 늘었어요", "만족도 확인 메시지", "📈"),
    ("error_count", "high"): ("오류를 자주 겪었어요", "기술 지원 연락 + 사과 혜택", "⚠️"),
    ("help_count", "high"): ("도움말을 자주 봤어요 · 불편 경험", "고객센터가 먼저 연락해 해결", "🆘"),
    ("level_last", "free"): ("무료 요금제 이용 중", "유료 전환 할인 쿠폰", "🎟️"),
    ("level_last", "paid"): ("유료 요금제 이용 중", "결제 할인·장기 이용 혜택", "💳"),
}
DEFAULT_ACTION = "만족도 조사 + 맞춤 혜택"


# ---------------------------------------------------------------
# 전처리
# ---------------------------------------------------------------
def signed_log1p(X):
    X = np.asarray(X, dtype=float)
    return np.sign(X) * np.log1p(np.abs(X))


def prepare_X(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"필요한 열이 없습니다: {missing}")
    checked = df.copy()
    if 'user_id' not in checked:
        checked['user_id'] = [f'검사{i}' for i in range(len(checked))]
    X = prepare_customers(checked)[FEATURES].copy()
    X.index = df.index
    for c in NUMERIC:
        X[c] = pd.to_numeric(X[c], errors="coerce").astype(float)
    X["level_last"] = X["level_last"].map(lambda v: LEVEL_MAP.get(str(v).strip().lower(), np.nan)).astype(float)
    X.attrs['customer_ids']=checked['user_id'].astype(str).tolist()
    X.attrs['as_of']=df.attrs.get('as_of',load_data_date())
    return X


def load_team_data(path: Path = DATA) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"user_id": str}, encoding="utf-8-sig")


class OriginalPipelineAdapter:
    """숫자로 변환한 요금제를 복원해 기존 전처리 포함 모델에 전달합니다."""
    classes_ = np.array([0,1])

    def __init__(self,pipeline):
        self.pipeline=pipeline

    def predict_proba(self,X):
        raw=X.copy()
        raw['level_last']=raw['level_last'].map({0.0:'free',1.0:'paid'})
        return self.pipeline.predict_proba(raw[FEATURES])


def service_signature():
    files=[DATA,ROOT/'models.py']
    files += [p for p in (ROOT/'model').rglob('*') if p.suffix in ['.pkl','.cbm','.joblib','.pth','.pt','.json']]
    files += list((ROOT/'data/processed').glob('*.npy'))+list((ROOT/'data/processed').glob('*_20181101_v1.csv'))
    files += list((STORE/'sequence_inputs').glob('*.npz'))
    return tuple((str(path),path.stat().st_mtime_ns,path.stat().st_size) for path in sorted(set(files)) if path.is_file())


def consistent_song_features(frame,changed=None):
    out=frame.copy()
    if changed=='song_count_change':
        recent=out['previous_7d_song_count']+out['song_count_change']
        out['recent_7d_song_count']=recent.clip(lower=0)
        out['previous_7d_song_count']=out['recent_7d_song_count']-out['song_count_change']
    total_week=out['recent_7d_song_count']+out['previous_7d_song_count']
    if changed=='song_count':
        over=total_week>out['song_count']
        ratio=out.loc[over,'song_count']/total_week[over]
        out.loc[over,'recent_7d_song_count']=np.floor(out.loc[over,'recent_7d_song_count']*ratio)
        out.loc[over,'previous_7d_song_count']=np.floor(out.loc[over,'previous_7d_song_count']*ratio)
    else:
        out['song_count']=np.maximum(out['song_count'],total_week)
    out['song_count_change']=out['recent_7d_song_count']-out['previous_7d_song_count']
    return out


def _has(pkg: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(pkg) is not None


# ---------------------------------------------------------------
# 학습
# ---------------------------------------------------------------
def _build(key: str, Xtr, ytr, Xva, yva):
    if key == "lightgbm":
        import lightgbm as lgb

        m = lgb.LGBMClassifier(objective="binary", learning_rate=0.03, max_depth=3, num_leaves=7, min_child_samples=50,
                               reg_alpha=1.0, reg_lambda=1.0, n_estimators=1000, random_state=42, verbose=-1)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="average_precision",
              callbacks=[lgb.early_stopping(50, verbose=False)])
        return m, int(m.best_iteration_ or 0)
    if key == "xgboost":
        from xgboost import XGBClassifier

        m = XGBClassifier(n_estimators=1000, learning_rate=0.03, max_depth=3, min_child_weight=5, subsample=0.9,
                          colsample_bytree=0.9, reg_lambda=1.0, eval_metric="aucpr", early_stopping_rounds=50,
                          tree_method="hist", random_state=42, n_jobs=-1)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        return m, int(m.best_iteration + 1)
    if key == "catboost":
        from catboost import CatBoostClassifier

        m = CatBoostClassifier(iterations=1000, learning_rate=0.03, depth=4, eval_metric="AUC", od_type="Iter",
                               od_wait=50, use_best_model=True, random_seed=42, verbose=0, allow_writing_files=False)
        m.fit(Xtr, ytr, eval_set=(Xva, yva))
        return m, int(m.get_best_iteration() + 1)
    if key == "hist_gb":
        from sklearn.ensemble import HistGradientBoostingClassifier

        m = HistGradientBoostingClassifier(learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=50,
                                           l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
                                           n_iter_no_change=30, random_state=42).fit(Xtr, ytr)
        return m, int(m.n_iter_)
    if key in NN_SPECS:
        from sklearn.impute import SimpleImputer
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import FunctionTransformer, StandardScaler

        m = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("log", FunctionTransformer(signed_log1p)),
            ("scale", StandardScaler()),
            # sklearn MLP의 조기 종료는 '정확도'로 멈춰서 불균형 데이터에서 너무 일찍 멈춤 →
            # 에폭 60 고정 + 강한 L2 규제(alpha=1.0). 검증셋 PR-AUC로 비교해서 고른 값
            ("mlp", MLPClassifier(hidden_layer_sizes=NN_SPECS[key][1], alpha=1.0, learning_rate_init=1e-3,
                                  batch_size=256, max_iter=60, random_state=42)),
        ]).fit(Xtr, ytr)
        return m, int(m.named_steps["mlp"].n_iter_)
    raise KeyError(key)


def train_all(model_dir: Path = MODEL_DIR, data_path: Path = DATA, progress=None) -> dict:
    df = load_team_data(data_path)
    parts = {s: df[df["split"] == s] for s in ("train", "valid", "test")}
    X = {s: prepare_X(d) for s, d in parts.items()}
    y = {s: d["churn"].to_numpy().astype(int) for s, d in parts.items()}
    keys = [k for k, (_, pkg) in ML_SPECS.items() if _has(pkg)] + list(NN_SPECS)
    model_dir.mkdir(parents=True, exist_ok=True)
    reg = {"created_at": datetime.now().isoformat(timespec="seconds"), "models": {},
           "data_sha256": hashlib.sha256(Path(data_path).read_bytes()).hexdigest()[:16],
           "split_sizes": {s: int(len(v)) for s, v in y.items()},
           "base_rate": {s: float(v.mean()) for s, v in y.items()}}
    probs = {f"y_{s}": y[s] for s in ("valid", "test")}
    for i, key in enumerate(keys):
        if progress:
            progress(i / len(keys), f"{(ML_SPECS | NN_SPECS)[key][0]} 학습 중")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model, n_iter = _build(key, X["train"], y["train"], X["valid"], y["valid"])
        p = {s: model.predict_proba(X[s])[:, 1] for s in X}
        joblib.dump(model, model_dir / f"{key}.joblib")
        for s in ("valid", "test"):
            probs[f"{key}_{s}"] = p[s]
        reg["models"][key] = {
            "name": (ML_SPECS | NN_SPECS)[key][0], "kind": "ml" if key in ML_SPECS else "nn", "iterations": n_iter,
            **{f"{s}_{m}": float(f(y[s], p[s])) for s in ("train", "valid", "test")
               for m, f in (("pr_auc", average_precision_score), ("roc_auc", roc_auc_score))},
        }
    retained = parts["train"][parts["train"]["churn"] == 0]
    reg["reference"] = {c: float(retained[c].median()) for c in NUMERIC}
    reg["reference"]["level_last"] = str(retained["level_last"].mode().iloc[0])
    ml = max((k for k in reg["models"] if reg["models"][k]["kind"] == "ml"), key=lambda k: reg["models"][k]["valid_pr_auc"])
    nn = max((k for k in reg["models"] if reg["models"][k]["kind"] == "nn"), key=lambda k: reg["models"][k]["valid_pr_auc"])
    reg["default"] = {"ml": ml, "nn": nn}
    np.savez_compressed(model_dir / "probs.npz", **probs)
    (model_dir / "registry.json").write_text(json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8")
    if progress:
        progress(1.0, "학습 완료")
    return reg


# ---------------------------------------------------------------
# 운영 설정
# ---------------------------------------------------------------
@dataclass(frozen=True)
class Setting:
    ml: str
    nn: str
    mode: str = "blend"          # nn | ml | blend
    ml_weight: float = 0.5       # blend일 때 머신러닝 비중 (0~1)
    use_saved: bool = True       # 검증셋에서 정한 판정 기준 사용
    manual_threshold: float = 80.0

    @property
    def label(self) -> str:
        if self.mode == "blend":
            return f"두 모델 결합 (머신러닝 {self.ml_weight:.0%})"
        return MODE_NAMES[self.mode]


@dataclass
class Service:
    models: dict
    reg: dict
    probs: dict
    team: pd.DataFrame
    _sorted: dict = field(default_factory=dict)

    # 점수 ------------------------------------------------------
    def to_score(self, key: str, p: np.ndarray) -> np.ndarray:
        """확률 → 검증셋 기준 백분위 점수 (0~100)."""
        if key not in self._sorted:
            self._sorted[key] = np.sort(self.probs[f"{key}_valid"])
        v = self._sorted[key]
        return 100.0 * np.searchsorted(v, np.asarray(p), side="right") / len(v)

    def model_scores(self, key: str, X: pd.DataFrame) -> np.ndarray:
        return self.to_score(key, self.models[key].predict_proba(X)[:, 1])

    def setting_model_scores(self, s, key, X):
        if s.mode=='optimal':return self.models[key].predict_proba(X)[:,1]*100
        return self.model_scores(key,X)

    def validate_population(self, s, df):
        if isinstance(self.models[s.nn],SequenceAdapter):
            self.models[s.nn].bank.validate(df,df.attrs.get('as_of',load_data_date()))

    def combine(self, s: Setting, ml_s: np.ndarray, nn_s: np.ndarray) -> np.ndarray:
        if s.mode == "ml":
            return ml_s
        if s.mode == "nn":
            return nn_s
        return s.ml_weight * ml_s + (1 - s.ml_weight) * nn_s

    def scores(self, s: Setting, df: pd.DataFrame, _scenario=False) -> pd.DataFrame:
        if not _scenario:self.validate_population(s,df)
        X = prepare_X(df)
        ml_s, nn_s = self.setting_model_scores(s,s.ml,X), self.setting_model_scores(s,s.nn,X)
        return pd.DataFrame({"ml_score": ml_s, "nn_score": nn_s, "score": self.combine(s, ml_s, nn_s)}, index=df.index)

    def split_scores(self, s: Setting, split: str) -> tuple[np.ndarray, np.ndarray]:
        if s.mode=='optimal':
            return self.combine(s,self.probs[f'{s.ml}_{split}']*100,self.probs[f'{s.nn}_{split}']*100),self.probs[f'y_{split}']
        ml_s = self.to_score(s.ml, self.probs[f"{s.ml}_{split}"])
        nn_s = self.to_score(s.nn, self.probs[f"{s.nn}_{split}"])
        return self.combine(s, ml_s, nn_s), self.probs[f"y_{split}"]

    def threshold(self, s: Setting) -> float:
        if not s.use_saved:
            return float(s.manual_threshold)
        if s.mode=='optimal':return float(self.reg['optimal']['best_threshold'])*100
        return _f2_threshold(*self.split_scores(s, "valid"))

    def model_threshold(self, key: str, kind: str, setting=None) -> float:
        if setting is not None and setting.mode=='optimal':
            return _f2_threshold(self.probs[f'{key}_valid']*100,self.probs['y_valid'])
        base = Setting(ml=key if kind == "ml" else self.reg["default"]["ml"],
                       nn=key if kind == "nn" else self.reg["default"]["nn"], mode=kind)
        return self.threshold(base)

    def evaluate(self, s: Setting, split: str = "test") -> dict:
        if split != 'valid':
            raise ValueError('운영 앱에서는 검증 자료만 조회합니다. 최종 평가는 모델과 설정을 확정한 뒤 별도로 수행하세요.')
        sc, y = self.split_scores(s, split)
        thr = self.threshold(s)
        pred = (sc >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        fpr, tpr, _ = roc_curve(y, sc)
        pr, rc, _ = precision_recall_curve(y, sc)
        o = np.argsort(-sc, kind="mergesort")
        cum = np.cumsum(y[o])
        return {"threshold": thr, "pr_auc": float(average_precision_score(y, sc)), "roc_auc": float(roc_auc_score(y, sc)),
                "precision": float(precision_score(y, pred, zero_division=0)),
                "recall": float(recall_score(y, pred, zero_division=0)),
                "f2": float(fbeta_score(y, pred, beta=2, zero_division=0)),
                "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn), "n": int(len(y)), "positives": int(y.sum()),
                "fpr": fpr, "tpr": tpr, "pr": pr, "rc": rc, "cum_caught": cum, "base_rate": float(y.mean())}

    # 이유·추천 -------------------------------------------------
    @property
    def reference(self) -> dict:
        return self.reg["reference"]

    def contributions(self, s: Setting, df: pd.DataFrame) -> pd.DataFrame:
        """값 = 현재 점수 − (그 피처만 유지 고객 기준값으로 바꾼 점수). 양수면 그 값이 위험을 올림."""
        X = prepare_X(df)
        base = self.combine(s, self.setting_model_scores(s,s.ml,X), self.setting_model_scores(s,s.nn,X))
        out = {}
        for f in FEATURES:
            Xc = X.copy()
            Xc[f] = LEVEL_MAP[self.reference[f]] if f == "level_last" else self.reference[f]
            Xc = consistent_song_features(Xc,f)
            out[f] = base - self.combine(s, self.setting_model_scores(s,s.ml,Xc), self.setting_model_scores(s,s.nn,Xc))
        return pd.DataFrame(out, index=df.index)

    def reasons(self, row: pd.Series, contrib: pd.Series, top: int = 3) -> list[dict]:
        level = str(row.get("level_last", "")).strip().lower()
        items = []
        for f in contrib.sort_values(ascending=False).index:
            c = float(contrib[f])
            if c <= 0:
                break
            d = _direction(f, row[f], self.reference[f])
            rule = RULES.get((f, d)) if d else None
            if rule is None:
                continue
            text, action, icon = rule
            if isinstance(action, dict):
                action = action.get(level, next(iter(action.values())))
            items.append({"feature": f, "name": FEATURE_KO[f], "contribution": c, "value": row[f],
                          "reference": self.reference[f], "reason": text, "action": action, "icon": icon})
            if len(items) >= top:
                break
        return items

    def analyze(self, s: Setting, df: pd.DataFrame) -> pd.DataFrame:
        """고객 목록: 점수, 판정, 두 모델 비교, 주요 이유, 추천 활동."""
        sc = self.scores(s, df)
        thr = self.threshold(s)
        thr_ml, thr_nn = self.model_threshold(s.ml, "ml",s), self.model_threshold(s.nn, "nn",s)
        contrib = self.contributions(s, df)
        r1, r2, acts = [], [], []
        for idx in df.index:
            items = self.reasons(df.loc[idx], contrib.loc[idx], top=2)
            r1.append(f"{items[0]['icon']} {items[0]['reason']}" if items else "-")
            r2.append(f"{items[1]['icon']} {items[1]['reason']}" if len(items) > 1 else "-")
            acts.append(" / ".join(dict.fromkeys(i["action"] for i in items)) or DEFAULT_ACTION)
        out = pd.DataFrame({"user_id": df["user_id"].astype(str)}, index=df.index).join(sc)
        out["pred"] = np.where(out["score"] >= thr, "이탈 예상", "유지 예상")
        ml_flag, nn_flag = out["ml_score"] >= thr_ml, out["nn_score"] >= thr_nn
        out["agree"] = np.where(ml_flag == nn_flag, "일치", "다름")
        out["reason_1"], out["reason_2"], out["action"] = r1, r2, acts
        return out

    def what_if(self, s: Setting, row: pd.Series, changes: dict) -> tuple[float, float]:
        a = pd.DataFrame([row[['user_id']+FEATURES].to_dict()])
        b = a.copy()
        for k, v in changes.items():
            b.loc[0, k] = v
        changed='song_count' if 'song_count' in changes else 'song_count_change' if 'song_count_change' in changes else None
        b = consistent_song_features(b,changed)
        return float(self.scores(s, a)["score"].iloc[0]), float(self.scores(s, b,_scenario=True)["score"].iloc[0])

    # 캠페인 ---------------------------------------------------
    def campaign(self, s: Setting, population: int, value: float, cost: float, success: float,
                 budget: float | None = None) -> tuple[pd.DataFrame, dict]:
        sc, y = self.split_scores(s, "valid")
        o = np.argsort(-sc, kind="mergesort")
        cum = np.cumsum(y[o])
        rows = []
        for q in np.linspace(0.01, 1, 100):
            k = max(1, int(round(q * len(y))))
            rows.append({"frac": q, "precision": cum[k - 1] / k, "recall": cum[k - 1] / max(y.sum(), 1)})
        g = pd.DataFrame(rows)
        g=pd.concat([pd.DataFrame([{'frac':0.0,'precision':0.0,'recall':0.0}]),g],ignore_index=True)
        g["contacts"] = np.where(g['frac']==0,0,np.maximum(1, np.round(g["frac"] * population))).astype(int)
        g["churners"] = g["precision"] * g["contacts"]
        g["saved"] = g["churners"] * success
        g["cost"] = g["contacts"] * cost
        g["profit"] = g["saved"] * value - g["cost"]
        ok = g if budget is None else g[g["cost"] <= budget]
        if ok.empty:
            return g, {"contacts": 0, "frac": 0.0, "churners": 0.0, "saved": 0.0, "cost": 0.0, "profit": 0.0,
                       "precision": float("nan"), "recall": 0.0}
        r = ok.loc[ok["profit"].idxmax()]
        best = {k: float(r[k]) for k in ["frac", "churners", "saved", "cost", "profit", "precision", "recall"]}
        best["contacts"] = int(r["contacts"])
        return g, best


def _f2_threshold(score: np.ndarray, y: np.ndarray) -> float:
    pr, rc, th = precision_recall_curve(y, score)
    f2 = 5 * pr[:-1] * rc[:-1] / np.maximum(4 * pr[:-1] + rc[:-1], 1e-12)
    return float(th[int(np.nanargmax(f2))])


def _direction(f, value, ref):
    if f == "level_last":
        v = str(value).strip().lower()
        return v if v in LEVEL_MAP else None
    try:
        v, r = float(value), float(ref)
    except (TypeError, ValueError):
        return None
    if np.isnan(v) or v == r:
        return None
    return "high" if v > r else "low"


def default_setting(svc: Service) -> Setting:
    """팀원이 확정한 모델·확률 비중·판정 기준을 그대로 사용합니다."""
    config=svc.reg['optimal']
    return Setting(Path(config['best_ml_model']).stem,'lstm_sequence',mode='optimal',ml_weight=float(config['ml_weight']))


SETTING_FILE = "settings.json"


def save_setting(s: Setting, store: Path = STORE) -> None:
    """운영 설정은 모든 사용자에게 같게 적용되도록 파일에 저장."""
    store.mkdir(parents=True, exist_ok=True)
    (store / SETTING_FILE).write_text(json.dumps(s.__dict__, ensure_ascii=False, indent=2), encoding="utf-8")


def load_setting(svc: "Service", store: Path = STORE) -> Setting:
    f = store / SETTING_FILE
    if f.exists():
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            s = Setting(**d)
            if s.ml in svc.models and s.nn in svc.models and s.mode in MODE_NAMES and 0<=s.ml_weight<=1 and 0<=s.manual_threshold<=100:
                if s.mode=='optimal':
                    fixed=default_setting(svc)
                    return Setting(fixed.ml,fixed.nn,mode='optimal',ml_weight=fixed.ml_weight,use_saved=s.use_saved,manual_threshold=s.manual_threshold)
                return s
        except Exception:
            pass
    return default_setting(svc)


def load_service(model_dir: Path = MODEL_DIR, data_path: Path = DATA, progress=None) -> Service:
    team = load_team_data(data_path)
    if team['user_id'].isna().any() or team['user_id'].duplicated().any():
        raise ValueError('공통 자료의 고객 번호 누락 또는 중복을 확인하세요.')
    if set(team['split']) != {'train','valid','test'}:
        raise ValueError('공통 학습·검증·최종 평가 분할이 필요합니다.')
    valid = team[team['split']=='valid']
    valid.attrs['as_of']='2018-11-01'
    Xv = prepare_X(valid)
    Xv.attrs['reference_input']=True
    yv = valid['churn'].to_numpy(dtype=int)
    models, specs, unavailable = {}, {}, []
    titles = {'ab_model':'에이다부스트','lr_model':'로지스틱 회귀','rf_model':'랜덤 포레스트','lgb_model':'라이트지비엠','catboost_model':'캐트부스트'}
    for path in sorted((ROOT/'model/machine_learning').glob('*')):
        if path.suffix not in ['.pkl','.cbm','.joblib']:
            continue
        try:
            if path.suffix=='.cbm':
                from catboost import CatBoostClassifier
                model = CatBoostClassifier();model.load_model(path)
            else:
                model = joblib.load(path)
            names = getattr(model,'feature_names_in_',getattr(model,'feature_names_',None))
            if names is None or list(names)!=FEATURES:
                raise ValueError('학습 피처 이름과 순서가 공통 14개 항목과 다릅니다.')
            if list(model.classes_) != [0,1]:
                raise ValueError('정답 1의 예측 확률을 확인할 수 없습니다.')
            key=path.stem
            models[key]=model
            specs[key]={'name':titles.get(key,path.stem),'kind':'ml','source':str(path.relative_to(ROOT)), 'training_split_verified':False}
        except Exception as error:
            unavailable.append({'file':path.name,'reason':str(error)})
    for path in sorted((ROOT/'model/deep_learning').glob('*.joblib')):
        try:
            saved=joblib.load(path)
            if not isinstance(saved,dict) or saved.get('feature_columns')!=FEATURES or 'pipeline' not in saved:
                raise ValueError('전처리와 피처 목록을 포함한 신경망 묶음이 필요합니다.')
            key=path.stem
            models[key]=OriginalPipelineAdapter(saved['pipeline'])
            specs[key]={'name':'다층 신경망 · 공통 피처','kind':'nn','source':str(path.relative_to(ROOT)), 'training_split_verified':True}
        except Exception as error:
            unavailable.append({'file':path.name,'reason':str(error)})
    models['lstm_sequence']=SequenceAdapter(ROOT,FEATURES)
    specs['lstm_sequence']={'name':'장단기 기억 신경망','kind':'nn','source':'model/deep_learning/lstm_model_weights.pth','training_split_verified':True}
    for path in sorted((ROOT/'model/deep_learning').glob('*')):
        if path.name=='lstm_model_weights.pth':continue
        if path.suffix in ['.pth','.pt']:
            unavailable.append({'file':path.name,'reason':'시퀀스 입력의 행동 사전·패딩·길이·고객 연결 규칙이 없어 연결 보류. 임의 입력으로 예측하지 않습니다.'})
    probs={'y_valid':yv}
    for key in list(models):
        try:
            p=models[key].predict_proba(Xv)[:,1]
            if not np.isfinite(p).all():raise ValueError('유한하지 않은 예측값')
            probs[key+'_valid']=p
            specs[key].update(valid_pr_auc=float(average_precision_score(yv,p)),valid_roc_auc=float(roc_auc_score(yv,p)),iterations=0)
        except Exception as error:
            unavailable.append({'file':specs[key]['source'],'reason':str(error)})
            del models[key];del specs[key]
    ml=[key for key in specs if specs[key]['kind']=='ml']
    nn=[key for key in specs if specs[key]['kind']=='nn']
    if not ml or not nn:
        raise ValueError('호환되는 머신러닝 모델과 신경망 모델이 각각 하나 이상 필요합니다. 자동 재학습은 수행하지 않습니다.')
    retained=team[(team['split']=='train')&(team['churn']==0)]
    reference={c:float(retained[c].median()) for c in NUMERIC}
    reference['level_last']=str(retained['level_last'].mode().iloc[0])
    reg={'models':specs,'default':{'ml':max(ml,key=lambda key:specs[key]['valid_pr_auc']),'nn':max(nn,key=lambda key:specs[key]['valid_pr_auc'])},
         'optimal':json.loads((ROOT/'model/best_hybrid_config.json').read_text(encoding='utf-8')),
         'reference':reference,'base_rate':{'valid':float(yv.mean())},'unavailable':unavailable,
         'data_sha256':hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
         'evaluation_scope':'검증 연결 점검. 제공 모델의 원래 학습 분할은 별도 확인 필요. 최종 평가 미실시.'}
    config=reg['optimal']
    if config['best_dl_model']!='lstm_model_weights.pth' or Path(config['best_ml_model']).stem not in models:
        raise ValueError('최적 설정의 모델과 현재 연결 코드가 다릅니다. 인수인계 안내서의 모델 교체 절차를 확인하세요.')
    weights=[float(config[k]) for k in ['ml_weight','dl_weight']]
    if not all(np.isfinite(v) and 0<=v<=1 for v in weights) or not np.isclose(sum(weights),1) or not 0<=float(config['best_threshold'])<=1:
        raise ValueError('최적 설정의 비중 합과 판정 기준을 확인하세요.')
    if progress:progress(1.0,'저장된 모델 연결 완료 · 재학습 없음')
    return Service(models=models,reg=reg,probs=probs,team=team)


# ---------------------------------------------------------------
# 고객 자료 관리 (service_data/ 에 저장)
# ---------------------------------------------------------------
CURRENT = "current_customers.csv"
HISTORY = "history.csv"


def default_customers(team: pd.DataFrame) -> pd.DataFrame:
    """기본 분석 자료 = 학습에 쓰지 않은 테스트 고객 (실제 결과는 검증용으로만 표시)."""
    return team[team["split"] == "valid"][["user_id"] + FEATURES + ["churn"]].reset_index(drop=True)


def load_customers(team: pd.DataFrame, store: Path = STORE) -> pd.DataFrame:
    f = store / CURRENT
    if f.exists():
        df = pd.read_csv(f, dtype={"user_id": str}, encoding="utf-8-sig")
        if "churn" not in df.columns:
            df["churn"] = np.nan
        errors,_=check_upload(df)
        if errors:raise ValueError('저장 고객 자료 검사 실패: '+' / '.join(errors))
        return df
    return default_customers(team)


def load_data_date(store: Path = STORE):
    f=store/'data_metadata.json'
    return json.loads(f.read_text(encoding='utf-8')).get('as_of','2018-11-01') if f.exists() else '2018-11-01'


def save_customers(df: pd.DataFrame, store: Path = STORE, as_of=None) -> None:
    store.mkdir(parents=True, exist_ok=True)
    errors,_=check_upload(df)
    if errors:raise ValueError(' / '.join(errors))
    current=store/CURRENT
    if current.is_file():
        import shutil
        backup=store/'backups';backup.mkdir(exist_ok=True)
        shutil.copyfile(current,backup/f'customers_{datetime.now():%Y%m%d_%H%M%S_%f}.csv')
    df.to_csv(store / CURRENT, index=False, encoding="utf-8-sig")
    day=str(as_of) if as_of is not None else load_data_date(store)
    (store/'data_metadata.json').write_text(json.dumps({'as_of':day},ensure_ascii=False),encoding='utf-8')


def reset_customers(store: Path = STORE):
    current=store/CURRENT
    if current.is_file():
        backup=store/'backups';backup.mkdir(parents=True,exist_ok=True)
        current.replace(backup/f'reset_{datetime.now():%Y%m%d_%H%M%S_%f}.csv')
    (store/'data_metadata.json').write_text(json.dumps({'as_of':'2018-11-01'}),encoding='utf-8')


def log_history(action: str, detail: str, total: int, store: Path = STORE) -> None:
    store.mkdir(parents=True, exist_ok=True)
    f = store / HISTORY
    row = pd.DataFrame([{"시각": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "작업": action, "내용": detail,
                         "적용 후 고객 수": total}])
    row.to_csv(f, mode="a", header=not f.exists(), index=False, encoding="utf-8-sig")


def load_history(store: Path = STORE) -> pd.DataFrame:
    f = store / HISTORY
    return pd.read_csv(f, encoding="utf-8-sig") if f.exists() else pd.DataFrame(columns=["시각", "작업", "내용", "적용 후 고객 수"])


def check_upload(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    errors, warns = [], []
    missing = [c for c in ["user_id"] + FEATURES if c not in df.columns]
    if missing:
        return [f"필요한 열이 없습니다: {', '.join(missing)}"], warns
    if df.empty:
        return ["행이 없습니다."], warns
    dup = int(df["user_id"].astype(str).duplicated().sum())
    if dup:
        warns.append(f"파일 안에 같은 고객 번호가 {dup}건 있어 마지막 행만 사용합니다.")
    for c in NUMERIC:
        v = pd.to_numeric(df[c], errors="coerce")
        bad = int(v.isna().sum() - df[c].isna().sum())
        if bad:
            errors.append(f"{c}: 숫자가 아닌 값 {bad}개를 수정하세요.")
        if c != "song_count_change" and (v < 0).any():
            errors.append(f"{c}: 허용하지 않는 음수 값 {int((v < 0).sum())}개")
    odd = sorted(set(df["level_last"].dropna().astype(str).str.strip().str.lower()) - set(LEVEL_MAP))
    if odd:
        errors.append(f"level_last: 알 수 없는 값 {odd[:5]}을 수정하세요. free/paid만 인식합니다.")
    try:
        if df['user_id'].isna().any() or df['user_id'].astype(str).str.strip().eq('').any():
            raise InputError('고객 번호가 비어 있습니다.')
        prepare_customers(df.drop_duplicates('user_id',keep='last'))
        if 'churn' in df:
            truth=pd.to_numeric(df['churn'],errors='coerce')
            if not truth.dropna().isin([0,1]).all() or (truth.isna()&df['churn'].notna()).any():
                raise InputError('실제 해지 정답은 비워 두거나 0과 1로만 입력하세요.')
    except InputError as error:
        errors.append(str(error))
    return errors, warns


def merge_customers(current: pd.DataFrame, new: pd.DataFrame, mode: str, on_conflict: str) -> tuple[pd.DataFrame, dict]:
    """mode: append(추가·갱신) | replace(교체). on_conflict: keep(기존 유지) | update(새 자료로 갱신)."""
    errors,_=check_upload(new)
    if errors:raise ValueError(' / '.join(errors))
    new = new.copy()
    new["user_id"] = new["user_id"].astype(str).str.strip()
    new = new.drop_duplicates("user_id", keep="last")
    checked=prepare_customers(new)
    for column in FEATURES:
        new[column]=checked[column].to_numpy()
    if "churn" not in new.columns:
        new["churn"] = np.nan
    else:
        new['churn']=pd.to_numeric(new['churn'],errors='coerce')
    new = new[["user_id"] + FEATURES + ["churn"]]
    if mode == "replace":
        return new.reset_index(drop=True), {"added": len(new), "updated": 0, "kept": 0, "total": len(new)}
    cur = current.set_index("user_id")
    for column in NUMERIC:
        cur[column]=pd.to_numeric(cur[column],errors='raise').astype(float)
    nw = new.set_index("user_id")
    overlap = nw.index.intersection(cur.index)
    fresh = nw.loc[~nw.index.isin(cur.index)]
    if on_conflict == "update":
        cur.loc[overlap, FEATURES] = nw.loc[overlap, FEATURES]
        cur.loc[overlap, "churn"] = np.nan  # 새 기록이라 실제 결과는 아직 모름
        updated, kept = len(overlap), 0
    else:
        updated, kept = 0, len(overlap)
    out = pd.concat([cur, fresh]).reset_index()
    return out, {"added": len(fresh), "updated": updated, "kept": kept, "total": len(out)}


# ---------------------------------------------------------------
# 대응 효과 측정 (A/B)
# ---------------------------------------------------------------
def uplift_test(n_t: int, churn_t: int, n_c: int, churn_c: int) -> dict:
    """대응군(t)과 대조군(c)의 14일 이탈률 비교. 두 비율 z-검정."""
    from math import erf, sqrt

    rt, rc = churn_t / n_t, churn_c / n_c
    diff = rc - rt  # 양수면 대응으로 이탈이 줄어든 것
    p = (churn_t + churn_c) / (n_t + n_c)
    se0 = sqrt(p * (1 - p) * (1 / n_t + 1 / n_c)) or 1e-12
    z = diff / se0
    pval = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
    se = sqrt(rt * (1 - rt) / n_t + rc * (1 - rc) / n_c)
    return {"rate_t": rt, "rate_c": rc, "diff": diff, "rel": diff / rc if rc else float("nan"), "z": z, "p": pval,
            "ci": (diff - 1.96 * se, diff + 1.96 * se), "saved": diff * n_t}


if __name__ == "__main__":
    service=load_service()
    print('저장 모델 연결 완료. 학습과 최종 평가를 수행하지 않았습니다.')
    for key,entry in service.reg['models'].items():
        print(entry['name'], '검증 평균정밀도', round(entry['valid_pr_auc'],4))
