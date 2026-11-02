"""전처리. gradient boosting(트리 모델)에 맞춰 최소한으로 합니다. 모든 기준은 학습셋으로만 정합니다.

| 항목 | 처리 | 이유 |
|---|---|---|
| 결측 (수치) | 중앙값으로 채움 | 현재 데이터에는 결측이 없지만, 새 고객 입력에 대비 |
| 결측 (범주) | 최빈값으로 채움 | 같은 이유 |
| 범주 인코딩 | level_last → 정수 (free=0, paid=1) | 값이 두 개라 정수 인코딩과 원-핫이 정보상 같음. 처음 보는 값은 -1 |
| 스케일링 | 하지 않음 | 트리는 값의 크기가 아니라 순서로 분할하므로 결과가 같음 |
| 이상치 | 자르지 않음 | 팀 기준(큰 값만으로 오류라 판단할 근거 부족). 트리는 극단값에 덜 민감 |
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


def make_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    transformers = [("num", SimpleImputer(strategy="median"), list(numeric))]
    if categorical:
        transformers.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]), list(categorical)))
    ct = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
    return ct.set_output(transform="pandas")  # 피처 이름을 모델까지 유지


def describe_preprocessor(prep: ColumnTransformer, categorical: list[str]) -> dict:
    """학습셋으로 정해진 기준 (모델 카드에 저장)."""
    out = {"feature_order": list(prep.get_feature_names_out()), "numeric_median": {}, "categories": {}}
    num = prep.named_transformers_["num"]
    for name, med in zip(num.feature_names_in_, num.statistics_):
        out["numeric_median"][str(name)] = float(med)
    if categorical:
        enc = prep.named_transformers_["cat"].named_steps["encode"]
        for c, cats in zip(categorical, enc.categories_):
            out["categories"][c] = {str(v): i for i, v in enumerate(cats)}
    return out
