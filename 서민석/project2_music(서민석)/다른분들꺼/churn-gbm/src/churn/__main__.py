"""명령줄 실행.

    python -m churn all                     # 전체: eda → train → evaluate → explain → report
    python -m churn report                  # 학습 결과서 HTML만 다시 만들기
    python -m churn train                   # 단계별 실행
    python -m churn all --models hist_gb lightgbm   # 일부 모델만
    python -m churn all --quick             # 탐색 횟수를 줄여 빠르게 확인
    python -m churn check                   # 설치된 패키지 확인
    python -m churn predict --input 새고객.csv
"""
from __future__ import annotations

import argparse
import sys
import time

STEPS = ["eda", "train", "evaluate", "explain", "report"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m churn", description="고객 이탈 예측 - gradient boosting")
    ap.add_argument("step", choices=STEPS + ["all", "check", "predict"], help="실행할 단계")
    ap.add_argument("--config", default=None, help="설정 파일 경로 (기본: configs/config.yaml)")
    ap.add_argument("--models", nargs="+", help="학습할 모델 (hist_gb gbm lightgbm xgboost catboost)")
    ap.add_argument("--from", dest="start", choices=STEPS, help="all일 때 이 단계부터 실행")
    ap.add_argument("--quick", action="store_true", help="탐색 횟수 5, 교차검증 2겹으로 빠르게")
    ap.add_argument("--input", help="predict: 예측할 CSV")
    ap.add_argument("--output", help="predict: 결과 CSV (기본: 입력파일_예측.csv)")
    a = ap.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):  # Windows 콘솔 한글 깨짐 방지
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

    if a.step == "check":
        return _check()

    from .config import load_config
    from .utils import log

    ov = {}
    if a.models:
        ov["models"] = a.models
    if a.quick:
        ov["tuning"] = {"n_iter": 5, "cv_folds": 2}
    cfg = load_config(a.config, ov)

    if a.step == "predict":
        if not a.input:
            ap.error("predict에는 --input 이 필요합니다.")
        from .predict import run_predict

        run_predict(cfg, a.input, a.output)
        return 0

    steps = STEPS if a.step == "all" else [a.step]
    if a.step == "all" and a.start:
        steps = STEPS[STEPS.index(a.start):]
    for s in steps:
        t0 = time.time()
        _run(s, cfg)
        log.info(f"── {s} 완료 ({time.time() - t0:.0f}초)")
    return 0


def _run(step, cfg):
    if step == "eda":
        from .eda import run_eda

        run_eda(cfg)
    elif step == "train":
        from .train import run_train

        run_train(cfg)
    elif step == "evaluate":
        from .evaluate import run_evaluate

        run_evaluate(cfg)
    elif step == "explain":
        from .explain import run_explain

        run_explain(cfg)
    elif step == "report":
        from .report_html import run_report

        run_report(cfg)


def _check() -> int:
    import importlib
    import platform

    print(f"Python {platform.python_version()}")
    ok = True
    for mod, need in [("numpy", True), ("pandas", True), ("scipy", True), ("sklearn", True), ("matplotlib", True),
                      ("yaml", True), ("joblib", True), ("lightgbm", False), ("xgboost", False), ("catboost", False), ("shap", False), ("pytest", False)]:
        try:
            m = importlib.import_module(mod)
            print(f"  [있음] {mod:<11} {getattr(m, '__version__', '')}")
        except ImportError:
            print(f"  [없음] {mod:<11} {'← 필수' if need else '(해당 모델·기능만 건너뜀)'}")
            ok = ok and not need
    from .config import load_config

    cfg = load_config()
    print(f"  데이터 파일: {'있음' if cfg.paths.data.exists() else '없음'} ({cfg.paths.data})")
    return 0 if ok and cfg.paths.data.exists() else 1


if __name__ == "__main__":
    sys.exit(main())
