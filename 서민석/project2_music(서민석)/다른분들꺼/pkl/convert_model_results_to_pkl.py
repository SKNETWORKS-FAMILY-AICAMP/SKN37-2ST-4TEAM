"""다른분들꺼 안의 저장된 평가 자료를 모델별 pickle 결과 묶음으로 저장합니다."""

import json
import pickle
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(__file__).resolve().parent

RESULT_FOLDERS = {
    "logistic_regression": "logistic_regression",
    "mlp": "mlp",
    "svm": "svm",
}
NOTEBOOK_PATH = ROOT_DIR / "02_랜덤포레스트_모델링.ipynb"
GBM_MODELS_DIR = ROOT_DIR / "churn-gbm" / "models"
GBM_REPORTS_DIR = ROOT_DIR / "churn-gbm" / "reports"


def load_result_folder(folder_name: str) -> dict:
    """폴더의 CSV, JSON, 그림을 읽어 하나의 결과 사전으로 묶습니다."""
    folder = ROOT_DIR / folder_name
    tables = {}
    summaries = {}
    figures = {}
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() == ".csv":
            tables[path.stem] = pd.read_csv(path, encoding="utf-8-sig")
        elif path.suffix.lower() == ".json":
            with path.open("r", encoding="utf-8-sig") as file:
                summaries[path.stem] = json.load(file)
        elif path.suffix.lower() == ".png":
            figures[path.name] = path.read_bytes()
    return {
        "model_group": folder_name,
        "artifact_type": "evaluation_results_bundle",
        "tables": tables,
        "summaries": summaries,
        "figures_png_bytes": figures,
        "source_folder": str(folder),
        "note": "평가 결과 자료이며, 학습된 estimator/model 객체가 아닙니다.",
    }


def load_random_forest_notebook_results() -> dict:
    """랜덤포레스트 노트북에 저장된 실행 출력들을 추출합니다."""
    if not NOTEBOOK_PATH.exists():
        return {
            "model_group": "random_forest",
            "artifact_type": "notebook_results_bundle",
            "outputs": [],
            "note": "랜덤포레스트 노트북을 찾을 수 없습니다.",
        }
    with NOTEBOOK_PATH.open("r", encoding="utf-8") as file:
        notebook = json.load(file)

    outputs = []
    for cell_index, cell in enumerate(notebook.get("cells", [])):
        for output in cell.get("outputs", []):
            outputs.append({
                "cell_index": cell_index,
                "execution_count": cell.get("execution_count"),
                "output": output,
            })
    return {
        "model_group": "random_forest",
        "artifact_type": "notebook_results_bundle",
        "outputs": outputs,
        "notebook_path": str(NOTEBOOK_PATH),
        "model_file_saved_in_this_workspace": False,
        "note": (
            "노트북 실행 결과만 포함합니다. 노트북 출력에 다른 컴퓨터의 저장 경로가 보이지만 "
            "실제 랜덤포레스트 모델 파일은 이 폴더에 없습니다."
        ),
    }


def dump_pickle(name: str, value: dict) -> dict:
    """결과 사전을 표준 pickle 형식으로 저장하고 manifest 정보를 반환합니다."""
    destination = OUTPUT_DIR / name
    with destination.open("wb") as file:
        pickle.dump(value, file, protocol=pickle.HIGHEST_PROTOCOL)
    return {
        "file": name,
        "artifact_type": value.get("artifact_type"),
        "source": value.get("source_folder", value.get("notebook_path", "")),
        "size_bytes": destination.stat().st_size,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for model_name, folder_name in RESULT_FOLDERS.items():
        manifest_rows.append(
            dump_pickle(f"{model_name}_results.pkl", load_result_folder(folder_name))
        )
    manifest_rows.append(
        dump_pickle("random_forest_results.pkl", load_random_forest_notebook_results())
    )

    gbm_model_files = [
        path for path in GBM_MODELS_DIR.rglob("*")
        if path.is_file() and path.name != ".gitkeep"
    ] if GBM_MODELS_DIR.exists() else []
    gbm_report_files = [
        path for path in GBM_REPORTS_DIR.rglob("*")
        if path.is_file() and path.name != ".gitkeep"
    ] if GBM_REPORTS_DIR.exists() else []
    manifest = {
        "saved_result_bundles": manifest_rows,
        "churn_gbm": {
            "model_files_found": [str(path) for path in gbm_model_files],
            "report_files_found": [str(path) for path in gbm_report_files],
            "status": (
                "generated model/results found"
                if gbm_model_files or gbm_report_files
                else "no generated model or report files found; source code only"
            ),
        },
        "important": (
            "These pickle files contain evaluation tables/notebook outputs, not fitted model objects. "
            "The source CSV/JSON/PNG/IPYNB files are left unchanged."
        ),
    }
    (OUTPUT_DIR / "conversion_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
