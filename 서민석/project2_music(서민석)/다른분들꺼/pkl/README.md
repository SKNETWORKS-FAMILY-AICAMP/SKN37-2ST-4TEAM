# Other model result pickle bundles

이 폴더의 .pkl 파일은 다른분들꺼에 이미 저장돼 있던 평가 결과를 모델별로 묶은 표준 Python pickle입니다.

- logistic_regression_results.pkl: 결과 CSV 표, 실험 요약 JSON, 그래프 PNG 바이트
- mlp_results.pkl: 결과 CSV 표, 실험 요약 JSON, 학습 그래프 PNG 바이트
- svm_results.pkl: 결과 CSV 표, 실험 요약 JSON, 그래프 PNG 바이트
- random_forest_results.pkl: 랜덤포레스트 노트북에 저장된 실행 출력

이 파일들은 학습된 모델 객체 파일이 아닙니다. 이 폴더에는 로지스틱 회귀·MLP·SVM의 저장된 estimator가 없고, 랜덤포레스트 노트북이 언급한 저장 경로의 모델 파일도 현재 워크스페이스에 없습니다. churn-gbm은 소스 코드만 있고 생성된 모델/보고서가 없습니다.

원본 결과 파일과 노트북은 그대로 유지했습니다. 변환 내용을 conversion_manifest.json에서 확인할 수 있습니다.

결과 묶음은 표준 pickle이므로 pandas와 pickle로 읽을 수 있습니다.

~~~python
import pickle

with open("다른분들꺼/pkl/logistic_regression_results.pkl", "rb") as file:
    results = pickle.load(file)
tables = results["tables"]
summary = results["summaries"]
~~~

변환 재실행:

~~~powershell
python "다른분들꺼/pkl/convert_model_results_to_pkl.py"
~~~
