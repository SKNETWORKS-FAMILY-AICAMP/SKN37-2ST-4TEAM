# Saved model artifacts

이 폴더에는 machine_learning 하위의 학습 모델 파일을 .pkl 확장자로 모았습니다. 각 파일명은 원래 모델명과 버전을 유지합니다.

기존 파일은 joblib.dump로 저장되어 있어, 파일 확장자를 .pkl로 바꾸어 복사했습니다. 따라서 불러올 때는 표준 pickle.load 대신 joblib.load를 사용하세요.

~~~python
import joblib

artifact = joblib.load(r"C:\project2_music\music_churn_data_v1\machine_learning\pkl\catboost_v1.pkl")
model = artifact["model"]
~~~

CatBoost, XGBoost, LightGBM 등 원래 학습에 사용한 라이브러리와 호환되는 Python 환경이 필요합니다. model_manifest.csv에서 원본 파일과 복사본을 확인할 수 있습니다.
