# Transformer v1

v1의 고객 행동 시퀀스와 기존 train/valid/test 분할로 Transformer 이탈 분류기를 학습합니다. 검증 데이터에서 F2 최대값으로 최적 epoch와 기본 임계값을 선택하며, 재현율 70%·80% 기준도 기록합니다. 결과 TXT 사본은 `music_churn_data_v1/result`에 저장됩니다.

프로젝트 루트에서 PyTorch가 설치된 인터프리터로 실행:

```powershell
python "music_churn_data_v1/deep_learning/transformer/transformer_v1.py"
```
