"""편집기의 실행 버튼이나 더블클릭으로 앱을 시작하는 진입점."""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
PACKAGES=['streamlit','numpy','pandas','sklearn','joblib','plotly','lightgbm','catboost','torch']
REQUIRED=['service/app.py','service/core.py','service/ui.py','models.py',
          'model/best_hybrid_config.json','model/sequence_manifest.json',
          'model/machine_learning/ab_model.pkl','model/machine_learning/lr_model.pkl',
          'model/deep_learning/lstm_model_weights.pth','model/deep_learning/mlp_service.joblib',
          'data/processed/customer_features_20181101_v1_with_split.csv']
REQUIRED += [f'data/processed/{s}{suffix}' for s in ['train','valid','test'] for suffix in ['.npy','_20181101_v1.csv']]


def check_environment():
    errors=[]
    if sys.version_info[:2]!=(3,12):errors.append('이 배포본은 파이썬 3.12 환경에서 확인했습니다. 처음설치 파일로 실행 환경을 준비하세요.')
    missing=[name for name in PACKAGES if importlib.util.find_spec(name) is None]
    if missing:errors.append('필요한 프로그램이 없습니다: '+', '.join(missing))
    absent=[name for name in REQUIRED if not (ROOT/name).is_file()]
    if absent:errors.append('필요한 파일이 없습니다:\n'+'\n'.join(absent))
    return errors


def build_command(port=8501):
    return [sys.executable,'-m','streamlit','run',str(ROOT/'service/app.py'),
            '--server.address','127.0.0.1','--server.port',str(port),'--browser.gatherUsageStats','false',
            '--server.showEmailPrompt','false']


def main(argv=None):
    parser=argparse.ArgumentParser(description='고객 이탈 대응 앱 실행')
    parser.add_argument('--check',action='store_true',help='설치와 필수 파일만 확인합니다.')
    parser.add_argument('--port',type=int,default=8501,help='이미 사용 중이면 다른 포트를 지정하세요.')
    args=parser.parse_args(argv)
    errors=check_environment()
    if errors:
        print('\n\n'.join(errors))
        print('\n압축을 모두 푼 뒤 처음설치 파일을 실행하세요.')
        return 1
    if args.check:
        print('실행 환경과 필수 파일 확인 완료')
        return 0
    if not 1<=args.port<=65535:
        print('접속 번호는 1부터 65535까지 지정하세요.')
        return 1
    print(f'앱을 시작합니다: http://127.0.0.1:{args.port}/')
    print('이 창을 닫거나 중지 키를 누르면 앱도 종료됩니다.')
    env=os.environ.copy()
    env['PYTHONUTF8']='1'
    try:return subprocess.call(build_command(args.port),cwd=ROOT,env=env)
    except KeyboardInterrupt:return 0


if __name__=='__main__':sys.exit(main())
