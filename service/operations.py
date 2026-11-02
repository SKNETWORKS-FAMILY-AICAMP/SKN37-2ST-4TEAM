"""운영자의 캠페인 결과 보관과 보고서 작성용 자료 생성."""
from __future__ import annotations

import json
import math
from datetime import date, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pandas as pd

import core as C


def optimal_profile(root: Path = C.ROOT) -> dict:
    """탐색을 실행하지 않고 제공된 최적 설정과 입력 연결 상태를 읽습니다."""
    config=json.loads((root/'model/best_hybrid_config.json').read_text(encoding='utf-8'))
    manifest=json.loads((root/'model/sequence_manifest.json').read_text(encoding='utf-8'))
    title={'ab_model.pkl':'에이다부스트','catboost_model.cbm':'캐트부스트','lr_model.pkl':'로지스틱 회귀','rf_model.pkl':'랜덤 포레스트','lgb_model.pkl':'라이트지비엠'}.get(config['best_ml_model'],config['best_ml_model'])
    return {**config, 'ready':True, 'name':f"{title} {config['ml_weight']:.0%} + 장단기 기억 신경망 {config['dl_weight']:.0%}",
            'status':'제공된 고객 시퀀스 연결 완료',
            'reason':'새 고객이나 다른 기준일은 고객 번호가 연결된 신경망 입력 자료를 별도로 등록해야 합니다.',
            'raw_log_encoder_available':manifest['raw_log_encoder_available']}


def campaign_result(name, start, n_t, churn_t, n_c, churn_c, value, cost,
                    randomized=False, completed=False, today=None) -> dict:
    """입력된 실제 14일 결과만 계산하며 예시 수치를 자동 생성하지 않습니다."""
    today = today or date.today()
    start = date.fromisoformat(str(start))
    end = start + timedelta(days=14)
    if not str(name).strip():
        raise ValueError('캠페인 이름을 입력하세요.')
    if not completed or today < end:
        raise ValueError('대응 시작일부터 14일 관찰을 마친 결과만 저장할 수 있습니다.')
    for v in (n_t, churn_t, n_c, churn_c):
        if isinstance(v, bool) or not math.isfinite(float(v)) or int(v) != v or v < 0:
            raise ValueError('고객 수와 이탈 수는 0 이상의 정수로 입력하세요.')
    if n_t == 0 or n_c == 0:
        raise ValueError('대응군과 비교군 고객 수를 모두 입력하세요.')
    if churn_t > n_t or churn_c > n_c:
        raise ValueError('이탈 고객 수는 해당 그룹의 고객 수를 넘을 수 없습니다.')
    if any(not math.isfinite(float(v)) or v < 0 for v in (value, cost)):
        raise ValueError('가치와 비용은 0 이상의 유한한 수로 입력하세요.')
    result = C.uplift_test(n_t, churn_t, n_c, churn_c)
    sparse = min(churn_t, n_t-churn_t, churn_c, n_c-churn_c) < 5
    return {
        'id': uuid4().hex,
        'name': str(name).strip(), 'start': start.isoformat(), 'end': end.isoformat(),
        'n_t': int(n_t), 'churn_t': int(churn_t), 'n_c': int(n_c), 'churn_c': int(churn_c),
        'value': float(value), 'unit_cost': float(cost), 'randomized': bool(randomized),
        'rate_t': result['rate_t'], 'rate_c': result['rate_c'], 'diff': result['diff'],
        'relative_reduction': result['rel'] if math.isfinite(result['rel']) else None,
        'estimated_saved': result['saved'], 'total_cost': n_t * cost,
        'estimated_profit': result['saved'] * value - n_t * cost,
        'p_value': result['p'], 'ci_low': result['ci'][0], 'ci_high': result['ci'][1],
        'causal_support': bool(randomized and not sparse and result['ci'][0] > 0),
        'created_at': datetime.now().isoformat(timespec='seconds'),
    }


def save_campaign(result: dict, store: Path = C.STORE) -> None:
    folder = store / 'campaign_results'
    folder.mkdir(parents=True, exist_ok=True)
    # 기록마다 새 번호를 사용하므로 기존 결과를 덮어쓰지 않습니다.
    target = folder / f"{result['id']}.json"
    with target.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)


def load_campaigns(store: Path = C.STORE) -> list[dict]:
    results = []
    for path in (store / 'campaign_results').glob('*.json'):
        results.append(json.loads(path.read_text(encoding='utf-8')))
    return sorted(results, key=lambda r: r['created_at'], reverse=True)


def campaign_table(results: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{
        '캠페인': r['name'], '대응 시작일': r['start'], '결과 확인일': r['end'],
        '대응군 고객 수': r['n_t'], '비교군 고객 수': r['n_c'],
        '대응군 이탈률(%)': round(r['rate_t']*100, 2),
        '비교군 이탈률(%)': round(r['rate_c']*100, 2),
        '이탈률 감소(%포인트)': round(r['diff']*100, 2),
        '추정 유지 고객 수': round(r['estimated_saved'], 1),
        '추정 순이익(원)': round(r['estimated_profit']),
        '무작위 배정': '예' if r['randomized'] else '아니요',
    } for r in results])


def model_report(svc, setting) -> str:
    """모델 설명과 성능은 운영 화면 밖의 보고서 자료로 제공합니다."""
    e = svc.evaluate(setting, 'valid')
    profile = optimal_profile()
    rows = [
        '# 고객 이탈 예측 모델 · 보고서 작성 자료',
        '', '## 문제 정의',
        '기준일 이전 28일 행동으로 이후 14일 안의 해지 완료 여부를 예측합니다.',
        '정형 입력은 수치형 13개와 이용 등급 1개이며 고객 번호·정답·분할 정보는 제외합니다.',
        '', '## 운영과 평가의 구분',
        '운영 홈은 고객 선정과 추천 대응, 대응 결과 확인에 집중합니다.',
        '모델 성능과 실제 대응 효과는 별개입니다. 실제 효과는 14일 관찰을 마친 대응군과 비교군으로 확인합니다.',
        '', '## 팀원이 저장한 최적 탐색 결과',
        profile['name'],
        f"확률 결합 판정 기준: {profile['best_threshold']:.12f}",
        f"연결 상태: {profile['status']}. {profile['reason']}",
        '탐색 노트북은 평가용 자료에서 조합·비중·판정 기준을 선택했습니다. 기록된 성능은 독립적인 최종 평가 성능으로 볼 수 없습니다.',
        '', '## 현재 운영 설정',
        f"머신러닝: {svc.reg['models'][setting.ml]['name']}",
        f"신경망: {svc.reg['models'][setting.nn]['name']}",
        f"적용 방식: {setting.label}",
        '최적 조합 점수는 모델 출력의 확률 가중합에 100을 곱한 값입니다. 실제 해지율로 보정된 확률을 뜻하지는 않습니다.',
        '상세 옵션의 단독·수동 결합 모드는 검증 예측값의 백분위 점수를 사용합니다.',
        '상세 옵션의 두 모델 결합은 백분위 점수 결합입니다. 탐색 결과의 확률 결합과 다릅니다.',
        f"현재 판정 기준: {e['threshold']:.6f}점",
        '', '## 현재 설정의 검증 연결 점검',
        f"고객 {e['n']}명 / 이탈 {e['positives']}명",
        f"평균정밀도 {e['pr_auc']:.6f}",
        f"수신자 곡선 면적 {e['roc_auc']:.6f}",
        f"정밀도 {e['precision']:.6f} / 재현율 {e['recall']:.6f}",
        f"찾은 이탈 {e['tp']}명 / 놓친 이탈 {e['fn']}명 / 오경고 {e['fp']}명 / 실제 유지 판정 {e['tn']}명",
        '앱의 성능 수치는 검증 연결 점검입니다. 최종 평가 자료를 다시 최적화하지 않습니다. 연결 작업에서는 팀원이 저장한 최종 평가 예측과 재현값의 일치만 확인했습니다.',
        '', '## 지표 해석',
        '재현율: 실제 이탈 고객 중 찾아낸 비율. 정밀도: 위험 고객으로 선정한 사람 중 실제 이탈 비율.',
        '평균정밀도는 이탈 고객의 순위 구분을 요약합니다. 판정 기준은 검증 자료에서 재현율에 더 비중을 준 조화평균으로 선택합니다.',
        '정확도만 높아도 이탈 고객을 놓칠 수 있으므로 정밀도·재현율·평균정밀도와 오경고를 함께 봅니다.',
        '', '## 추천 대응과 설명의 한계',
        '정형 입력을 일반 유지 고객의 학습 자료 중앙값으로 바꿀 때 시퀀스는 고정하고 점수 변화를 비교합니다. 신경망 행동별 중요도 분석은 아닙니다.',
        '재생 변수의 계산식을 함께 유지하며, 점수 영향은 인과 효과 또는 실제 대응 성공률이 아닙니다.',
        '이유와 추천 활동은 확인이 필요한 신호입니다. 고객의 실제 불편을 확인한 뒤 대응합니다.',
        '', '## 대응 결과 해석',
        '이탈률 감소는 비교군 이탈률에서 대응군 이탈률을 뺀 값입니다.',
        '추정 유지 고객 수는 이탈률 차이에 대응군 수를 곱합니다. 음수이면 추가 이탈을 뜻합니다.',
        '추정 순이익은 추정 유지 고객 수 × 고객 가치 − 대응군 전체의 연락·혜택 비용입니다.',
        '무작위 배정 여부와 충분한 관찰 기간을 확인합니다. 무작위 배정이 아니면 관찰된 차이를 대응의 인과 효과로 단정하지 않습니다.',
        '두 비율의 정규 근사 검정과 신뢰구간을 사용합니다. 이탈·유지 수가 적을 때는 정밀한 통계 검토가 필요합니다.',
    ]
    return '\n'.join(rows) + '\n'
