"""저장된 모델 불러오기, 입력 검사, 고객별 예측을 담당합니다."""
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd


LABELS = {
    'active_days': '활동 일수', 'session_count': '이용 세션 수',
    'song_count': '곡 재생 횟수', 'days_since_last_activity': '마지막 활동 이후 경과 일수',
    'recent_7d_song_count': '최근 7일 곡 재생 횟수',
    'previous_7d_song_count': '이전 7일 곡 재생 횟수',
    'song_count_change': '최근·이전 7일 재생 횟수 차이',
    'thumbs_up_count': '좋아요 횟수', 'thumbs_down_count': '싫어요 횟수',
    'playlist_add_count': '재생목록 추가 횟수', 'ad_count': '광고 이벤트 횟수',
    'error_count': '오류 이벤트 횟수', 'help_count': '도움말 이용 횟수',
    'level_last': '마지막 이용 등급',
}
FEATURES = list(LABELS)
NUMERIC = FEATURES[:-1]
COUNTS = [c for c in NUMERIC if c not in ['days_since_last_activity', 'song_count_change']]


class InputError(ValueError):
    """고객 데이터에서 발견한 수정 가능한 문제입니다."""


def read_customers(content: bytes) -> pd.DataFrame:
    for encoding in ['utf-8-sig', 'cp949']:
        try:
            return pd.read_csv(BytesIO(content), dtype=str, encoding=encoding)
        except UnicodeDecodeError:
            continue
        except (pd.errors.ParserError, pd.errors.EmptyDataError) as error:
            raise InputError('표를 읽을 수 없습니다. 쉼표로 구분한 파일인지 확인해 주세요.') from error
    raise InputError('문자 인코딩을 읽을 수 없습니다. 파일을 한글 또는 유니코드 형식으로 저장해 주세요.')


def prepare_customers(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        raise InputError('고객 행이 없습니다. 한 명 이상의 이용 기록을 넣어 주세요.')
    data = frame.copy()
    data.columns = [str(c).strip() for c in data.columns]
    if data.columns.duplicated().any():
        raise InputError('이름이 중복된 열이 있습니다.')
    missing = [LABELS[c] for c in FEATURES if c not in data]
    if missing:
        raise InputError('필요한 항목이 없습니다: ' + ', '.join(missing))
    if 'user_id' not in data:
        data['user_id'] = [f'고객{i + 1:06d}' for i in range(len(data))]
    if data.user_id.isna().any():
        raise InputError('고객 번호가 비어 있습니다.')
    data['user_id'] = data.user_id.astype(str).str.strip()
    if data.user_id.eq('').any() or data.user_id.duplicated().any():
        raise InputError('고객 번호가 비어 있거나 중복됐습니다. 고객당 한 행인지 확인해 주세요.')
    for column in NUMERIC:
        value = pd.to_numeric(data[column], errors='coerce')
        bad = value.isna() | ~np.isfinite(value)
        if bad.any():
            rows = ', '.join(str(i + 2) for i in np.flatnonzero(bad.to_numpy())[:5])
            raise InputError(f'{LABELS[column]}에 숫자가 아닌 값 또는 빈값이 있습니다. 파일의 {rows}번째 줄을 확인해 주세요.')
        data[column] = value.astype(float)
    for column in COUNTS:
        if ((data[column] < 0) | (data[column] % 1 != 0)).any():
            raise InputError(f'{LABELS[column]}는 0 이상의 정수여야 합니다.')
    if not data.active_days.between(1, 28).all():
        raise InputError('활동 일수는 관찰 기간에 맞게 1일부터 28일 사이여야 합니다.')
    if (data.session_count < 1).any():
        raise InputError('이용 세션 수는 1 이상이어야 합니다.')
    if not ((data.days_since_last_activity >= 0) & (data.days_since_last_activity < 28)).all():
        raise InputError('마지막 활동 이후 경과 일수는 0 이상, 28 미만이어야 합니다.')
    if (data.song_count_change % 1 != 0).any():
        raise InputError('재생 횟수 차이는 정수여야 합니다. 감소를 뜻하는 음수는 사용할 수 있습니다.')
    if not np.allclose(data.song_count_change, data.recent_7d_song_count - data.previous_7d_song_count, atol=0, rtol=0):
        raise InputError('재생 횟수 차이가 최근 7일 횟수에서 이전 7일 횟수를 뺀 값과 다릅니다.')
    if ((data.recent_7d_song_count + data.previous_7d_song_count) > data.song_count).any():
        raise InputError('최근·이전 7일 재생 횟수의 합이 전체 28일 재생 횟수보다 큽니다.')
    data['level_last'] = data.level_last.astype('string').str.strip().str.lower()
    if not data.level_last.isin(['free', 'paid']).all():
        raise InputError('이용 등급은 입력 양식의 무료·유료 표기 두 종류만 사용할 수 있습니다.')
    # 정답과 분할 열 등은 입력에 포함하지 않습니다.
    return data[['user_id'] + FEATURES].reset_index(drop=True)

