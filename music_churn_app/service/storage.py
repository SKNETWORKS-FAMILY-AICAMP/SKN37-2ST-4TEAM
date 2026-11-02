"""고객별 관찰 자료를 기준일과 고객 번호로 저장하고 이력을 보존합니다."""
from datetime import date, datetime, timezone
from pathlib import Path
from contextlib import contextmanager
import json
import sqlite3

import pandas as pd
from validation import FEATURES, InputError, prepare_customers


@contextmanager
def connect(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=30)
    db.execute('CREATE TABLE IF NOT EXISTS snapshots (as_of TEXT NOT NULL, customer_id TEXT NOT NULL, payload TEXT NOT NULL, updated_at TEXT NOT NULL, PRIMARY KEY(as_of,customer_id))')
    db.execute('CREATE TABLE IF NOT EXISTS revisions (as_of TEXT NOT NULL, customer_id TEXT NOT NULL, payload TEXT NOT NULL, replaced_at TEXT NOT NULL)')
    try:
        with db:
            yield db
    finally:
        db.close()


def merge_customers(existing, incoming, policy='keep'):
    if policy not in ['keep', 'update']:
        raise ValueError('중복 고객 처리 기준을 확인해 주세요.')
    if 'user_id' not in incoming:
        raise InputError('고객 추가·갱신에는 고객 번호 열이 필요합니다. 임시 번호로 기존 고객을 덮어쓰지 않습니다.')
    old, new = prepare_customers(existing), prepare_customers(incoming)
    overlap = set(old.user_id) & set(new.user_id)
    result = pd.concat([old, new], ignore_index=True).drop_duplicates('user_id', keep='first' if policy == 'keep' else 'last')
    return prepare_customers(result), {'추가': len(new) - len(overlap), '갱신': len(overlap) if policy == 'update' else 0,
                                      '유지': len(overlap) if policy == 'keep' else 0}


def snapshot_dates(path):
    with connect(path) as db:
        rows = db.execute('SELECT as_of,COUNT(*) FROM snapshots GROUP BY as_of ORDER BY as_of DESC').fetchall()
    return [{'기준일': row[0], '고객 수': row[1]} for row in rows]


def save_snapshot(path, as_of, frame, policy='keep'):
    day = date.fromisoformat(str(as_of)).isoformat()
    if policy not in ['keep', 'update']:
        raise ValueError('중복 고객 처리 기준을 확인해 주세요.')
    data = prepare_customers(frame)
    now = datetime.now(timezone.utc).isoformat()
    stats = {'추가': 0, '갱신': 0, '유지': 0}
    with connect(path) as db:
        for row in data.to_dict('records'):
            customer_id = str(row['user_id'])
            previous = db.execute('SELECT payload FROM snapshots WHERE as_of=? AND customer_id=?', (day, customer_id)).fetchone()
            payload = json.dumps({c: row[c] for c in FEATURES}, ensure_ascii=False, allow_nan=False)
            if previous:
                if policy == 'keep':
                    stats['유지'] += 1
                    continue
                db.execute('INSERT INTO revisions VALUES (?,?,?,?)', (day, customer_id, previous[0], now))
                stats['갱신'] += 1
            else:
                stats['추가'] += 1
            db.execute('INSERT INTO snapshots VALUES (?,?,?,?) ON CONFLICT(as_of,customer_id) DO UPDATE SET payload=excluded.payload,updated_at=excluded.updated_at', (day, customer_id, payload, now))
    return stats


def load_snapshot(path, as_of):
    day = date.fromisoformat(str(as_of)).isoformat()
    with connect(path) as db:
        rows = db.execute('SELECT customer_id,payload FROM snapshots WHERE as_of=? ORDER BY customer_id', (day,)).fetchall()
    if not rows:
        raise InputError('선택한 기준일에 저장된 고객이 없습니다.')
    return prepare_customers(pd.DataFrame([{'user_id': customer_id, **json.loads(payload)} for customer_id, payload in rows]))
