"""고객 번호와 기준일을 보존하는 제공 시퀀스 연결. 행동 번호를 추측하지 않습니다."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


class SequenceInputError(ValueError):
    pass


def check_tokens(ids, tokens, vocab_size=50, length=50):
    ids=np.char.strip(np.asarray(ids).astype(str))
    tokens=np.asarray(tokens)
    if ids.ndim!=1 or len(ids)==0 or any(not v.strip() or v.lower() in ['nan','none','null'] for v in ids) or len(set(ids))!=len(ids):
        raise SequenceInputError('시퀀스 고객 번호는 비어 있지 않은 중복 없는 목록이어야 합니다.')
    if tokens.shape!=(len(ids),length) or tokens.dtype.kind not in 'iu':
        raise SequenceInputError('고객별 시퀀스는 정수 50개이며 고객 번호 목록과 행 수가 같아야 합니다.')
    if np.any(tokens<0) or np.any(tokens>=vocab_size):
        raise SequenceInputError(f'행동 번호는 0부터 {vocab_size-1}까지만 사용할 수 있습니다.')
    return ids, tokens.astype(np.int64)


def fingerprints(frame):
    values=frame.copy()
    if values['level_last'].dtype.kind not in 'biuf':
        values['level_last']=values['level_last'].map({'free':0,'paid':1})
    numeric=values.to_numpy(dtype=np.float64)
    return [hashlib.sha256(row.tobytes()).hexdigest() for row in numeric]


class SequenceBank:
    def __init__(self, root, features, as_of='2018-11-01'):
        self.entries={}
        self.original={}
        self.as_of=as_of
        self.features=features
        for split in ['train','valid','test']:
            frame=pd.read_csv(root/f'data/processed/{split}_20181101_v1.csv',dtype={'user_id':'string'})
            ids,tokens=check_tokens(frame.user_id.to_numpy(),np.load(root/f'data/processed/{split}.npy',allow_pickle=False))
            for uid, token, fingerprint in zip(ids,tokens,fingerprints(frame[features])):
                key=(as_of,uid)
                if key in self.entries: raise SequenceInputError('분할 사이 고객 번호가 중복됩니다.')
                self.entries[key]=token
                self.original[key]=fingerprint
        self.baseline=self.entries.copy()
        sidecars=root/'service_data/sequence_inputs'
        for p in sorted(sidecars.glob('*.npz'),key=lambda f:(f.stat().st_mtime_ns,f.name)):
            with np.load(p,allow_pickle=False) as data:
                ids,tokens=check_tokens(data['user_id'],data['tokens'])
                day=str(data['as_of'].item())
            for uid,token in zip(ids,tokens):
                self.entries[(day,uid)]=token
                # 운영자가 이 기준일용 입력을 별도로 제공한 경우에만 갱신을 허용합니다.
                self.original.pop((day,uid),None)

    def lookup(self, ids, as_of):
        missing=[str(uid) for uid in ids if (as_of,str(uid)) not in self.entries]
        if missing:
            raise SequenceInputError(f'{as_of} 기준 시퀀스가 없는 고객 {len(missing):,}명입니다. 고객 자료에서 해당 기준일의 신경망 입력을 먼저 등록하세요. 예: '+', '.join(missing[:3]))
        return np.stack([self.entries[(as_of,str(uid))] for uid in ids])

    def validate(self, frame, as_of):
        ids=frame.user_id.astype(str).tolist()
        self.lookup(ids,as_of)
        changed=[uid for uid,fp in zip(ids,fingerprints(frame[self.features]))
                 if (as_of,uid) in self.original and self.original[(as_of,uid)]!=fp]
        if changed:
            raise SequenceInputError(f'행동 집계가 바뀐 고객 {len(changed):,}명입니다. 기존 시퀀스를 재사용할 수 없으므로 같은 기준일의 갱신 시퀀스를 등록하세요.')


class SequenceAdapter:
    classes_=np.array([0,1])

    def __init__(self, root, features):
        import torch
        spec=importlib.util.spec_from_file_location('provided_neural_models',root/'models.py')
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.model=module.ChurnLSTM(vocab_size=50,embedding_dim=32,hidden_dim=64)
        self.model.load_state_dict(torch.load(root/'model/deep_learning/lstm_model_weights.pth',map_location='cpu',weights_only=True))
        self.model.eval()
        self.bank=SequenceBank(root,features)
        self.cache={}

    def predict_proba(self, frame):
        import torch
        ids=frame.attrs.get('customer_ids')
        day=frame.attrs.get('as_of',self.bank.as_of)
        reference=bool(frame.attrs.get('reference_input',False))
        scope='reference' if reference else 'current'
        if ids is None or len(ids)!=len(frame):
            raise SequenceInputError('시퀀스 예측에는 고객 번호와 기준일이 필요합니다.')
        missing=list(dict.fromkeys(uid for uid in ids if (scope,day,uid) not in self.cache))
        if missing:
            tokens=np.stack([self.bank.baseline[(day,uid)] for uid in missing]) if reference else self.bank.lookup(missing,day)
            with torch.inference_mode():
                for i in range(0,len(tokens),128):
                    probs=torch.sigmoid(self.model(torch.as_tensor(tokens[i:i+128],dtype=torch.long))).numpy().ravel()
                    for uid,p in zip(missing[i:i+128],probs):self.cache[(scope,day,uid)]=float(p)
        p=np.asarray([self.cache[(scope,day,uid)] for uid in ids])
        return np.column_stack([1-p,p])


def register_inputs(source, as_of, store):
    """번호·순서가 명시된 준비 완료 입력만 보관합니다. 원시 로그 변환은 수행하지 않습니다."""
    from datetime import date
    day=date.fromisoformat(str(as_of)).isoformat()
    with np.load(source,allow_pickle=False) as values:
        ids,tokens=check_tokens(values['user_id'],values['tokens'])
    folder=store/'sequence_inputs'
    folder.mkdir(parents=True,exist_ok=True)
    from uuid import uuid4
    path=folder/f'{day}_{uuid4().hex}.npz'
    # 겹치는 고객이 있으면 실제 등록 순서대로 최신 파일이 우선하도록 수정 시각을 사용합니다.
    np.savez_compressed(path,user_id=ids,tokens=tokens,as_of=np.asarray(day))
    return len(ids)
