"""확률 결합과 고객·기준일별 입력 연결의 의미를 검사합니다."""
from datetime import date
from pathlib import Path
import io
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).parent))
import core as C
from sequence_runtime import SequenceInputError, SequenceBank, register_inputs


class SequenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.svc=C.load_service()
        cls.setting=C.default_setting(cls.svc)
        cls.frame=C.default_customers(cls.svc.team).head(8)

    def test_exact_probability_blend_and_fixed_threshold(self):
        X=C.prepare_X(self.frame)
        ml=self.svc.models['ab_model'].predict_proba(X)[:,1]
        nn=self.svc.models['lstm_sequence'].predict_proba(X)[:,1]
        actual=self.svc.scores(self.setting,self.frame)
        np.testing.assert_allclose(actual.score/100,.6*ml+.4*nn)
        self.assertAlmostEqual(self.svc.threshold(self.setting),39.8226772012445)
        reordered=self.frame.iloc[::-1]
        np.testing.assert_allclose(self.svc.scores(self.setting,reordered).score,actual.score.iloc[::-1])

    def test_unknown_changed_and_wrong_date_blocked(self):
        unknown=self.frame.copy();unknown.loc[unknown.index[0],'user_id']='새고객'
        changed=self.frame.copy();changed.loc[changed.index[0],'help_count']+=1
        day=self.frame.copy();day.attrs['as_of']='2026-10-30'
        for frame in [unknown,changed,day]:
            with self.assertRaises(SequenceInputError):self.svc.scores(self.setting,frame)

    def test_operational_updates_do_not_replace_validation_reference(self):
        model=self.svc.models['lstm_sequence']
        X=C.prepare_X(self.frame)
        X.attrs['reference_input']=True
        expected=model.predict_proba(X)
        uid=str(self.frame.iloc[0].user_id)
        key=('2018-11-01',uid)
        original=model.bank.entries[key]
        try:
            model.bank.entries[key]=np.zeros(50,dtype=np.int64)
            model.cache.clear()
            np.testing.assert_allclose(model.predict_proba(X),expected)
        finally:
            model.bank.entries[key]=original
            model.cache.clear()

    def test_external_inputs_persist_by_customer_and_day(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);data=root/'data/processed';data.mkdir(parents=True)
            for split in ['train','valid','test']:
                pd.DataFrame({'user_id':[split], 'song_count':[10], 'level_last':['free']}).to_csv(data/f'{split}_20181101_v1.csv',index=False)
                np.save(data/f'{split}.npy',np.zeros((1,50),dtype=np.int64))
            source=io.BytesIO()
            np.savez_compressed(source,user_id=np.array(['새고객']),tokens=np.ones((1,50),dtype=np.int64))
            source.seek(0)
            self.assertEqual(register_inputs(source,date(2026,10,30),root/'service_data'),1)
            bank=SequenceBank(root,['song_count','level_last'])
            np.testing.assert_array_equal(bank.lookup(['새고객'],'2026-10-30'),np.ones((1,50)))
            with self.assertRaises(SequenceInputError):bank.lookup(['새고객'],'2018-11-01')

    def test_invalid_external_token_bounds(self):
        with tempfile.TemporaryDirectory() as folder:
            source=io.BytesIO()
            np.savez_compressed(source,user_id=np.array(['새고객']),tokens=np.full((1,50),50,dtype=np.int64))
            source.seek(0)
            with self.assertRaises(SequenceInputError):register_inputs(source,'2026-10-30',Path(folder))


if __name__=='__main__':unittest.main()
