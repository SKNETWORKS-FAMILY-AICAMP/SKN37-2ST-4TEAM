"""기존 디자인을 유지한 통합 서비스의 예측·저장·화면을 확인합니다."""
from pathlib import Path
import sys,tempfile,unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import core as C


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.object(C,'train_all',side_effect=AssertionError('앱에서 학습 금지')):
            cls.svc=C.load_service()
        cls.setting=C.default_setting(cls.svc)
        cls.customers=C.default_customers(cls.svc.team).head(10)

    def test_saved_models_and_validation_only(self):
        self.assertEqual(len(self.svc.models),7)
        self.assertEqual(set(self.svc.team.loc[self.svc.team.user_id.isin(C.default_customers(self.svc.team).user_id),'split']),{'valid'})
        self.assertNotIn('y_test',self.svc.probs)
        with self.assertRaises(ValueError):self.svc.evaluate(self.setting,'test')

    def test_model_selection_and_blend(self):
        keys=[k for k,v in self.svc.reg['models'].items() if v['kind']=='ml']
        nn=self.svc.reg['default']['nn']
        a=self.svc.scores(C.Setting(keys[0],nn,mode='ml'),self.customers)
        b=self.svc.scores(C.Setting(keys[-1],nn,mode='ml'),self.customers)
        self.assertFalse(np.allclose(a.score,b.score))
        mixed=self.svc.scores(C.Setting(keys[0],nn,ml_weight=.3),self.customers)
        np.testing.assert_allclose(mixed.score,.3*mixed.ml_score+.7*mixed.nn_score)

    def test_invalid_records_are_blocked(self):
        for column,value in [('user_id',''),('user_id',np.nan),('song_count',np.inf),('active_days',99),
                             ('active_days',1.5),('ad_count',-1),('song_count_change',99999),('churn','정답아님')]:
            d=self.customers.head(1).copy();d[column]=value
            with self.subTest(column=column,value=str(value)):
                self.assertTrue(C.check_upload(d)[0])

    def test_existing_pipeline_prediction_preserved(self):
        import joblib
        saved=joblib.load(C.ROOT/'model/deep_learning/mlp_service.joblib')['pipeline']
        adapter=self.svc.models['mlp_service']
        np.testing.assert_allclose(saved.predict_proba(self.customers[C.FEATURES]),adapter.predict_proba(C.prepare_X(self.customers)),atol=1e-9)

    def test_consistent_changes(self):
        x=C.prepare_X(self.customers)
        x['recent_7d_song_count']=10000
        changed=C.consistent_song_features(x,'recent_7d_song_count')
        np.testing.assert_array_equal(changed.song_count_change,changed.recent_7d_song_count-changed.previous_7d_song_count)
        self.assertTrue((changed.song_count>=changed.recent_7d_song_count+changed.previous_7d_song_count).all())
        row=self.customers.iloc[0]
        self.svc.what_if(self.setting,row,{'recent_7d_song_count':0})

    def test_backup_snapshots_and_restore(self):
        with tempfile.TemporaryDirectory() as directory:
            store=Path(directory)
            C.save_customers(self.customers,store,as_of='2018-11-01')
            changed=self.customers.copy();changed['help_count']=999
            C.save_customers(changed,store,as_of='2018-11-02')
            self.assertEqual(C.load_data_date(store),'2018-11-02')
            self.assertEqual(len(list((store/'backups').glob('*.csv'))),1)
            db=store/'customer_snapshots.sqlite3'
            C.snapshot_storage.save_snapshot(db,'2018-11-01',self.customers)
            C.snapshot_storage.save_snapshot(db,'2018-11-02',changed)
            old=C.snapshot_storage.load_snapshot(db,'2018-11-01')
            self.assertFalse(old.help_count.eq(999).any())
            C.reset_customers(store)
            self.assertFalse((store/C.CURRENT).exists())
            self.assertEqual(len(C.load_customers(self.svc.team,store)),2721)

    def test_merge_and_unknown_outcome(self):
        new=self.customers.head(1).copy();new['help_count']=99
        merged,counts=C.merge_customers(self.customers,new,'append','update')
        self.assertEqual(counts['updated'],1)
        row=merged[merged.user_id==new.iloc[0].user_id].iloc[0]
        self.assertEqual(row.help_count,99)
        self.assertTrue(pd.isna(row.churn))

    def test_campaign_can_choose_no_contact(self):
        table,best=self.svc.campaign(self.setting,100,1,100000,.1)
        self.assertEqual(best['contacts'],0)
        self.assertEqual(best['profit'],0)

    def test_all_five_screens(self):
        app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=90).run()
        self.assertEqual(len(app.exception),0,[(e.message) for e in app.exception])
        pages=['🏠 홈','👤 고객 상세','💰 캠페인','🗂️ 고객 자료','⚙️ 설정']
        for page in pages[1:]:
            app.radio(key='nav').set_value(page).run()
            with self.subTest(page=page):
                self.assertEqual(len(app.exception),0,[(e.message) for e in app.exception])


if __name__=='__main__':unittest.main()
