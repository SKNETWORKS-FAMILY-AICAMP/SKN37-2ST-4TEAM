"""운영 홈 이동, 고객 대응 결과 계산·보관과 보고서 분리를 검사합니다."""
from datetime import date
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).parent))
import core as C
import operations as O


class OperationTests(unittest.TestCase):
    def record(self, **changes):
        values = dict(name='추천 개선', start=date(2018,11,1), n_t=400, churn_t=40,
                      n_c=400, churn_c=60, value=120000, cost=5000,
                      randomized=True, completed=True, today=date(2018,11,15))
        values.update(changes)
        return O.campaign_result(**values)

    def test_campaign_arithmetic_and_storage(self):
        record = self.record()
        self.assertAlmostEqual(record['diff'], .05)
        self.assertAlmostEqual(record['estimated_saved'], 20)
        self.assertAlmostEqual(record['estimated_profit'], 400000)
        with tempfile.TemporaryDirectory() as folder:
            O.save_campaign(record, Path(folder))
            loaded = O.load_campaigns(Path(folder))
            self.assertEqual(loaded, [record])
            self.assertEqual(O.campaign_table(loaded).iloc[0]['캠페인'], '추천 개선')
            with self.assertRaises(FileExistsError):
                O.save_campaign(record, Path(folder))

    def test_invalid_or_immature_results(self):
        for changes in [dict(n_t=0), dict(churn_t=401), dict(churn_t=-1), dict(n_c=1.5),
                        dict(name=' '), dict(cost=-1), dict(value=float('nan')),
                        dict(completed=False), dict(today=date(2018,11,14))]:
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError): self.record(**changes)

    def test_no_unsupported_effect_claims(self):
        self.assertFalse(self.record(randomized=False)['causal_support'])
        self.assertFalse(self.record(churn_t=0, churn_c=0)['causal_support'])
        self.assertIsNone(self.record(churn_t=0, churn_c=0)['relative_reduction'])
        self.assertLess(self.record(churn_t=80)['estimated_saved'], 0)
        self.assertFalse(self.record(churn_t=80)['causal_support'])

    def test_profile_and_fixed_fallback(self):
        profile=O.optimal_profile()
        self.assertEqual(profile['best_ml_model'], 'ab_model.pkl')
        self.assertAlmostEqual(profile['ml_weight'], .6)
        self.assertAlmostEqual(profile['best_threshold'], .39822677201244505)
        self.assertTrue(profile['ready'])
        svc=C.load_service()
        with patch.object(C.Service, 'split_scores', side_effect=AssertionError('기본 설정에서 재탐색 금지')):
            setting=C.default_setting(svc)
        self.assertEqual(setting.mode,'optimal')
        self.assertEqual(setting.nn,'lstm_sequence')
        report=O.model_report(svc,setting)
        self.assertIn('정밀도', report)
        self.assertIn('시퀀스', report)
        self.assertNotIn('y_test', svc.probs)

    def test_home_targets_and_detail_navigation(self):
        app=AppTest.from_file(str(Path(__file__).parent/'app.py'),default_timeout=90).run()
        self.assertEqual(len(app.exception),0)

        self.assertEqual(app.radio(key='nav').value,'🏠 홈')
        self.assertEqual(app.radio(key='nav').options,['🏠 홈','👤 고객 상세','💰 캠페인','🗂️ 고객 자료','⚙️ 설정'])
        self.assertEqual([t.label for t in app.tabs],['대응 고객 목록','대응 결과 확인'])
        self.assertIn('추천 대응',app.dataframe[0].value.columns)
        self.assertNotIn('실제 해지(검증용)',app.dataframe[0].value.columns)
        self.assertEqual(len(app.get('plotly_chart')),0)
        self.assertTrue(any('아직 기록한 대응 결과가 없습니다' in item.value for item in app.info))
        button=next(b for b in app.button if b.label=='고객 상세 열기')
        button.click().run()
        self.assertEqual(app.radio(key='nav').value,'👤 고객 상세')
        self.assertTrue(app.text_input(key='detail_uid').value)
        self.assertEqual(len(app.exception),0)

    def test_actual_result_form_saves_and_reloads(self):
        save=O.save_campaign
        load=O.load_campaigns
        with tempfile.TemporaryDirectory() as folder:
            store=Path(folder)
            with patch.object(O,'save_campaign',side_effect=lambda r:save(r,store)), \
                 patch.object(O,'load_campaigns',side_effect=lambda:load(store)):
                app=AppTest.from_file(str(Path(__file__).parent/'app.py'),default_timeout=90).run()
                next(w for w in app.text_input if w.label=='캠페인 이름').set_value('추천 개선 실제 결과')
                for label,value in [('대응군 고객 수',400),('대응군 이탈 고객 수',40),
                                    ('비교군 고객 수',400),('비교군 이탈 고객 수',60)]:
                    next(w for w in app.number_input if w.label==label).set_value(value)
                next(w for w in app.checkbox if '같은 14일' in w.label).check()
                next(b for b in app.button if b.label=='결과 계산하고 저장').click().run()
                self.assertEqual(len(app.exception),0)
                self.assertEqual(len(load(store)),1)
                self.assertTrue(any('대응군 이탈률' in str(frame.value.columns) for frame in app.dataframe))



if __name__=='__main__': unittest.main()
