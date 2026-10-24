"""Checks for temporal leakage, preprocessing leakage and scoring semantics."""

import importlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import solution
from benchmark import Experiment, make_estimator, score_predictions, temporal_split
from metric import precision_at_recall


class BenchmarkTests(unittest.TestCase):
    def test_solution_cli_writes_to_requested_output(self):
        train = pd.DataFrame({'cookie_id': ['a', 'b'], 'target': [0, 1]})
        test = pd.DataFrame({'cookie_id': ['c']})
        features = pd.DataFrame(np.zeros((3, 110)), index=['a', 'b', 'c'])
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory) / 'inputs'
            output = Path(directory) / 'outputs' / 'submission.csv'
            with patch('sys.argv', ['solution.py', '--data-dir', str(data_dir), '--output', str(output)]), \
                    patch('solution.pd.read_csv', side_effect=[train, test, pd.DataFrame()]) as read, \
                    patch('solution.build_features', return_value=features), \
                    patch('solution.create_model') as create:
                create.return_value.predict_proba.return_value = np.array([[.75, .25]])
                solution.main()
                paths = [call.args[0] for call in read.call_args_list]
                self.assertEqual(paths, [data_dir / 'train.csv', data_dir / 'test.csv',
                                         data_dir / 'events.csv/events.csv'])
            actual = pd.read_csv(output)
            self.assertEqual(actual.to_dict('list'), {'cookie_id': ['c'], 'score': [.25]})

    def test_import_does_not_read_data_or_train(self):
        with patch('pandas.read_csv') as read:
            importlib.reload(solution)
            read.assert_not_called()

    def test_temporal_split_excludes_unfinished_training_windows(self):
        meta = pd.DataFrame({
            'cookie_id': ['a', 'b', 'c', 'd', 'unfinished', 'future'],
            'window_start_ts': pd.to_datetime(['2026-04-10', '2026-04-11', '2026-04-12',
                                               '2026-04-13', '2026-04-11', '2026-04-14']),
            'window_end_ts': pd.to_datetime(['2026-04-11', '2026-04-12', '2026-04-13',
                                             '2026-04-14', '2026-04-13', '2026-04-15']),
            'target': [0, 1, 0, 1, 1, 0],
        })
        train, valid = temporal_split(meta, '2026-04-12', '2026-04-14')
        self.assertEqual(list(train), [0, 1])
        self.assertEqual(list(valid), [2, 3])
        self.assertFalse(set(train) & set(valid))
        meta.loc[3, 'cookie_id'] = 'a'
        with self.assertRaises(ValueError):
            temporal_split(meta, '2026-04-12', '2026-04-14')

    def test_scaler_statistics_come_only_from_training(self):
        train = pd.DataFrame({'x': [0., 1., 2., 3.], 'z': [1., 0., 1., 0.]})
        estimator = make_estimator(Experiment('lr', 'lr', 'all110'), threads=1)
        estimator.fit(train, [0, 0, 1, 1])
        scaler = estimator.named_steps['standardscaler']
        before = scaler.mean_.copy()
        estimator.predict_proba(pd.DataFrame({'x': [10000.], 'z': [-10000.]}))
        np.testing.assert_array_equal(scaler.mean_, before)
        np.testing.assert_allclose(before, train.mean())

    def test_rounding_and_ties_match_submission_metric(self):
        self.assertEqual(score_predictions([0, 1], [0.500001, 0.500002])['precision_at_recall70'], .5)
        self.assertEqual(precision_at_recall([1, 1, 0, 0], [.5] * 4), .5)
        self.assertEqual(precision_at_recall([1, 0, 1, 1, 1], [5, 4, 3, 2, 1]), .8)

    def test_features_ignore_events_at_and_after_window_end(self):
        meta = pd.DataFrame({
            'cookie_id': ['a'], 'cookie_created_at': pd.to_datetime(['2026-04-01']),
            'window_start_ts': pd.to_datetime(['2026-04-10']),
            'window_end_ts': pd.to_datetime(['2026-04-11']),
        })
        events = pd.DataFrame({
            'cookie_id': ['a', 'a'], 'event_ts': pd.to_datetime(['2026-04-10 01:00', '2026-04-10 01:01']),
            'eid': [1, 2], 'event_name': ['item_view', 'search_results_view'],
            'platform': ['desktop', 'desktop'], 'user_agent': ['firefox/test', 'firefox/test'],
            'item_id': [10, 11], 'item_category': ['x', 'y'], 'item_location': ['a', 'b'],
            'search_query': ['x', 'y'], 'search_page': [1, 2],
            'pointer_x': [10., 20.], 'pointer_y': [15., 25.],
        })
        expected = solution.build_features(events, meta)
        extra = events.copy()
        extra['event_ts'] = pd.to_datetime(['2026-04-11', '2026-04-12'])
        extra['pointer_x'] = 1e9
        actual = solution.build_features(pd.concat([events, extra], ignore_index=True), meta)
        pd.testing.assert_frame_equal(expected, actual)
        self.assertEqual(actual.shape, (1, 110))
        self.assertTrue(np.isfinite(actual.to_numpy()).all())


if __name__ == '__main__':
    unittest.main()
