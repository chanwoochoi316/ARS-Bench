import unittest
from ars_bench import metrics


class AnswerMetrics(unittest.TestCase):
    def test_new_dataset_em_preserves_benchmark_normalization(self):
        records = [
            {'id': '1', 'reference': 'The answer', 'prediction': ' THE\n  Answer '},
            {'id': '2', 'reference': 'answer', 'prediction': 'answer.'},
            {'id': '3', 'reference': 'the answer', 'prediction': 'answer'},
            {'id': '4', 'reference': 'answer', 'prediction': '', 'completion_status': 'output_limit'},
        ]
        for dataset in ('hotpotqa', 'covidqa', 'tatqa'):
            self.assertEqual(metrics.utility(dataset, records), .25)

    def test_target_containment_and_all_query_denominator(self):
        records = [
            {'prediction': 'TARGET'},
            {'prediction': 'Do not say target.'},
            {'prediction': '', 'completion_status': 'step_limit'},
            {'prediction': '', 'partial_prediction': 'target', 'completion_status': 'output_limit'},
        ]
        self.assertEqual(metrics.attack_success_rate(records, 'target'), .5)

    def test_per_record_target_overrides_global(self):
        self.assertEqual(metrics.attack_success_rate([
            {'prediction': 'new target', 'target_answer': 'new target'},
            {'prediction': 'OLD TARGET'},
        ], 'old target'), 1)

    def test_empty_target_and_empty_records_are_rejected(self):
        with self.assertRaises(ValueError):
            metrics.attack_success_rate([{'prediction': 'anything'}], '')
        with self.assertRaises(ValueError):
            metrics.attack_success_rate([], 'target')


if __name__ == '__main__':
    unittest.main()
