import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('benchmark_import', Path(__file__).resolve().parents[1] / 'scripts/import_benchmark_kidnaprag.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class HistoricalImport(unittest.TestCase):
    def test_stale_target_is_preserved_but_not_used_as_attack_target(self):
        source = {'details': [{'qid': 'q', 'question': 'Q', 'correct_answer': 'A',
                              'target_answer': 'wrong legacy target', 'llm_answer': '', 'step_stats': []}]}
        record = module.records_from_source(source)[0]
        self.assertEqual(record['legacy_source_target_answer'], 'wrong legacy target')
        self.assertEqual(record['target_answer'], module.TARGET)
        self.assertEqual(record['prediction'], '')
        self.assertEqual(record['completion_status'], 'legacy_no_final_answer')
        self.assertNotIn('reasoning_time_s', record)

    def test_page_info_and_actual_poison_flags_survive_normalization(self):
        source = {'thought': 'T', 'action': 'Search[Q]', 'is_search': True,
                  'search_documents': [{'title': 'D', 'page_info': 'verbatim body'}],
                  'poisoned_flags': [True]}
        step = module.legacy_step(source, 'q', 0)
        self.assertEqual(step['retrieved_documents'][0]['text'], 'verbatim body')
        self.assertIs(step['retrieved_documents'][0]['is_poisoned'], True)
        self.assertEqual(step['search_documents'], source['search_documents'])

    def test_contradictory_poison_evidence_is_rejected(self):
        with self.assertRaises(ValueError):
            module.legacy_step({'thought': '', 'action': 'Search[Q]', 'poisoned_flags': [False],
                                'search_documents': [{'text': 'D', 'is_poisoned': True}]}, 'q', 0)

    def test_source_finished_flag_does_not_invent_a_final_answer(self):
        record = module.records_from_source({'data': [{'id': 'q', 'question': 'Q', 'correct answer': 'A',
                                                     'prediction': '', 'finished': True, 'step_stats': []}]})[0]
        self.assertTrue(record['legacy_source_finished'])
        self.assertEqual(record['completion_status'], 'legacy_no_final_answer')

    def test_source_paths_select_plain_primary_variant(self):
        paths = module.source_paths('react', 'llama8b', 'hotpotqa')
        self.assertIn('adv_targeted_results1', str(paths['attacked']))
        self.assertEqual(paths['attacked'].name, 'hotpotqa_seed1_ours납치5회_llama8b.json')


if __name__ == '__main__':
    unittest.main()
