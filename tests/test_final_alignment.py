"""Regression checks for unified normal/defended Benchmark settings; no model calls."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from ars_bench.cli import main
from ars_bench.retrieval import build_index, DenseRetriever, resolve_device
from ars_bench.io import read, write
from ars_bench.agents import run_one
from ars_bench.defenses import Defense


class ServerSettings(unittest.TestCase):
    def serve(self, *options):
        with patch('ars_bench.cli.subprocess.call', return_value=0) as launch:
            self.assertEqual(main(['serve', '--model', 'fixture', '--name', 'fixture', *options]), 0)
        return launch.call_args.args[0], launch.call_args.kwargs['env']

    def test_generation_server_matches_context_eager_memory_and_parallelism(self):
        command, env = self.serve('--profile', 'generation', '--gpus', '0,1,2,3', '--tensor-parallel-size', '4')
        self.assertEqual(command[command.index('--max-model-len')+1], '100000')
        self.assertEqual(command[command.index('--tensor-parallel-size')+1], '4')
        self.assertEqual(command[command.index('--gpu-memory-utilization')+1], '0.9')
        self.assertIn('--enforce-eager', command)
        self.assertIn('--trust-remote-code', command)
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '0,1,2,3')
        command, _ = self.serve('--profile', 'generation', '--tensor-parallel-size', '2')
        self.assertEqual(command[command.index('--max-model-len')+1], '50000')

    def test_explicit_agent_loading_and_memory_settings(self):
        command, _ = self.serve('--dtype', 'bfloat16', '--trust-remote-code', '--gpu-memory-utilization', '0.95')
        self.assertEqual(command[command.index('--dtype')+1], 'bfloat16')
        self.assertEqual(command[command.index('--gpu-memory-utilization')+1], '0.95')
        self.assertIn('--trust-remote-code', command)

    def test_agent_server_retains_existing_defaults(self):
        command, _ = self.serve()
        self.assertEqual(command[command.index('--max-model-len')+1], '30000')
        self.assertNotIn('--enforce-eager', command)


class RetrievalSettings(unittest.TestCase):
    def test_index_model_is_reused_for_queries_and_poisoned_documents(self):
        import torch
        calls = []
        class Encoder:
            def __init__(self, model, device, batch_size):
                calls.append((model, device, batch_size))
                self.batch_size = batch_size
            def encode_corpus(self, docs):
                assert self.batch_size == 1024
                return torch.tensor([[1., 0.] if d['text'] == 'relevant' else [0., 1.] for d in docs])
            def encode_queries(self, queries):
                return torch.tensor([[1., 0.] for q in queries])
        with tempfile.TemporaryDirectory() as tmp, patch('ars_bench.encoder.SimpleEncoder', Encoder):
            root = Path(tmp)
            write(root/'corpus.jsonl', [{'id': 'a', 'title': '', 'text': 'unrelated'}])
            build_index(root/'corpus.jsonl', root/'index', device='cpu')
            manifest = read(root/'index/manifest.json')
            self.assertEqual(manifest['model'], 'intfloat/e5-large-v2')
            self.assertEqual(manifest['encoding']['batch_size'], 1024)
            self.assertEqual(manifest['encoding']['max_length'], 512)
            retriever = DenseRetriever(root/'index', device='cpu', attack_documents=[{'id':'p','title':'','text':'relevant'}])
            self.assertEqual(retriever.search('needle', k=1)[0]['id'], 'p')
            self.assertEqual(calls, [('intfloat/e5-large-v2', 'cpu', 1024), ('intfloat/e5-large-v2', 'cpu', 32)])
            with self.assertRaisesRegex(ValueError, 'must match'):
                DenseRetriever(root/'index', model='other-encoder', device='cpu')
            torch.save(torch.tensor([[1., 0.]], dtype=torch.float16), root/'index/embedding-shard-0.pt')
            with self.assertRaisesRegex(ValueError, 'Index precision'):
                DenseRetriever(root/'index', device='cpu')

    def test_auto_device_uses_cuda_when_available(self):
        with patch('torch.cuda.is_available', return_value=True):
            self.assertEqual(resolve_device('auto'), 'cuda')
        with patch('torch.cuda.is_available', return_value=False):
            self.assertEqual(resolve_device('auto'), 'cpu')


class UnifiedDefenseSettings(unittest.TestCase):
    def test_web_normal_and_defended_tokenizer_routes(self):
        from test_benchmark_alignment import Client, Retriever, QUERY
        class Tokenizer:
            eos_token = '<eos>'
            def __init__(self, tag): self.tag = tag
            def apply_chat_template(self, messages, **options):
                return self.tag + ''.join(m['content'] for m in messages)
        for method, expected in [('none', 'MAIN'), ('spotlight', 'AUX')]:
            client = Client(['**ANSWER: gold**'], model='Qwen3-main')
            result = run_one(QUERY, 'webthinker', client, Retriever(), Defense(method),
                aux_client=Client([], model='Qwen2.5-32B-Instruct'),
                main_tokenizer=Tokenizer('MAIN'), aux_tokenizer=Tokenizer('AUX'))
            self.assertEqual(result['prediction'], 'gold')
            self.assertTrue(client.calls[0][0].startswith(expected))

    def test_react_spotlight_instruction_and_observation_keep_raw_documents(self):
        from test_benchmark_alignment import Client, Retriever, QUERY
        client = Client(['Find it\nAction 1: Search[primary query]', 'Done\nAction 2: Finish[gold]'])
        result = run_one(QUERY, 'react', client, Retriever(), Defense('spotlight'))
        self.assertIn('special character ‘‘ˆ’’ between every word', client.calls[0][0])
        self.assertIn('Observation 1: Title\ngold^evidence', client.calls[1][0])
        self.assertEqual(result['steps'][0]['retrieved_documents'][0]['text'], 'gold evidence')
        self.assertEqual(result['steps'][1]['observation'], 'Episode^finished,^reward^=^1\n')

    def test_piguard_backend_matches_architecture(self):
        with patch('ars_bench._benchmark.react_piguard.PIGuardDefender') as react, patch('ars_bench._benchmark.piguard.PIGuardDefender') as web:
            Defense('piguard', model_path='fixture', architecture='react')
            react.assert_called_once_with(model_path='fixture', device='cpu')
            web.assert_not_called()
            Defense('piguard', model_path='fixture', architecture='webthinker')
            web.assert_called_once_with(model_path='fixture', device='cpu')


if __name__ == '__main__':
    unittest.main()
