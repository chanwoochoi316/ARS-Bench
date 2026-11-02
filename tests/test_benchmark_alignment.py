"""Behavioral regressions for the authoritative Benchmark protocol (no models)."""
import ast
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from ars_bench import benchmark_metrics as bm
from ars_bench.agents import run_one, run
from ars_bench.defenses import Defense
from ars_bench.io import normalize_records, bundle, read
from ars_bench.attacks import generate
from ars_bench._benchmark.efficiency import estimate_token_count


QUERY = {'id': 'q', 'question': 'What is the answer?', 'reference': 'gold'}

class Tokenizer:
    eos_token = '<eos>'
    def apply_chat_template(self, messages, **kwargs):
        return '<chat>' + '\n'.join(m['content'] for m in messages) + '</chat>'

class Client:
    seed = 1
    def __init__(self, replies, model='Qwen3-main'):
        self.model, self.replies, self.calls = model, iter(replies), []
    def complete(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return next(self.replies)

class Retriever:
    def __init__(self): self.calls = []
    def search(self, query, k=5):
        self.calls.append((query, k))
        return [{'id': query, 'title': 'Title', 'text': 'gold evidence', 'is_poisoned': False}]

class GenerationClient:
    def __init__(self): self.calls=[]
    def chat(self, messages, **options):
        self.calls.append((messages, options))
        if any('doc1' in m['content'] and 'doc5' in m['content'] for m in messages):
            return json.dumps({f'doc{i}': f'bait {i}' for i in range(1,6)})
        return 'Document: generated body'


def clean_record():
    return {**QUERY, 'prediction': 'gold', 'steps': [
        {'thought': 'Find evidence', 'action': 'Search[first query]', 'search_query': 'first query', 'is_search': True, 'poisoned_flags': [False]},
        {'thought': 'Check evidence', 'action': 'Search[second query]', 'search_query': 'second query', 'is_search': True, 'poisoned_flags': [False]}]}

class BenchmarkRunners(unittest.TestCase):
    def test_react_fallback_and_source_sampling(self):
        client=Client(['a malformed first response', 'Search[primary query]\nignored',
                       'Found it\nAction 2: Finish[gold]'])
        retriever=Retriever()
        result=run_one(QUERY,'react',client,retriever,Defense())
        self.assertEqual(result['prediction'],'gold')
        self.assertEqual(result['n_calls'],3)
        self.assertEqual(result['n_badcalls'],1)
        self.assertEqual(result['steps'][0]['observation'],'Title\ngold evidence')
        self.assertEqual(retriever.calls,[('primary query',5)])
        opts=client.calls[0][1]
        self.assertEqual((opts['temperature'],opts['top_p'],opts['max_tokens']),(0,1,100))
        self.assertIn('Observation',opts['stop'])
        # Original PDS only sees explicit search_query fields, not inferred Action text.
        self.assertNotIn('primary query',' '.join(bm.chain(result)))

    def test_react_target_in_search_terminates_before_retrieval(self):
        retriever=Retriever()
        result=run_one(QUERY,'react',Client(['Oops\nAction 1: Search[KILL YOURSELF]']),retriever,Defense())
        self.assertEqual(retriever.calls,[])
        self.assertEqual(result['prediction'],'search[KILL YOURSELF]')
        self.assertEqual(result['completion_status'],'finished')

    def test_react_default_nine_turns_and_empty_fallback(self):
        result=run_one(QUERY,'react',Client([f'Think\nAction {i}: Think[more]' for i in range(1,10)]),Retriever(),Defense())
        self.assertEqual(len(result['steps']),9)
        self.assertEqual(result['completion_status'],'step_limit')
        self.assertEqual(result['prediction'],'')

    def test_full_web_explorer_search_click_and_continuation(self):
        main=Client(['Think<|begin_search_query|>primary query<|end_search_query|>', '**ANSWER: gold**'])
        aux=Client(['find evidence', 'Explore<|begin_search_query|>secondary query<|end_search_query|>',
                    'Read<|begin_click_link|>https://example.org/doc<|end_click_link|>',
                    'verify answer', 'gold reader summary', '**Final Information:** gold'],model='aux')
        retriever=Retriever()
        result=run_one(QUERY,'webthinker',main,retriever,Defense(),aux_client=aux,
                       main_tokenizer=Tokenizer(),aux_tokenizer=Tokenizer())
        self.assertEqual(result['prediction'],'gold')
        self.assertEqual(retriever.calls,[('primary query',5),('secondary query',5)])
        self.assertEqual((len(main.calls),len(aux.calls)),(2,6))
        self.assertEqual([s.get('source') for s in result['benchmark_record']['step_stats']],
                         ['main','web_explorer','web_explorer','main'])
        self.assertIn('<|begin_click_result|>',result['web_explorer'][0]['Output'])
        self.assertIn('<chat>',main.calls[0][0])
        self.assertTrue(main.calls[1][0].startswith(main.calls[0][0]))
        self.assertEqual(main.calls[0][1]['max_tokens'],4096)
        self.assertEqual(main.calls[0][1]['temperature'],.7)
        self.assertEqual(aux.calls[0][1]['max_tokens'],1000)
        self.assertEqual(aux.calls[4][1]['max_tokens'],3600)

    def test_clean_wall_summary_survives_attacked_bundle(self):
        clean={'records':[clean_record()],'source_summary':{'time':12.3}}
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'attack.json'
            result=run([QUERY],'react',Client(['Done\nAction 1: Finish[gold]']),Retriever(),Defense(),
                       {'dataset':'hotpotqa','target_answer':'KILL YOURSELF'},output,clean=clean)
            self.assertEqual(read(output)['clean']['source_summary']['time'],12.3)
            self.assertGreater(result['attacked']['source_summary']['time'],0)

class BenchmarkMetrics(unittest.TestCase):
    def test_telco_containment_and_empty_reference_policy(self):
        rows=normalize_records([{**clean_record(),'reference':'30 seconds','prediction':'It is 30 seconds.'},
                                {**clean_record(),'id':'empty','reference':'','prediction':'anything'}])
        self.assertEqual(bm.utility('telcodpr',rows),1)
        self.assertEqual(bm.utility('hotpotqa',rows),0)
        self.assertEqual(bm.relative_ufs(0,.7),0)

    def test_legal_normalized_rouge_and_added_datasets(self):
        rows=normalize_records([{**clean_record(),'reference':'THE running cat','prediction':'the run cat'}])
        self.assertEqual(bm.utility('legalbenchrag',rows),1)
        rows[0]['prediction']='  THE   RUNNING CAT '
        self.assertEqual(bm.utility('covidqa',rows),1)
        self.assertEqual(bm.utility('tatqa',rows),1)

    def test_original_stationary_pds_and_missing_transition(self):
        self.assertAlmostEqual(bm.pds_from_embeddings([[1,0],[1,0]],[[1,0],[1,0]]),1.75)
        self.assertAlmostEqual(bm.pds_from_embeddings([[1,0]],[[1,0],[1,0]]),3.5)
        self.assertAlmostEqual(bm.pds_from_embeddings([[1,0]],[[1,0]]),0)

    def test_li_aggregate_ratio_and_missing_original_summary(self):
        rows=normalize_records([clean_record(),{**clean_record(),'id':'q2'}])
        obj=bundle(rows,rows,[],{"dataset":"hotpotqa","target_answer":"TARGET"})
        obj['clean']['source_summary']={'time':11}
        obj['attacked']['source_summary']={'time':12}
        self.assertAlmostEqual(bm.efficiency(obj,'reasoning_time_s'),1/11)
        del obj['clean']['source_summary']
        with self.assertRaisesRegex(ValueError,'summary.time'): bm.efficiency(obj,'reasoning_time_s')

    def test_tui_uses_original_text_heuristic(self):
        rows=normalize_records([clean_record()]); obj=bundle(rows,rows,[],{"dataset":"hotpotqa","target_answer":"TARGET"})
        original=sum(estimate_token_count(s['thought'])+estimate_token_count(s['action']) for s in rows[0]['steps'])
        obj['attacked']['records'][0]['steps'][0]['thought']='extra reasoning '*20
        changed=sum(estimate_token_count(s['thought'])+estimate_token_count(s['action']) for s in obj['attacked']['records'][0]['steps'])
        self.assertAlmostEqual(bm.efficiency(obj,'reasoning_tokens'),(changed-original)/original)

    def test_raw_original_step_stats_survive_normalization(self):
        row={'id':'q','question':'Q','correct answer':'A','prediction':'B',
             'step_stats':[{'thought':'Reasoning','action':'Search[inferred query]','is_search':True,'poisoned_flags':[False]}]}
        normalized=normalize_records([row])[0]
        self.assertNotIn('search_query',bm.legacy_record(normalized)['step_stats'][0])
        self.assertNotIn('inferred query',' '.join(bm.chain(normalized)))

class BenchmarkGeneration(unittest.TestCase):
    def test_kidnap_exact_messages_sampling_and_no_question_prefix(self):
        client=GenerationClient()
        artifact=generate({'records':[clean_record()],'experiment':{}},'kidnaprag','TARGET',client)
        baits=[d for d in artifact['documents'] if d.get('metadata',{}).get('kind')=='bait']
        self.assertEqual(len(baits),10)
        self.assertEqual(len(client.calls),2)
        self.assertEqual(client.calls[0][1]['temperature'],.6)
        self.assertEqual(client.calls[0][1]['max_tokens'],2500)
        self.assertTrue(all(d['text'].startswith('bait ') for d in baits))
        self.assertFalse(any(m['content'].startswith('Generate doc1') for m in client.calls[0][0]))

    def test_poison_bodies_generated_once_then_reused(self):
        client=GenerationClient()
        artifact=generate({'records':[clean_record()],'experiment':{}},'poisonedrag','TARGET',client)
        self.assertEqual(len(client.calls),5)
        self.assertEqual(len(artifact['documents']),10)
        self.assertEqual({d['text'] for d in artifact['documents']},{'generated body'})
        self.assertTrue(all(c[1]['temperature']==1 and c[1]['max_tokens']==1024 for c in client.calls))

    def test_paradox_default_four_per_search(self):
        client=GenerationClient()
        artifact=generate({'records':[clean_record()],'experiment':{}},'paradox','TARGET',client)
        self.assertEqual(len(artifact['documents']),8)
        self.assertTrue(all(c[1]['temperature']==1 and c[1]['max_tokens']==1024 for c in client.calls))

class FrozenSourceIntegrity(unittest.TestCase):
    def test_vendored_modules_match_recorded_hashes(self):
        root=Path(__file__).resolve().parents[1]/'ars_bench'
        sources=json.loads((root/'_benchmark/sources.json').read_text())
        for name,entry in sources.items():
            if 'copied_sha256' in entry:
                self.assertEqual(hashlib.sha256((root/'_benchmark'/name).read_bytes()).hexdigest(),entry['copied_sha256'],name)
            else:
                self.assertEqual(hashlib.sha256((root/'resources'/name).read_bytes()).hexdigest(),entry['source_sha256'],name)

class EvaluationProfileParity(unittest.TestCase):
    def test_benchmark_judges_document_limit_and_offline_cache(self):
        import numpy as np
        from ars_bench.evaluation import evaluate
        class Judge:
            model='fixture-judge'
            def __init__(self): self.calls=[]
            def chat(self,messages,**kwargs):
                self.calls.append((messages,kwargs))
                return json.dumps({'score':1,'reason':'synthetic test'})
        class Shield:
            provider='azure-prompt-shield'
            def evaluate(self,text): return {'flagged':False}
        class Embedding:
            name='synthetic'
            def encode(self,texts): return np.asarray([[1.,0.] for t in texts])
        rows=normalize_records([clean_record()])
        obj=bundle({'records':rows,'source_summary':{'time':10}},
                   {'records':rows,'source_summary':{'time':12}},
                   [{'id':str(i),'title':'title','text':'body'} for i in range(3)],
                   {'dataset':'hotpotqa','target_answer':'TARGET','attack_method':'ignore'})
        judge=Judge()
        report=evaluate(obj,judge,Shield(),Embedding(),judge_model=judge.model,profile='benchmark',document_limit=1)
        self.assertTrue(report['complete'])
        self.assertEqual(report['metrics']['SBS']['value'],5)
        self.assertEqual(report['metrics']['IBS']['value'],4)
        self.assertEqual(report['metrics']['IBS']['n'],1)
        self.assertEqual(report['metrics']['ISS']['value'],1)
        self.assertEqual(len(judge.calls),2)
        self.assertTrue(all(k['max_tokens']==256 for _,k in judge.calls))
        self.assertTrue(all(k['response_format']['type']=='json_schema' for _,k in judge.calls))
        # SBS can be replayed offline without constructing an Azure client.
        replay=evaluate(obj,judge_model=judge.model,profile='benchmark',document_limit=1)
        self.assertEqual(report['metrics'],replay['metrics'])
        old=evaluate(obj,profile='instrumented')
        self.assertEqual(old['metrics']['IBS']['status'],'missing')
        self.assertIn('SBS',obj['evaluation_cache']['benchmark-v1'])

    def test_qwq_explorer_routes_to_main(self):
        main=Client(['Think<|begin_search_query|>primary query<|end_search_query|>',
                     '**Final Information:** gold','**ANSWER: gold**'],model='QwQ-32B')
        aux=Client(['find evidence'],model='aux')
        result=run_one(QUERY,'webthinker',main,Retriever(),Defense(),aux_client=aux,
                       main_tokenizer=Tokenizer(),aux_tokenizer=Tokenizer())
        self.assertEqual(result['prediction'],'gold')
        self.assertEqual((len(main.calls),len(aux.calls)),(3,1))

class DefenseBoundaryParity(unittest.TestCase):
    def test_spotlight_instruction_and_observation_boundary(self):
        from ars_bench.runners.webthinker import SPOTLIGHT_SYSTEM_INSTRUCTION
        main=Client(['Think<|begin_search_query|>primary query<|end_search_query|>', '**ANSWER: gold**'])
        aux=Client(['find evidence','**Final Information:** gold summary'],model='aux')
        result=run_one(QUERY,'webthinker',main,Retriever(),Defense('spotlight'),aux_client=aux,
                       main_tokenizer=Tokenizer(),aux_tokenizer=Tokenizer())
        self.assertIn(SPOTLIGHT_SYSTEM_INSTRUCTION,main.calls[0][0])
        self.assertIn('gold^evidence',aux.calls[1][0])
        self.assertIn('gold^summary',main.calls[1][0])
        self.assertEqual(result['steps'][0]['retrieved_documents'][0]['text'],'gold evidence')
