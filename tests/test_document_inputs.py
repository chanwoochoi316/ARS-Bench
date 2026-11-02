"""Plain document exchange, legacy input, and deferred clean pairing."""
import contextlib
import http.server
import io
import json
import tempfile
import threading
import unittest
from pathlib import Path
from ars_bench.attack_inputs import resolve_attack
from ars_bench.cli import main
from ars_bench.io import read, write, load_bundle
from ars_bench.retrieval import validate_clean_corpus, load_corpus
from test_http import Handler


def document(**metadata):
    return {'_id': 'p', 'title': 'code', 'text': 'poison text',
            'metadata': {'attack_method': 'ours납치5회', 'target_answer': 'TARGET', **metadata}}


class DocumentInputs(unittest.TestCase):
    def test_jsonl_contains_only_individual_documents(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'documents.jsonl'
            rows=[document(),{**document(),'_id':'p2'}]
            write(path,rows)
            self.assertEqual(len(path.read_text().splitlines()),2)
            self.assertEqual(json.loads(path.read_text().splitlines()[0]),rows[0])
            self.assertEqual(read(path),rows)
            self.assertNotIn('clean',read(path)[0])
            self.assertEqual(resolve_attack(read(path))['attack_method'],'kidnaprag')

    def test_legacy_documents_accept_explicit_attack_and_target(self):
        row=document();row['metadata']={'target_answer':'OLD TARGET'}
        result=resolve_attack([row],method='completion_real',target='NEW TARGET')
        self.assertEqual(result['attack_method'],'fake_completion')
        self.assertEqual(result['target_answer'],'NEW TARGET')
        self.assertEqual(result['documents'][0]['text'],'poison text')
        self.assertIsNone(result['clean'])

    def test_missing_and_conflicting_metadata_require_flags(self):
        with self.assertRaisesRegex(ValueError,'--attack'):
            resolve_attack([{'id':'p','title':'','text':'body'}])
        with self.assertRaisesRegex(ValueError,'--target-answer'):
            resolve_attack([document(),{**document(target_answer='OTHER'),'_id':'p2'}])
        self.assertEqual(resolve_attack([document(),document(target_answer='OTHER')],target='EXPLICIT')['target_answer'],'EXPLICIT')
        with self.assertRaisesRegex(ValueError,'empty'):resolve_attack([])
        with self.assertRaisesRegex(ValueError,'JSON object'):resolve_attack(['bad row'])

    def test_clean_corpus_validation_detects_content_order_and_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'corpus.jsonl';write(path,[{'id':'a','title':'t','text':'clean'},{'id':'b','title':'u','text':'other'}])
            indexed=load_corpus(path);validate_clean_corpus(path,indexed)
            with self.assertRaisesRegex(ValueError,'does not match'):validate_clean_corpus(path,list(reversed(indexed)))
            with self.assertRaisesRegex(ValueError,'does not match'):validate_clean_corpus(path,indexed[:1])
            indexed[0]['text']='changed'
            with self.assertRaisesRegex(ValueError,'does not match'):validate_clean_corpus(path,indexed)


class PlainDocumentHTTPWorkflow(unittest.TestCase):
    def test_generate_jsonl_run_with_or_without_clean_and_evaluate_later(self):
        server=http.server.ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
                root=Path(tmp);endpoint=f'http://127.0.0.1:{server.server_port}/v1'
                queries=root/'queries.json';corpus=root/'corpus.jsonl';clean=root/'clean.json';poison=root/'poisoned.jsonl'
                write(queries,[{'id':'q','question':'What is code?','reference':'gold'}])
                write(corpus,[{'id':'c','title':'code','text':'code is gold'}])
                common=['run','--dataset','hotpotqa','--architecture','react','--model','synthetic-model',
                        '--base-url',endpoint,'--queries',str(queries),'--corpus',str(corpus),'--retriever','bm25']
                self.assertEqual(main(common+['--output',str(clean)]),0)
                self.assertEqual(main(['generate','--clean',str(clean),'--attack','ignore','--target-answer','TARGET','--output',str(poison)]),0)
                docs=read(poison);self.assertIsInstance(docs,list);self.assertEqual(len(docs),5)
                self.assertTrue(all('text' in d and 'clean' not in d for d in docs))
                attack=root/'attack-only.json'
                self.assertEqual(main(common+['--attack-documents',str(poison),'--output',str(attack)]),0)
                trace=read(attack);self.assertEqual(trace['artifact_version'],'ars-trace/1')
                self.assertEqual(trace['experiment']['attack_method'],'ignore')
                self.assertEqual(trace['experiment']['target_answer'],'TARGET')
                self.assertEqual(len(trace['documents']),5)
                with self.assertRaisesRegex(ValueError,'--clean'):load_bundle(attack)
                wrong_clean=read(clean)
                wrong_clean['experiment']['model']='different-model'
                with self.assertRaisesRegex(ValueError,'Clean model differs'):load_bundle(attack,wrong_clean)
                paired=load_bundle(attack,read(clean))
                self.assertEqual(paired['clean']['source_summary'],read(clean)['source_summary'])
                self.assertEqual(paired['attacked']['source_summary'],trace['source_summary'])
                # Pair during inference; queries can be taken directly from the clean result.
                paired_output=root/'paired.json'
                no_query=common[:common.index('--queries')]+common[common.index('--corpus'):]
                self.assertEqual(main(no_query+['--attack-documents',str(poison),'--clean',str(clean),'--output',str(paired_output)]),0)
                self.assertEqual(read(paired_output)['schema_version'],'ars-bench/1')
                self.assertEqual(main(['evaluate',str(attack),'--clean',str(clean),'--offline','--output',str(root/'report.json')]),0)
                report=read(root/'report.json');self.assertEqual(report['metrics']['UFS']['value'],0)
                self.assertEqual(report['metrics']['LI']['status'],'ok')
                # The enriched bundle no longer depends on the source documents or clean file.
                poison.unlink();clean.unlink();queries.unlink();corpus.unlink()
                self.assertEqual(main(['validate',str(root/'report.bundle.json')]),0)
        finally:
            server.shutdown();server.server_close();thread.join()
