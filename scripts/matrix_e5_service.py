"""Shared local E5 retrieval over the already-built pinned dense indexes."""
import sys,json,threading,functools,http.server,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(ROOT))
import torch
from ars_bench.encoder import SimpleEncoder
from ars_bench.io import read,digest
E5='/home/work/Redteaming/data1/VIDEO_HALLUCINATION/hf_cache/hub/models--intfloat--e5-large-v2/snapshots/f169b11e22de13617baa190a028a32f3493550b6'
torch.set_num_threads(4)
encoder=SimpleEncoder(E5,device='cuda:2',batch_size=1)
indexes={}
for ds in ('covidqa','tatqa'):
 p=ROOT/'indexes'/ds/'e5';m=read(p/'manifest.json');docs=read(p/'documents.json')
 assert m['model']==E5 and m['documents_digest']==digest(docs)
 files=sorted(p.glob('embedding-shard-*.pt'),key=lambda x:int(x.stem.rsplit('-',1)[1]))
 vectors=torch.cat([torch.load(x,map_location='cpu',weights_only=True) for x in files]).to('cuda:2')
 assert len(docs)==len(vectors);indexes[ds]=(docs,vectors)
lock=threading.Lock()
@functools.lru_cache(maxsize=2048)
def search(ds,query,k):
 with lock:
  docs,vectors=indexes[ds]
  q=encoder.encode_queries([query]).to('cuda:2',dtype=vectors.dtype)
  scores,ids=torch.topk((q@vectors.T)[0],min(k,len(docs)))
  return [{**docs[i],'score':float(score)} for score,i in zip(scores.tolist(),ids.tolist())]
class Handler(http.server.BaseHTTPRequestHandler):
 def log_message(self,*a):pass
 def do_GET(self):
  b=json.dumps({'status':'ready','snapshot':E5,'device':'cuda:2','documents':{k:len(v[0]) for k,v in indexes.items()}}).encode();self.send_response(200);self.end_headers();self.wfile.write(b)
 def do_POST(self):
  try:
   payload=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
   result=search(payload['dataset'],payload['query'],int(payload.get('k',5)))
   b=json.dumps(result,ensure_ascii=False).encode();self.send_response(200);self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(b)
  except Exception:
   traceback.print_exc();self.send_response(500);self.end_headers()
print('E5_SERVICE_READY',flush=True)
http.server.ThreadingHTTPServer(('127.0.0.1',8134),Handler).serve_forever()
