"""CLI-to-HTTP smoke test using a local synthetic server, never a real model API."""
import contextlib
import http.server
import io
import json
import tempfile
import threading
import unittest
from pathlib import Path
from ars_bench.cli import main
from ars_bench.io import read, write


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/v1/completions":
            prompt = body["prompt"]
            if prompt.endswith("Thought 1:"):
                text = "Search for the code\nAction 1: Search[code]"
            else:
                text = "I have the result\nAction 2: Finish[gold]"
            result = {"choices": [{"text": text}]}
        elif self.path == "/v1/chat/completions":
            result = {"choices": [{"message": {"content": '{"score": 0, "reason": "synthetic API fixture"}'}}]}
        else:
            result = {"documentsAnalysis": [{"attackDetected": False}]}
        data = json.dumps(result).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers(); self.wfile.write(data)


class HTTPWorkflow(unittest.TestCase):
    def test_clean_generate_attack_cli_and_portable_result(self):
        try:
            server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        except PermissionError:
            self.skipTest("Current host sandbox does not allow binding a local test HTTP server")
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
                p = Path(d); endpoint = f"http://127.0.0.1:{server.server_port}/v1"
                write(p/"q.json", [{"id": "1", "question": "What is code?", "reference": "gold"}])
                write(p/"c.json", [{"id": "c", "title": "code", "text": "code is gold"}])
                common = ["run", "--dataset", "hotpotqa", "--architecture", "react", "--model", "synthetic-model",
                          "--base-url", endpoint, "--queries", str(p/"q.json"), "--corpus", str(p/"c.json"), "--retriever", "bm25"]
                self.assertEqual(main(common + ["--output", str(p/"clean.json")]), 0)
                self.assertEqual(main(["generate", "--clean", str(p/"clean.json"), "--attack", "ignore",
                                       "--target-answer", "A DIFFERENT TARGET", "--output", str(p/"docs.json")]), 0)
                self.assertEqual(main(common + ["--attack-documents", str(p/"docs.json"), "--defense", "spotlight",
                                                "--output", str(p/"result.json")]), 0)
                obj = read(p/"result.json")
                self.assertEqual(obj["experiment"]["target_answer"], "A DIFFERENT TARGET")
                self.assertTrue(obj["documents"])
                self.assertEqual(obj["clean"]["records"][0]["prediction"], "gold")
                self.assertEqual(obj["attacked"]["records"][0]["prediction"], "gold")
                # Remove input dependencies: validation/evaluation need only the exported bundle.
                (p/"q.json").unlink(); (p/"c.json").unlink(); (p/"clean.json").unlink(); (p/"docs.json").unlink()
                self.assertEqual(main(["validate", str(p/"result.json")]), 0)
                self.assertEqual(main(["evaluate", str(p/"result.json"), "--offline", "--output", str(p/"report.json"), "--require-all"]), 2)
                report = read(p/"report.json")
                self.assertEqual(report["metrics"]["UFS"]["value"], 0)
                self.assertEqual(report["metrics"]["TUI"]["status"], "missing")
                self.assertTrue((p/"report.csv").exists())
        finally:
            server.shutdown(); server.server_close(); thread.join()


if __name__ == "__main__":
    unittest.main()
