"""Opt-in real-container tests. CTF_TEST_DOCKER=1 python3 -m unittest discover -v."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
import uuid

from ctf_agent.cli import DEMO_FLAG, demo_materials
from ctf_agent.config import Config
from ctf_agent.engine import Engine, create_run
from ctf_agent.runner import DockerRunner
from ctf_agent.storage import digest


@unittest.skipUnless(os.environ.get('CTF_TEST_DOCKER') == '1', 'set CTF_TEST_DOCKER=1 for real Docker tests')
class DockerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.root = self.base / 'run'
        cfg = Config(command_seconds=5, expected_flag_sha256=digest(DEMO_FLAG))
        create_run(self.root, 'Crypto: single-byte XOR CTF.', demo_materials(self.base / 'inputs'), 'crypto', [], cfg)
        self.runner = DockerRunner(self.root, cfg, [])
        self.engines = []

    def tearDown(self):
        for engine in self.engines:
            engine.close()
        self.tmp.cleanup()

    def action(self, code, timeout=3, tool_id=None):
        return self.runner.run({'kind': 'run', 'argv': ['python3', '-I', '-c', code], 'timeout': timeout},
                               tool_id or 'test-' + uuid.uuid4().hex, 10)

    def test_timeout_kills_container(self):
        tool_id = 'timeout-' + uuid.uuid4().hex
        result = self.action('import time; print("started", flush=True); time.sleep(60)', 0.8, tool_id)
        self.assertEqual(result['status'], 'timeout', result)
        self.assertLess(result['seconds'], 8)
        state = subprocess.run(['docker', 'inspect', self.runner.container_name(tool_id)], capture_output=True)
        self.assertNotEqual(state.returncode, 0)
        self.assertTrue((self.root / result['log']).exists())

    def test_output_preview_separate_from_full_log_and_limit(self):
        self.runner.config.output_bytes = 100
        result = self.action('print("a"*20000)')
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(len(result['output']), 100)
        self.assertGreater((self.root / result['log']).stat().st_size, 100)
        self.runner.config.log_bytes = 4096
        result = self.action('import sys\nwhile True: sys.stdout.write("x"*8192); sys.stdout.flush()')
        self.assertEqual(result['status'], 'output_limit')
        self.assertEqual((self.root / result['log']).stat().st_size, 4096)

    def test_nonroot_readonly_originals_and_no_network_route(self):
        result = self.action("import os,json\nfrom pathlib import Path\nassert os.getuid()==10001\nassert not Path('/var/run/docker.sock').exists()\nassert len(Path('/proc/net/route').read_text().splitlines())==1\ntry:\n Path('/materials/cipher.hex').write_text('bad')\nexcept OSError:\n print('readonly enforced')\nelse:\n raise RuntimeError('writable originals')")
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['exit_code'], 0, result['output'])
        self.assertIn('readonly enforced', result['output'])

    def test_real_demo_resume_and_candidate_verification(self):
        engine = Engine(self.root)
        self.assertEqual(engine.run(stop_after=1)['status'], 'paused')
        first_tool = engine.s['experiments'][0]['tools'][-1]['id']
        first_log = self.root / engine.s['experiments'][0]['tools'][-1]['result']['log']
        before = first_log.stat().st_mtime_ns
        engine.close()
        engine = Engine(self.root)
        self.engines.append(engine)
        result = engine.run()
        self.assertEqual(result['status'], 'verified')
        self.assertEqual(first_log.stat().st_mtime_ns, before)
        calls = [t for e in engine.s['experiments'] for t in e['tools'] if t['action']['kind'] == 'run']
        self.assertEqual(len(calls), 2)
        self.assertEqual(sum(t['id'] == first_tool for t in calls), 1)
        self.assertTrue(all(h['category'] == 'crypto' for h in engine.s['hypotheses']))

    def test_real_candidate_remains_unverified_without_criterion(self):
        engine = Engine(self.root)
        self.engines.append(engine)
        engine.cfg.expected_flag_sha256 = ''
        engine.s['config']['expected_flag_sha256'] = ''
        self.assertEqual(engine.run()['status'], 'candidate_unverified')

    def test_http_broker_exact_target_and_no_redirect(self):
        name = 'ctf-test-http-' + uuid.uuid4().hex[:12]
        code = """from http.server import BaseHTTPRequestHandler,HTTPServer
class H(BaseHTTPRequestHandler):
 def do_GET(self):
  if self.path=='/redirect':
   self.send_response(302);self.send_header('Location','http://192.0.2.99:9999/denied');self.end_headers()
  else:
   self.send_response(200);self.end_headers();self.wfile.write(b'LOCAL_CTF_FIXTURE')
HTTPServer(('0.0.0.0',8000),H).serve_forever()
"""
        try:
            subprocess.run(['docker', 'run', '-d', '--rm', '--name', name, '--user', '10001:10001',
                            '--cap-drop=ALL', self.runner.config.runner_image, 'python3', '-I', '-c', code],
                           check=True, capture_output=True)
            inspect = json.loads(subprocess.check_output(['docker', 'inspect', name]))[0]
            ip = inspect['NetworkSettings']['Networks']['bridge']['IPAddress']
            self.runner.targets = [{'host': ip, 'port': 8000}]
            # Bounded readiness check of the disposable local test container.
            for _ in range(30):
                ready = subprocess.run(['docker', 'exec', name, 'python3', '-c',
                                        "import socket; socket.create_connection(('127.0.0.1',8000),.2).close()"], capture_output=True)
                if ready.returncode == 0:
                    break
                time.sleep(.1)
            action = {'kind': 'http', 'host': ip, 'port': 8000, 'path': '/', 'tls': False}
            result = self.runner.run(action, 'http-' + uuid.uuid4().hex, 10)
            self.assertEqual(result['exit_code'], 0, result['output'])
            self.assertIn('LOCAL_CTF_FIXTURE', result['output'])
            action['path'] = '/redirect'
            result = self.runner.run(action, 'http-' + uuid.uuid4().hex, 10)
            self.assertEqual(result['exit_code'], 0, result['output'])
            self.assertIn('302', result['output'])
            self.assertNotIn('LOCAL_CTF_FIXTURE', result['output'])
            action['port'] = 8001
            with self.assertRaises(ValueError):
                self.runner.run(action, 'denied', 10)
        finally:
            subprocess.run(['docker', 'rm', '-f', name], capture_output=True)


if __name__ == '__main__':
    unittest.main()
