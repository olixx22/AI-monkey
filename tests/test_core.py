import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from ctf_agent.adapters import FakeModels, LiveModels, ModelError, post_json
from ctf_agent.cli import DEMO_FLAG, demo_materials
from ctf_agent.config import CATEGORIES, Config, initial_vector, parse_target, update_vector
from ctf_agent.engine import Engine, create_run
from ctf_agent.protocol import ACTION, DECISION, REPORT, ProtocolError, obj, parse, string
from ctf_agent.runner import DockerRunner
from ctf_agent.storage import Store, digest, safe_path


class StubRunner:
    """Only preprogrammed test observations; never executes model code on the host."""
    def __init__(self, root, failure=None):
        self.root = Path(root)
        self.failure = failure
        self.calls = []
        self.cleaned = []

    def run(self, action, tool_id, remaining):
        self.calls.append(tool_id)
        output = 'NO_CANDIDATE\n' if len(self.calls) == 1 else DEMO_FLAG + '\n'
        if self.failure:
            output = 'Connection failed.\n'
        log = 'logs/' + tool_id + '.log'
        (self.root / log).write_text(output)
        return {'status': self.failure or 'completed', 'exit_code': None if self.failure else 0,
                'output': output, 'log': log, 'truncated': False}

    def cleanup(self, tool_id):
        self.cleaned.append(tool_id)


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.root = self.base / 'run'
        self.materials = demo_materials(self.base / 'inputs')
        self.engines = []

    def tearDown(self):
        for engine in self.engines:
            engine.close()
        self.tmp.cleanup()

    def make(self, config=None, runner=None, models=None, category='crypto'):
        cfg = config or Config(expected_flag_sha256=digest(DEMO_FLAG))
        create_run(self.root, 'Crypto CTF: recover the XOR flag.', self.materials, category, [], cfg)
        engine = Engine(self.root, models=models, runner=runner or StubRunner(self.root))
        self.engines.append(engine)
        return engine

    def close_engine(self, engine):
        engine.close()
        self.engines.remove(engine)

    def prepare_experiment(self, engine):
        engine.inspect('originals/cipher.hex')
        engine.propose('Test the ciphertext using concrete key hypotheses.')
        engine.start_experiment(engine.queue()[0])
        return engine.s['experiments'][-1]


class ProtocolTests(unittest.TestCase):
    def test_bad_json(self):
        for raw in ('no JSON', '```json\n{}\n```', '{"status":"ok","status":"ok"}', '{"status":NaN}'):
            with self.subTest(raw=raw), self.assertRaises(ProtocolError):
                parse(raw, obj(status=string(['ok'])))

    def test_unknown_actions_and_fields(self):
        for data in ({'kind': 'shell', 'cmd': 'id'}, {'kind': 'read', 'path': 'x', 'override': True},
                     {'kind': 'run', 'argv': ['id'], 'timeout': True}):
            with self.assertRaises(ProtocolError):
                parse(json.dumps(data), ACTION)

    def test_decision_payload_validated(self):
        with self.assertRaises(ProtocolError):
            parse(json.dumps({'kind': 'experiment', 'evidence': ['task'], 'explanation': 'x', 'payload': {'cmd': 'id'}}), DECISION)

    def test_vector_normalizes_clips_and_restores(self):
        initial = initial_vector('crypto')
        u, p = update_vector(initial, {'crypto': 100, 'web': -100}, initial, 1, 0.25)
        self.assertAlmostEqual(sum(p.values()), 1)
        self.assertEqual(u['crypto'], 1)
        self.assertEqual(u['web'], 0)
        zero = {c: 0 for c in CATEGORIES}
        u, p = update_vector(zero, zero, initial, 1, 1)
        self.assertEqual(u, initial)
        self.assertAlmostEqual(sum(p.values()), 1)

    def test_configuration_validation(self):
        for cfg in (Config(eta=float('nan')), Config(max_model_calls=1.2), Config(bridge_url='https://example.org'), Config(output_bytes=999999999)):
            with self.assertRaises(ValueError):
                cfg.validate()
        self.assertEqual(parse_target('[::1]:8080'), {'host': '::1', 'port': 8080})
        with self.assertRaises(ValueError):
            parse_target('example.org:80')


class EngineTests(Fixture):
    def test_full_vertical_cycle_and_no_web(self):
        engine = self.make()
        result = engine.run()
        self.assertEqual(result['status'], 'verified')
        self.assertEqual(result['budget_used']['experiments'], 2)
        self.assertEqual([h['status'] for h in engine.s['hypotheses']], ['rejected', 'confirmed'])
        self.assertEqual({h['category'] for h in engine.s['hypotheses']}, {'crypto'})
        self.assertEqual(len(engine.store.rows('vectors')), 3)
        self.assertTrue(engine.store.rows('observations'))
        self.assertNotEqual(engine.s['sessions']['interpreter'], engine.s['sessions']['executor'])

    def test_resume_does_not_reexecute(self):
        runner = StubRunner(self.root)
        engine = self.make(runner=runner)
        self.assertEqual(engine.run(stop_after=1)['status'], 'paused')
        first_id = runner.calls[0]
        self.close_engine(engine)
        resumed = Engine(self.root, runner=runner)
        self.engines.append(resumed)
        self.assertEqual(resumed.run()['status'], 'verified')
        self.assertEqual(len(runner.calls), 2)
        self.assertEqual(runner.calls.count(first_id), 1)
        calls = resumed.s['budget']['model_calls']
        resumed.run()
        self.assertEqual(calls, resumed.s['budget']['model_calls'])
        self.assertEqual(len(runner.calls), 2)

    def test_resume_after_command_before_report(self):
        runner = StubRunner(self.root)
        engine = self.make(runner=runner)
        exp = self.prepare_experiment(engine)
        engine.continue_experiment()  # write
        engine.continue_experiment()  # run
        engine.continue_experiment()  # executor finish
        self.assertEqual(exp['status'], 'reporting')
        self.close_engine(engine)
        resumed = Engine(self.root, runner=runner)
        self.engines.append(resumed)
        resumed.run(stop_after=1)
        self.assertEqual(len(runner.calls), 1)
        self.assertEqual(resumed.s['experiments'][0]['status'], 'completed')

    def test_uncertain_side_effect_not_replayed(self):
        runner = StubRunner(self.root)
        engine = self.make(runner=runner)
        exp = self.prepare_experiment(engine)
        exp['tools'].append({'id': 'tool-crash', 'action': {'kind': 'run', 'argv': ['true'], 'timeout': 1}, 'status': 'running'})
        engine.checkpoint('simulated_crash')
        engine.recover()
        engine.continue_experiment()
        self.assertEqual(runner.calls, [])
        self.assertIn('tool-crash', runner.cleaned)
        self.assertEqual(exp['report']['signal'], 'inconclusive')

    def test_connection_error_is_neutral(self):
        engine = self.make(runner=StubRunner(self.root, 'tool_error'))
        exp = self.prepare_experiment(engine)
        before = dict(engine.s['p'])
        engine.execute_action(exp, {'kind': 'run', 'argv': ['true'], 'timeout': 1})
        engine.continue_experiment()
        self.assertEqual(engine.s['p'], before)
        self.assertEqual(exp['report']['signal'], 'inconclusive')
        self.assertNotEqual(engine.s['hypotheses'][0]['status'], 'rejected')

    def test_branch_suspends_at_uninformative_limit(self):
        engine = self.make(Config(max_uninformative=1), runner=StubRunner(self.root, 'timeout'))
        exp = self.prepare_experiment(engine)
        engine.execute_action(exp, {'kind': 'run', 'argv': ['true'], 'timeout': 1})
        engine.continue_experiment()
        self.assertEqual(engine.s['hypotheses'][0]['status'], 'suspended')

    def test_identical_experiment_blocked_despite_vector_change(self):
        engine = self.make(runner=StubRunner(self.root, 'timeout'))
        exp = self.prepare_experiment(engine)
        engine.execute_action(exp, {'kind': 'run', 'argv': ['true'], 'timeout': 1})
        engine.continue_experiment()
        h = engine.s['hypotheses'][0]
        engine.s['p']['crypto'] = 0.9
        engine.start_experiment(h)
        self.assertEqual(h['status'], 'suspended')
        self.assertEqual(len(engine.s['experiments']), 1)

    def test_candidate_is_not_success_without_criterion(self):
        engine = self.make(Config())
        self.assertEqual(engine.run()['status'], 'candidate_unverified')
        self.assertFalse(engine.s['checks'][0]['verified'])

    def test_wrong_criterion_does_not_verify(self):
        engine = self.make(Config(expected_flag_sha256=digest('different-flag')))
        self.assertEqual(engine.run()['status'], 'candidate_unverified')

    def test_invalid_model_json_blocks_and_saves(self):
        class Bad:
            def complete(self, *args):
                return '{bad JSON', None
        engine = self.make(models=Bad())
        self.assertEqual(engine.run()['status'], 'blocked')
        self.assertEqual(engine.store.load()['budget']['model_calls'], 1)
        self.assertEqual(len(engine.s['experiments']), 0)
        self.assertEqual(engine.store.rows('model_calls')[0]['status'], 'error')

    def test_budget_stops_calls(self):
        engine = self.make(Config(max_model_calls=1))
        self.assertEqual(engine.run()['status'], 'partial')
        self.assertEqual(engine.s['budget']['model_calls'], 1)

    def test_crypto_rejects_web_branch_without_evidence(self):
        class WebFake(FakeModels):
            def complete(self, role, *args):
                raw, session = super().complete(role, *args)
                if role == 'interpreter':
                    data = json.loads(raw)
                    data['category'] = 'web'
                    raw = json.dumps(data)
                return raw, session
        engine = self.make(models=WebFake())
        self.assertEqual(engine.run()['status'], 'blocked')
        self.assertEqual(engine.s['budget']['experiments'], 0)

    def test_model_selects_initial_direction_when_unknown(self):
        engine = self.make(category=None)
        engine.run(stop_after=1)
        self.assertEqual(engine.s['initial'], initial_vector('crypto'))
        self.assertEqual(engine.s['inferred_category'], 'crypto')

    def test_originals_unchanged_and_integrity_checked(self):
        engine = self.make()
        original = (self.materials / 'cipher.hex').read_bytes()
        engine.run()
        self.assertEqual((self.root / 'originals/cipher.hex').read_bytes(), original)
        target = self.root / 'originals/cipher.hex'
        target.chmod(0o644)
        target.write_text('changed')
        with self.assertRaises(ValueError):
            engine.verify_originals()

    def test_path_traversal_and_symlinks(self):
        engine = self.make()
        (self.root / 'work/link').symlink_to(self.materials / 'cipher.hex')
        for name in ('../outside', '/etc/passwd', 'work/link'):
            with self.assertRaises(ValueError):
                safe_path(self.root, name)

    def test_second_active_engine_is_rejected(self):
        self.make()
        with self.assertRaises(ValueError):
            Engine(self.root)

    def test_new_related_data_resumes_suspended_branch(self):
        engine = self.make()
        self.prepare_experiment(engine)
        h = engine.s['hypotheses'][0]
        h['status'] = 'suspended'
        ob = engine.observe('New user evidence', ['originals/cipher.hex'], 'new_data', h['object_id'])
        self.assertEqual(h['status'], 'pending')
        self.assertIn(ob['id'], h['evidence'])

    def test_nonexistent_evidence_is_rejected(self):
        engine = self.make()
        with self.assertRaises(ProtocolError):
            engine.evidence(['invented_source'])


class AdapterTests(unittest.TestCase):
    def test_gemini_exact_model_and_no_secret_body(self):
        calls = []
        def transport(url, body, headers, *args):
            calls.append((url, body, headers))
            return {'candidates': [{'content': {'parts': [{'text': '{"status":"ok"}'}]}}]}
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test-only-secret'}):
            adapter = LiveModels(Config(), transport)
            raw, _ = adapter.complete('orchestrator', obj(status=string()), {}, None, 1)
        self.assertIn('/gemini-4-hydrogen:generateContent', calls[0][0])
        self.assertNotIn('test-only-secret', json.dumps(calls[0][1]))
        self.assertEqual(calls[0][2]['x-goog-api-key'], 'test-only-secret')
        self.assertEqual(json.loads(raw)['status'], 'ok')

    def test_bridge_explicit_sessions_and_no_native_tools(self):
        calls = []
        def transport(url, body, *args):
            calls.append(body)
            return {'conversation_id': 'session-' + str(len(calls)), 'choices': [{'message': {'content': '{}'}}]}
        adapter = LiveModels(Config(), transport)
        adapter.complete('interpreter', obj(), {}, None, 1)
        adapter.complete('executor', obj(), {}, None, 1)
        adapter.complete('interpreter', obj(), {}, 'session-1', 1)
        self.assertNotIn('conversation_id', calls[0])
        self.assertNotIn('conversation_id', calls[1])
        self.assertEqual(calls[2]['conversation_id'], 'session-1')
        self.assertNotIn('tools', calls[0])
        self.assertFalse(calls[0]['search'])

    def test_http_status_mapping_without_retry(self):
        for status, code in ((401, 'login_required'), (403, 'access_denied'), (404, 'model_unavailable'),
                             (429, 'rate_limited'), (503, 'session_or_service_unavailable')):
            with self.subTest(status=status), patch('ctf_agent.adapters.build_opener') as opener:
                opener.return_value.open.side_effect = HTTPError('http://127.0.0.1', status, 'secret body', {}, None)
                with self.assertRaises(ModelError) as raised:
                    post_json('http://127.0.0.1', {}, {}, 1, 1024)
                self.assertEqual(raised.exception.code, code)
                self.assertNotIn('secret', str(raised.exception))
                self.assertEqual(opener.return_value.open.call_count, 1)


class RunnerPolicyTests(Fixture):
    def test_command_hardening_and_network_allowlist(self):
        engine = self.make()
        runner = DockerRunner(self.root, Config(), [{'host': '192.0.2.10', 'port': 80}])
        cmd = runner.command({'kind': 'run', 'argv': ['python3', '-c', 'pass']}, 'test', 1)
        self.assertEqual(cmd[cmd.index('--network') + 1], 'none')
        self.assertIn('--read-only', cmd)
        self.assertIn('--cap-drop=ALL', cmd)
        self.assertNotIn('/var/run/docker.sock', ' '.join(cmd))
        with self.assertRaises(ValueError):
            runner.command({'kind': 'http', 'host': '192.0.2.10', 'port': 81, 'path': '/', 'tls': False}, 'test', 1)
        cmd = runner.command({'kind': 'http', 'host': '192.0.2.10', 'port': 80, 'path': '/', 'tls': False}, 'test', 1)
        self.assertNotIn('--mount', cmd)
        self.assertIn('/opt/ctf/http_fetch.py', cmd)

class CrashBoundaryTests(Fixture):
    def test_report_cannot_turn_failed_tool_into_negative_evidence(self):
        engine = self.make(runner=StubRunner(self.root, 'timeout'))
        exp = self.prepare_experiment(engine)
        engine.execute_action(exp, {'kind': 'run', 'argv': ['true'], 'timeout': 1})
        before = dict(engine.s['p'])
        report = engine.forced_report(exp)
        report['status'], report['signal'] = 'informative', 'negative'
        source = exp['tools'][0]['result']['log']
        report['changes'] = [{'category': 'crypto', 'delta': -1, 'reason': 'Untrusted model claim', 'evidence': [source]}]
        exp.pop('forced_error')  # Simulate crash before secondary failure checkpoint.
        engine.finish_experiment(exp, report)
        self.assertEqual(engine.s['p'], before)
        self.assertEqual(exp['report']['signal'], 'inconclusive')

    def test_known_credentials_redacted_from_model_response_file(self):
        class Echo:
            def complete(self, *args):
                return 'secret-test-token', None
        engine = self.make(models=Echo())
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'secret-test-token'}):
            result = engine.run()
        self.assertEqual(result['status'], 'blocked')
        response = next((self.root / 'models').glob('*.response.txt')).read_text()
        self.assertEqual(response, '[REDACTED]')

    def test_experiment_finalization_rolls_back_as_a_unit(self):
        engine = self.make()
        exp = self.prepare_experiment(engine)
        engine.continue_experiment()
        engine.continue_experiment()
        engine.continue_experiment()
        report = engine.call('reporter', REPORT,
                             engine.experiment_context(exp))
        prior_vectors = len(engine.store.rows('vectors'))
        with patch.object(engine, 'check_candidates', side_effect=ValueError('simulated verification failure')):
            with self.assertRaises(ValueError):
                engine.finish_experiment(exp, report)
        self.assertEqual(len(engine.store.rows('vectors')), prior_vectors)
        self.assertIsNotNone(engine.store.load()['active'])
        self.assertEqual(engine.store.load()['experiments'][0]['status'], 'reporting')


class SourceReferenceTests(Fixture):
    def test_written_script_source_can_be_cited(self):
        engine = self.make()
        exp = self.prepare_experiment(engine)
        engine.execute_action(exp, {'kind': 'write', 'path': 'work/analysis.py', 'content': 'print(1)'})
        source = exp['tools'][0]['result']['source']
        engine.evidence([source])
        self.assertTrue((self.root / source).is_file())


if __name__ == '__main__':
    unittest.main()
