import io
import json
import os
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from ctf_agent.adapters import LiveModels, ModelError, post_json
from ctf_agent.bridge import DEFAULT_DIRECTORY, REVISION, environment, launch
from ctf_agent.config import Config
from ctf_agent.protocol import ACTION, ProtocolError, obj
from tests.test_core import Fixture


class TsuevTests(unittest.TestCase):
    def test_explicit_tsuev_provider(self):
        self.assertEqual(Config().bridge_provider, 'tsuev')
        with self.assertRaises(ValueError):
            Config(bridge_provider='unknown').validate()

    def test_reporter_keeps_interpreter_session_namespace(self):
        def transport(url, body, headers, timeout, max_bytes):
            self.assertEqual(body['user'], 'ctf-agent:interpreter')
            self.assertEqual(body['conversation_id'], 'session-a:1')
            self.assertNotIn('tools', body)
            return {'conversation_id': 'session-a:2', 'choices': [{'message': {'content': '{}'}}]}
        _, session = LiveModels(Config(), transport).complete('reporter', obj(), {}, 'session-a:1', 1)
        self.assertEqual(session, 'session-a:2')

    def test_native_or_emulated_tool_calls_are_not_executed(self):
        def transport(*args):
            return {'conversation_id': 'a:1', 'choices': [{'message': {'content': '{}', 'tool_calls': [{'id': 'bad'}]}}]}
        with self.assertRaises(ModelError) as error:
            LiveModels(Config(), transport).complete('executor', obj(), {}, None, 1)
        self.assertEqual(error.exception.code, 'unexpected_tool_calls')

    def test_login_required_and_retry_after_sanitized(self):
        error = HTTPError('http://127.0.0.1', 503, 'hidden', {'Retry-After': '120'},
                          io.BytesIO(json.dumps({'error': {'type': 'login_required', 'message': 'secret-cookie'}}).encode()))
        with patch('ctf_agent.adapters.build_opener') as opener:
            opener.return_value.open.side_effect = error
            with self.assertRaises(ModelError) as result:
                post_json('http://127.0.0.1', {}, {}, 1, 4096)
            self.assertEqual(result.exception.code, 'login_required')
            self.assertEqual(result.exception.retry_after, 120)
            self.assertNotIn('secret-cookie', str(result.exception))
            self.assertEqual(opener.return_value.open.call_count, 1)

    def test_invalid_gemini_key_reason_without_server_message(self):
        error = HTTPError('https://generativelanguage.googleapis.com', 400, 'hidden', {},
                          io.BytesIO(json.dumps({'error': {'message': 'echoed-secret', 'details': [{'reason': 'API_KEY_INVALID'}]}}).encode()))
        with patch('ctf_agent.adapters.build_opener') as opener:
            opener.return_value.open.side_effect = error
            with self.assertRaises(ModelError) as result:
                post_json('https://generativelanguage.googleapis.com', {}, {}, 1, 4096)
            self.assertEqual(result.exception.code, 'invalid_api_key')
            self.assertNotIn('echoed-secret', str(result.exception))

    def test_bridge_env_does_not_receive_model_credentials_or_personal_profile(self):
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'secret', 'GOOGLE_API_KEY': 'secret',
                                     'DEEPSEEK_PROFILE_DIR': '/personal/profile', 'PYTHONPATH': '/wrong/modules'}):
            env = environment(Path('/service'))
        self.assertNotIn('GEMINI_API_KEY', env)
        self.assertNotIn('GOOGLE_API_KEY', env)
        self.assertNotIn('PYTHONPATH', env)
        self.assertEqual(env['DEEPSEEK_PROFILE_DIR'], '/service/session/profile')
        self.assertEqual(env['SERVER_INTERACTIVE_LOGIN'], '0')
        self.assertEqual(env['CONVERSATION_REUSE'], '0')


class TsuevSessionTests(Fixture):
    def test_different_turn_ids_cannot_hide_same_conversation(self):
        from ctf_agent.adapters import FakeModels
        class Collision(FakeModels):
            def complete(self, role, *args):
                raw, _ = super().complete(role, *args)
                return raw, 'same-session:2' if role == 'executor' else 'same-session:1'
        engine = self.make(models=Collision())
        exp = self.prepare_experiment(engine)
        with self.assertRaises(ProtocolError):
            engine.call('executor', ACTION, engine.experiment_context(exp))


@unittest.skipUnless(os.environ.get('CTF_TEST_TSUEV') == '1', 'set CTF_TEST_TSUEV=1 for installed upstream contract tests')
class InstalledTsuevTests(unittest.TestCase):
    def test_real_upstream_routes_with_fake_model_backend(self):
        project = Path(__file__).resolve().parents[1]
        env = environment(DEFAULT_DIRECTORY)
        result = subprocess.run([str(DEFAULT_DIRECTORY / '.venv/bin/python'), str(project / 'tests/tsuev_contract_check.py')],
                                env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('contract verified', result.stdout)

class GeminiCredentialTests(unittest.TestCase):
    def test_smoke_uses_low_thinking_and_restores_environment(self):
        from ctf_agent.cli import main
        def complete(adapter, role, schema, context, session, timeout):
            self.assertEqual(os.environ['GEMINI_API_KEY'], 'temporary-test-key')
            self.assertEqual(adapter.config.gemini_thinking_level, 'low')
            return '{"status":"ok"}', None
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'existing-test-key'}), \
             patch('ctf_agent.cli.sys.stdin.isatty', return_value=True), \
             patch('ctf_agent.cli.getpass.getpass', return_value='temporary-test-key'), \
             patch.object(LiveModels, 'complete', complete), patch('sys.stdout', new_callable=io.StringIO) as output:
            self.assertEqual(main(['live-check', '--provider', 'gemini', '--prompt-key']), 0)
            self.assertEqual(os.environ['GEMINI_API_KEY'], 'existing-test-key')
            self.assertNotIn('temporary-test-key', output.getvalue())

    def test_low_thinking_and_output_limit_reach_official_request(self):
        def transport(url, body, headers, timeout, max_bytes):
            self.assertEqual(body['generationConfig']['thinkingConfig'], {'thinkingLevel':'LOW', 'includeThoughts':False})
            self.assertEqual(body['generationConfig']['maxOutputTokens'], 2048)
            return {'candidates':[{'content':{'parts':[{'text':'{}'}]}}]}
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test-key'}):
            LiveModels(Config(gemini_thinking_level='low', gemini_output_tokens=2048), transport).complete('orchestrator', obj(), {}, None, 1)


if __name__ == '__main__':
    unittest.main()
