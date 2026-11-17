"""Actual pinned FastAPI routes; replace upstream generation, never contact an account."""
import asyncio
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

project = Path(__file__).resolve().parents[1]
service = project / '.services/opencode-deepseek'
sys.path[:0] = [str(project), str(service)]
from fastapi.testclient import TestClient
from server import api
from deepseek.auth import LoginRequired
from server.safety import CooldownActive
from ctf_agent.adapters import LiveModels, ModelError
from ctf_agent.config import Config
from ctf_agent.protocol import obj, string, parse

calls = []
async def generate(prompt, req, model_type):
    calls.append(req)
    cid = req.conversation_id or ('session-' + str(len(calls)) + ':0')
    session_id, turn = cid.split(':')
    return SimpleNamespace(text='{"status":"ok"}', conversation_id=session_id + ':' + str(int(turn) + 1))

with patch.object(api, '_run_chat_with_retry', generate), \
     patch.object(api._budget, 'reserve'), patch.object(api._cooldown, 'ensure'), \
     patch.object(api, '_remember'), TestClient(api.app) as client:
    assert client.get('/healthz').json()['status'] == 'ok'
    models = client.get('/v1/models').json()['data']
    assert 'deepseek-chat' in [m['id'] for m in models]
    def transport(url, body, headers, timeout, max_bytes):
        response = client.post('/v1/chat/completions', json=body)
        assert response.status_code == 200, response.status_code
        return response.json()
    adapter = LiveModels(Config(), transport)
    schema = obj(status=string(['ok']))
    raw, interpreter = adapter.complete('interpreter', schema, {}, None, 10)
    assert parse(raw, schema)['status'] == 'ok'
    _, executor = adapter.complete('executor', schema, {}, None, 10)
    _, interpreter2 = adapter.complete('reporter', schema, {}, interpreter, 10)
    assert interpreter.split(':')[0] != executor.split(':')[0]
    assert interpreter.split(':')[0] == interpreter2.split(':')[0]
    assert calls[2].conversation_id == interpreter
    assert all(not c.tools and not c.stream and not c.search for c in calls)
    async def login_error(*_):
        raise LoginRequired()
    with patch.object(api, '_run_chat_with_retry', login_error):
        response = client.post('/v1/chat/completions', json={'model':'deepseek-chat', 'messages':[{'role':'user','content':'test'}]})
        assert response.status_code == 503
        assert response.json()['error']['type'] == 'login_required'
    with patch.object(api._cooldown, 'ensure', side_effect=CooldownActive('test cooldown')):
        response = client.post('/v1/chat/completions', json={'model':'deepseek-chat', 'messages':[{'role':'user','content':'test'}]})
        assert response.status_code == 429
print('Tsuev pinned upstream HTTP contract verified; no external generation.')
