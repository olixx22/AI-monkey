"""Explicit lifecycle for the separately installed, pinned Tsuev bridge."""
import json
import os
from pathlib import Path
import subprocess
from urllib.parse import urlsplit
from urllib.request import Request, build_opener, ProxyHandler
from urllib.error import URLError, HTTPError

from .adapters import NoRedirect

REPOSITORY = 'https://github.com/Tsuev/opencode-deepseek'
REVISION = '2e3c81d7ea47a5fabb5aa7fa2a4f45cf681981c9'
DEFAULT_DIRECTORY = Path(__file__).resolve().parents[1] / '.services' / 'opencode-deepseek'


def checkout(directory):
    directory = Path(directory).resolve()
    python = directory / '.venv/bin/python'
    if not python.is_file():
        raise ValueError('bridge not installed; run scripts/setup_deepseek.sh explicitly')
    try:
        rev = subprocess.check_output(['git', '-C', str(directory), 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL, timeout=5).decode().strip()
        dirty = subprocess.check_output(['git', '-C', str(directory), 'status', '--porcelain', '--untracked-files=no'], timeout=5).strip()
    except (OSError, subprocess.SubprocessError):
        raise ValueError('cannot verify bridge checkout revision') from None
    if rev != REVISION or dirty:
        raise ValueError('bridge must have the pinned revision and unmodified tracked sources')
    return directory, python


def environment(directory):
    env = dict(os.environ)
    # Neither a model API key nor another user's browser profile belongs in this service.
    for name in list(env):
        if name.startswith(('GEMINI_', 'GOOGLE_', 'OPENAI_', 'DEEPSEEK_')) or name in ('PYTHONPATH', 'PYTHONHOME'):
            env.pop(name)
    env.update(SERVER_INTERACTIVE_LOGIN='0', SESSION_REFRESH_ENABLED='0',
               CONVERSATION_REUSE='0', DEBUG_TOOLCALLS='0',
               DEEPSEEK_PROFILE_DIR=str(Path(directory) / 'session/profile'))
    # Keep upstream pacing, quotas and cooldown defaults; never evade a refusal.
    return env


def launch(action, directory=DEFAULT_DIRECTORY, port=8000):
    directory, python = checkout(directory)
    if not 1 <= port <= 65535:
        raise ValueError('port must be 1..65535')
    if action == 'login':
        # Do not use upstream __main__: it prints a token prefix. Login stays manual.
        code = ("import os; os.umask(0o077); import deepseek.auth as auth; "
                "auth.LAUNCH_ARGS=[]; auth.login(); print('DeepSeek login completed; session saved locally.')")
    elif action == 'start':
        if not (directory / 'session/session.json').is_file():
            raise ValueError('login_required: run python3 -m ctf_agent bridge login first')
        # No .env loader, interactive login, or background refresh from application startup.
        code = ("import os; os.umask(0o077); import deepseek.auth as auth; auth.LAUNCH_ARGS=[]; "
                "import uvicorn; uvicorn.run('server.api:app', host='127.0.0.1', "
                f"port={port}, access_log=False)")
    else:
        raise ValueError('unknown bridge lifecycle action')
    try:
        return subprocess.call([str(python), '-c', code], cwd=directory, env=environment(directory))
    except KeyboardInterrupt:
        return 130


def status(base_url, directory=DEFAULT_DIRECTORY):
    u = urlsplit(base_url)
    origin = f'{u.scheme}://{u.netloc}'
    result = {'provider': 'tsuev', 'repository': REPOSITORY, 'revision': REVISION,
              'installed': (Path(directory) / '.venv/bin/python').is_file(),
              'session_file_present': (Path(directory) / 'session/session.json').is_file(),
              'reachable': False}
    opener = build_opener(ProxyHandler({}), NoRedirect())
    try:
        with opener.open(Request(origin + '/healthz'), timeout=3) as response:
            data = json.loads(response.read(4096))
            result['reachable'] = data.get('status') == 'ok'
    except (URLError, HTTPError, OSError, ValueError):
        pass
    # A health response does not prove authentication or the identity of a process on that port.
    return result
