import fcntl
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time
import uuid

TABLES = ('objects', 'relations', 'hypotheses', 'experiments', 'attempts', 'observations',
          'vectors', 'model_calls', 'tool_calls', 'checks', 'budgets', 'events')


def uid(prefix):
    return prefix + '_' + uuid.uuid4().hex[:12]


def digest(data):
    if isinstance(data, str):
        data = data.encode()
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def safe_path(root, name):
    root = Path(root).resolve()
    rel = Path(name)
    if rel.is_absolute() or '..' in rel.parts or not rel.parts:
        raise ValueError('path must be relative without traversal')
    current = root
    for part in rel.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError('symlinks are not allowed')
    if not current.resolve().is_relative_to(root):
        raise ValueError('path escapes storage')
    return current


class Store:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._lock = None
        self.db = sqlite3.connect(self.root / 'state.sqlite3', isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA foreign_keys=ON')
        self.db.execute('CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, payload TEXT NOT NULL)')
        for table in TABLES:
            self.db.execute(f'CREATE TABLE IF NOT EXISTS {table} (id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(id), created REAL NOT NULL, payload TEXT NOT NULL)')
        for folder in ('originals', 'work', 'scripts', 'logs', 'artifacts', 'models'):
            (self.root / folder).mkdir(exist_ok=True)

    def lock(self):
        self._lock = open(self.root / '.lock', 'a')
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._lock.close()
            self._lock = None
            raise ValueError('another process already owns this run') from None

    def close(self):
        self.db.close()
        if self._lock:
            self._lock.close()

    def save(self, state):
        self.db.execute('INSERT OR REPLACE INTO runs VALUES (?, ?)', (state['id'], canonical(state)))

    def load(self):
        row = self.db.execute('SELECT payload FROM runs').fetchone()
        if not row:
            raise ValueError('run not initialized')
        return json.loads(row[0])

    def put(self, table, run_id, value, item_id=None):
        if table not in TABLES:
            raise ValueError('unknown table')
        item_id = item_id or value.get('id') or uid(table[:-1])
        self.db.execute(f'INSERT OR REPLACE INTO {table} VALUES (?, ?, ?, ?)',
                        (item_id, run_id, time.time(), canonical(value)))
        return item_id

    def rows(self, table):
        if table not in TABLES:
            raise ValueError('unknown table')
        return [{'id': r['id'], **json.loads(r['payload'])} for r in
                self.db.execute(f'SELECT id,payload FROM {table} ORDER BY created,id')]

    def checkpoint(self, state, event, details=None):
        self.db.execute('BEGIN IMMEDIATE')
        try:
            self.save(state)
            for h in state['hypotheses']:
                self.put('hypotheses', state['id'], h)
            for exp in state['experiments']:
                self.put('experiments', state['id'], exp)
            self.put('budgets', state['id'], state['budget'], state['id'])
            self.put('events', state['id'], {'event': event, 'details': details or {}, 'time': time.time()})
            self.db.execute('COMMIT')
        except BaseException:
            self.db.execute('ROLLBACK')
            raise

    def write_json(self, name, value):
        target = safe_path(self.root, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + '.tmp')
        tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        os.replace(tmp, target)
        return name

    def export_journal(self):
        self.write_json('journal.json', {'events': self.rows('events'), 'vectors': self.rows('vectors'),
                                        'hypotheses': self.rows('hypotheses'), 'checks': self.rows('checks')})
