"""One experiment at a time. Durable checkpoints surround all external effects."""
import copy
import json
import os
from pathlib import Path
import re
import shutil
import stat
import time

from .adapters import FakeModels, LiveModels, ModelError
from .config import CATEGORIES, Config, initial_vector, update_vector
from .protocol import ACTION, DECISION, DECISION_PAYLOADS, HYPOTHESIS, REPORT, ProtocolError, parse
from .runner import DockerRunner, regular_files
from .storage import Store, canonical, digest, safe_path, uid

FLAG = re.compile(r'flag\{[^\r\n{}]{1,200}\}')
TERMINAL = {'verified', 'candidate_unverified', 'partial', 'completed'}


class BudgetExhausted(RuntimeError):
    pass


def create_run(root, task, materials, category, targets, config):
    config.validate()
    if category is not None and category not in CATEGORIES:
        raise ValueError('unknown category')
    root, materials = Path(root).resolve(), Path(materials).resolve()
    if not materials.is_dir() or root == materials or root.is_relative_to(materials) or materials.is_relative_to(root):
        raise ValueError('materials and run directories must be distinct and not nested')
    if not task.strip() or len(task.encode()) > config.model_bytes // 2:
        raise ValueError('task text is empty or too large')
    if (root / 'state.sqlite3').exists():
        raise ValueError('run already exists; use resume')
    files = []
    total = 0
    for directory, folders, names in os.walk(materials, followlinks=False):
        for name in folders + names:
            p = Path(directory) / name
            if p.is_symlink():
                raise ValueError('material symlinks are not accepted')
        for name in names:
            p = Path(directory) / name
            if not stat.S_ISREG(p.stat().st_mode):
                raise ValueError('only regular material files are accepted')
            total += p.stat().st_size
            files.append(p)
    if total > config.material_bytes or len(files) > 256:
        raise ValueError('material byte/file limit exceeded')
    store = Store(root)
    try:
        store.lock()
        initial = initial_vector(category)
        state = {'id': uid('run'), 'task': task, 'category': category, 'targets': targets,
                 'config': config.to_dict(), 'status': 'ready', 'reason': '', 'created': time.time(),
                 'deadline': time.time() + config.max_seconds, 'initial': initial, 'u': dict(initial),
                 'p': dict(initial), 'hypotheses': [], 'experiments': [], 'observations': [],
                 'objects': [], 'checks': [], 'inspected': [], 'sessions': {}, 'web_evidence': [],
                 'data_version': 0, 'active': None,
                 'budget': {'experiments': 0, 'model_calls': 0, 'actions': 0}}
        store.save(state)
        store.write_json('task.json', {'task': task, 'category': category, 'targets': targets})
        # Host writes scripts; arbitrary unprivileged container writes only inside work.
        (root / 'work').chmod(0o777)
        for source in sorted(files):
            name = 'originals/' + source.relative_to(materials).as_posix()
            target = safe_path(root, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            target.chmod(0o444)
            item = {'id': name, 'sha256': digest(target.read_bytes()), 'size': target.stat().st_size, 'source': name}
            state['objects'].append(item)
            store.put('objects', state['id'], item)
            store.put('relations', state['id'], {'from': 'task', 'to': name, 'kind': 'supplied_material'})
        store.write_json('manifest.json', state['objects'])
        store.put('vectors', state['id'], {'u': initial, 'p': initial, 'changes': [], 'reason': 'user_category' if category else 'uniform_until_interpreted'})
        store.checkpoint(state, 'created', {'category': category, 'materials': len(files)})
    finally:
        store.close()
    return state


class Engine:
    def __init__(self, root, models=None, runner=None):
        self.store = Store(root)
        try:
            self.store.lock()
            self.s = self.store.load()
            self.cfg = Config(**self.s['config']).validate()
            if self.cfg.mode == 'live' and 'bridge_provider' not in self.s['config'] and any(self.s['sessions'].get(r) for r in ('interpreter', 'executor')):
                raise ValueError('legacy live run has unbound bridge sessions; create a new Tsuev run')
            self.models = models or (FakeModels() if self.cfg.mode == 'fake' else LiveModels(self.cfg))
            self.runner = runner or DockerRunner(self.store.root, self.cfg, self.s['targets'])
            self.verify_originals()
        except BaseException:
            self.store.close()
            raise

    def close(self):
        self.store.close()

    def verify_originals(self):
        for item in self.s['objects']:
            if item['id'].startswith('originals/'):
                p = safe_path(self.store.root, item['id'])
                if not p.is_file() or p.stat().st_size != item['size'] or digest(p.read_bytes()) != item['sha256']:
                    raise ValueError('original material integrity check failed')

    def checkpoint(self, event, details=None):
        self.store.checkpoint(self.s, event, details)

    def remaining(self):
        remaining = self.s['deadline'] - time.time()
        if remaining <= 0:
            raise BudgetExhausted('run wall-time budget exhausted')
        return remaining

    def refs(self):
        refs = {'task', 'task.json'} | {x['id'] for x in self.s['objects']}
        refs |= {x['id'] for x in self.s['observations']}
        for observation in self.s['observations']:
            refs.update(observation['sources'])
        for exp in self.s['experiments']:
            refs.add(exp['id'])
            refs.update(exp.get('artifacts', []))
            for tool in exp['tools']:
                refs.add(tool['id'])
                for field in ('log', 'source'):
                    if tool.get('result', {}).get(field):
                        refs.add(tool['result'][field])
        return refs

    def evidence(self, refs):
        if not refs or any(r not in self.refs() for r in refs):
            raise ProtocolError('unknown or missing evidence reference')

    def redact(self, text):
        for name in ('GEMINI_API_KEY', 'DEEPSEEK_API_KEY', 'DEEPSEEK_TOKEN'):
            value = os.environ.get(name)
            if value:
                text = text.replace(value, '[REDACTED]')
        return text

    def call(self, role, schema, context):
        self.remaining()
        if self.s['budget']['model_calls'] >= self.cfg.max_model_calls:
            raise BudgetExhausted('model-call budget exhausted')
        self.s['budget']['model_calls'] += 1
        item_id = uid('model')
        session_role = 'interpreter' if role == 'reporter' else role
        request_path = f'models/{item_id}.request.json'
        request_text = self.redact(canonical({'role': role, 'context': context, 'schema': schema}))
        if len(request_text.encode()) > self.cfg.model_bytes * 4:
            raise ProtocolError('model context byte budget exceeded; compact the investigation')
        self.store.write_json(request_path, json.loads(request_text))
        item = {'id': item_id, 'role': role, 'status': 'started', 'request': request_path,
                'session_role': session_role}
        self.store.put('model_calls', self.s['id'], item)
        self.checkpoint('model_started', {'id': item_id, 'role': role})
        try:
            raw, session = self.models.complete(role, schema, context, self.s['sessions'].get(session_role),
                                                 min(self.cfg.model_seconds, self.remaining()))
            # Never let a faulty bridge merge interpreter and executor conversations.
            other = 'executor' if session_role == 'interpreter' else 'interpreter'
            if session_role in ('executor', 'interpreter') and session and session.split(':', 1)[0] == (self.s['sessions'].get(other) or '').split(':', 1)[0]:
                raise ProtocolError('bridge merged distinct role sessions')
            self.s['sessions'][session_role] = session
            if not isinstance(raw, str) or len(raw.encode()) > self.cfg.model_bytes:
                raise ProtocolError('model output byte limit exceeded')
            response_path = f'models/{item_id}.response.txt'
            safe_path(self.store.root, response_path).write_text(self.redact(raw))
            item['response'] = response_path
            result = parse(raw, schema)
            item['status'] = 'completed'
            return result
        except (ModelError, ProtocolError) as e:
            item['status'] = 'error'
            item['error'] = getattr(e, 'code', 'invalid_model_json')
            if getattr(e, 'retry_after', None) is not None:
                item['retry_after'] = e.retry_after
            raise
        finally:
            self.store.put('model_calls', self.s['id'], item)
            self.checkpoint('model_finished', {'id': item_id, 'status': item['status']})

    def queue(self):
        return sorted((h for h in self.s['hypotheses'] if h['status'] == 'pending'),
                      key=lambda h: (-self.s['p'][h['category']] * h['local_priority'] / (1 + h['attempts']), h['id']))

    def context(self):
        return {'task': self.s['task'], 'category': self.s['category'], 'targets': self.s['targets'],
                'objects': self.s['objects'], 'priorities': self.s['p'], 'hypotheses': self.s['hypotheses'],
                'queue': self.queue(), 'inspected': self.s['inspected'], 'checks': self.s['checks'],
                'observations': self.s['observations'][-8:], 'web_evidence': self.s['web_evidence'],
                'previous_results': [e.get('report') for e in self.s['experiments'][-8:] if e.get('report')],
                'remaining_budget': {'seconds': max(0, self.s['deadline'] - time.time()),
                                     'model_calls': self.cfg.max_model_calls - self.s['budget']['model_calls'],
                                     'experiments': self.cfg.max_experiments - self.s['budget']['experiments']},
                'decision_payload_schemas': DECISION_PAYLOADS}

    def observe(self, text, sources, kind='analysis', related=None):
        self.evidence(sources)
        observation = {'id': uid('obs'), 'text': text[:self.cfg.output_bytes], 'sources': sources, 'kind': kind}
        self.s['observations'].append(observation)
        self.store.put('observations', self.s['id'], observation)
        if kind == 'new_data':
            self.s['data_version'] += 1
            for h in self.s['hypotheses']:
                if h['status'] == 'suspended' and related == h['object_id']:
                    h['status'] = 'pending'
                    h['uninformative'] = 0
                    h['attempts_since_data'] = 0
                    h['evidence'].append(observation['id'])
                    self.store.put('relations', self.s['id'], {'from': observation['id'], 'to': h['id'], 'kind': 'resumes_with_new_data'})
        return observation

    def read(self, name):
        if not (name.startswith('originals/') or name.startswith('work/')):
            raise ProtocolError('only originals/ and work/ can be read')
        path = safe_path(self.store.root, name)
        if not stat.S_ISREG(path.stat().st_mode):
            raise ProtocolError('only regular files can be read')
        with path.open('rb') as f:
            data = f.read(self.cfg.output_bytes)
        return data.decode('utf-8', 'replace')

    def inspect(self, name):
        if name not in {o['id'] for o in self.s['objects']}:
            raise ProtocolError('inspection requires a registered material object')
        content = self.read(name)
        if name in self.s['inspected']:
            raise ProtocolError('unchanged material was already inspected; request a concrete experiment')
        self.s['inspected'].append(name)
        ob = self.observe(content or '(empty file)', [name], 'material')
        # Concrete locally supplied web component, not category speculation.
        if name.lower().endswith(('.html', '.htm')) and re.search(r'<(?:html|form|script)\b', content, re.I):
            self.s['web_evidence'].append(ob['id'])
        self.checkpoint('material_inspected', {'path': name, 'observation': ob['id']})

    def propose(self, assignment):
        ctx = self.context()
        ctx['assignment'] = assignment
        h = self.call('interpreter', HYPOTHESIS, ctx)
        self.evidence(h['evidence'])
        known = {o['id'] for o in self.s['objects']}
        if h['object_id'] not in known or any(i not in known for i in h['experiment']['inputs']):
            raise ProtocolError('hypothesis must reference existing objects and inputs')
        if len(h['hypothesis'].split()) < 5 or len(h['experiment']['procedure']) < 8:
            raise ProtocolError('hypothesis/procedure must be concrete')
        if h['category'] == 'web' and self.s['category'] == 'crypto':
            if not set(h['evidence']) & set(self.s['web_evidence']):
                raise ProtocolError('crypto task cannot branch into web without concrete web evidence')
        h.update(id=uid('hyp'), status='pending', attempts=0, attempts_since_data=0,
                 uninformative=0, experiments=[], created_version=self.s['data_version'])
        self.s['hypotheses'].append(h)
        for ref in h['evidence']:
            self.store.put('relations', self.s['id'], {'from': ref, 'to': h['id'], 'kind': 'supports'})
        self.checkpoint('hypothesis_created', {'id': h['id'], 'category': h['category'], 'hypothesis': h['hypothesis']})

    def get_hypothesis(self, item_id):
        for h in self.s['hypotheses']:
            if h['id'] == item_id:
                return h
        raise ProtocolError('unknown hypothesis')

    def apply_changes(self, changes, reason, allow=True):
        if len({c['category'] for c in changes}) != len(changes):
            raise ProtocolError('duplicate direction delta')
        deltas = {c: 0. for c in CATEGORIES}
        for change in changes:
            self.evidence(change['evidence'])
            if change['delta']:
                observation_refs = {o['id'] for o in self.s['observations'] if o['kind'] != 'tool_error'}
                for ob in self.s['observations']:
                    if ob['kind'] != 'tool_error':
                        observation_refs.update(ob['sources'])
                if not set(change['evidence']) & observation_refs:
                    raise ProtocolError('priority changes require a concrete informative observation')
            if allow:
                deltas[change['category']] = change['delta']
        self.s['u'], self.s['p'] = update_vector(self.s['u'], deltas, self.s['initial'], self.cfg.eta, self.cfg.delta_bound)
        self.store.put('vectors', self.s['id'], {'u': self.s['u'], 'p': self.s['p'], 'changes': changes if allow else [],
                                                'applied_delta': {c: max(-self.cfg.delta_bound, min(self.cfg.delta_bound, v)) for c, v in deltas.items()},
                                                'reason': reason})

    def fingerprint(self, h):
        inputs = []
        for name in h['experiment']['inputs']:
            path = safe_path(self.store.root, name)
            inputs.append((name, digest(path.read_bytes())))
        return digest(canonical({'object': h['object_id'], 'procedure': h['experiment']['procedure'],
                                 'inputs': inputs, 'data_version': self.s['data_version']}))

    def start_experiment(self, h):
        self.remaining()
        if self.s['active'] or h['status'] != 'pending':
            raise ProtocolError('only one pending experiment may start')
        if not self.queue() or self.queue()[0]['id'] != h['id']:
            raise ProtocolError('choose queue head or explicitly reprioritize/suspend first')
        if self.s['budget']['experiments'] >= self.cfg.max_experiments:
            raise BudgetExhausted('experiment budget exhausted')
        fingerprint = self.fingerprint(h)
        if any(e['fingerprint'] == fingerprint for e in self.s['experiments']):
            h['status'] = 'suspended'
            h['resume_when'] = 'New input data or a materially different procedure; identical experiment blocked.'
            self.checkpoint('duplicate_experiment_blocked', {'hypothesis': h['id']})
            return
        if h['attempts_since_data'] >= self.cfg.max_attempts:
            h['status'] = 'suspended'
            self.checkpoint('branch_budget_exhausted', {'hypothesis': h['id']})
            return
        exp = {'id': uid('exp'), 'hypothesis_id': h['id'], 'fingerprint': fingerprint,
               'status': 'executing', 'tools': [], 'artifacts': [], 'report': None,
               'data_version': self.s['data_version'], 'started': time.time()}
        self.s['experiments'].append(exp)
        self.s['active'] = exp['id']
        self.s['budget']['experiments'] += 1
        h['attempts'] += 1
        h['attempts_since_data'] += 1
        h['status'] = 'active'
        h['experiments'].append(exp['id'])
        self.store.put('attempts', self.s['id'], {'id': exp['id'], 'hypothesis': h['id'], 'status': 'started'})
        self.checkpoint('experiment_started', {'id': exp['id'], 'hypothesis': h['id']})

    def experiment_context(self, exp):
        h = self.get_hypothesis(exp['hypothesis_id'])
        return {'task': self.s['task'], 'hypothesis': h, 'experiment_id': exp['id'],
                'targets': self.s['targets'], 'tools': exp['tools'], 'artifacts': exp['artifacts'],
                'observations': self.s['observations'][-8:], 'command_seconds': self.cfg.command_seconds}

    def artifact_snapshot(self, exp, tool_id):
        files = list(regular_files(self.store.root / 'work'))
        if len(files) > 256:
            raise ProtocolError('artifact file count exceeded')
        total = 0
        result = []
        for p in files:
            total += p.stat().st_size
            if total > self.cfg.material_bytes:
                raise ProtocolError('artifact byte budget exceeded')
            name = 'artifacts/' + exp['id'] + '/' + tool_id + '/' + p.relative_to(self.store.root / 'work').as_posix()
            target = safe_path(self.store.root, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, target)
            result.append(name)
        exp['artifacts'].extend(result)

    def execute_action(self, exp, action):
        if len(exp['tools']) >= self.cfg.max_actions:
            exp['forced_error'] = 'experiment action budget exhausted'
            exp['status'] = 'reporting'
            self.checkpoint('action_budget_exhausted')
            return
        self.remaining()
        if any(t['action'] == action for t in exp['tools'] if action['kind'] != 'status'):
            raise ProtocolError('identical action cannot be replayed within an experiment')
        tool = {'id': uid('tool'), 'action': action, 'status': 'running'}
        exp['tools'].append(tool)
        self.s['budget']['actions'] += 1
        self.store.put('tool_calls', self.s['id'], tool)
        self.checkpoint('tool_started', {'id': tool['id'], 'kind': action['kind']})
        kind = action['kind']
        if kind in ('run', 'http'):
            result = self.runner.run(action, tool['id'], self.remaining())
        elif kind == 'read':
            result = {'status': 'completed', 'output': self.read(action['path']), 'source': action['path']}
        elif kind == 'write':
            if not action['path'].startswith('work/'):
                raise ProtocolError('scripts may only be written to work/')
            p = safe_path(self.store.root, action['path'])
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(action['content'])
            p.chmod(0o666)
            script = f'scripts/{tool["id"]}.txt'
            safe_path(self.store.root, script).write_text(action['content'])
            result = {'status': 'completed', 'output': 'Script saved.', 'source': script}
        elif kind == 'status':
            found = next((t for t in exp['tools'] if t['id'] == action['tool_id'] and t['id'] != tool['id']), None)
            if not found:
                raise ProtocolError('unknown process/tool ID')
            result = {'status': 'completed', 'output': canonical(found.get('result', {'status': found['status']}))}
        else:
            raise ProtocolError('unsupported runner action')
        tool['status'] = 'completed'
        tool['result'] = result
        if kind in ('run', 'http'):
            self.artifact_snapshot(exp, tool['id'])
        if kind == 'http' and result['status'] == 'completed' and result['exit_code'] == 0:
            # The broker prints a parsed HTTP status header before the body.
            ob = self.observe(result['output'], [result['log']], 'new_data')
            self.s['web_evidence'].append(ob['id'])
        self.store.put('tool_calls', self.s['id'], tool)
        self.checkpoint('tool_finished', {'id': tool['id'], 'status': result['status']})
        if kind in ('run', 'http') and (result['status'] != 'completed' or result['exit_code'] != 0):
            # A crashing program is not a negative hypothesis test by itself.
            exp['forced_error'] = result['status'] if result['status'] != 'completed' else 'command_nonzero_exit'
            exp['status'] = 'reporting'
            self.checkpoint('tool_failure', {'experiment': exp['id'], 'reason': exp['forced_error']})

    def recover(self):
        for exp in self.s['experiments']:
            if exp['status'] not in ('executing', 'reporting'):
                continue
            for tool in exp['tools']:
                if tool['status'] == 'running':
                    self.runner.cleanup(tool['id'])
                    log = f'logs/{tool["id"]}.log'
                    if not safe_path(self.store.root, log).exists():
                        safe_path(self.store.root, log).write_text('Interrupted before durable tool completion. Not replayed.\n')
                    tool['status'] = 'completed'
                    tool['result'] = {'status': 'interrupted', 'exit_code': None, 'output': 'Interrupted; outcome unknown.', 'log': log}
                    exp['forced_error'] = 'interrupted; uncertain effect is not replayed'
                    exp['status'] = 'reporting'
                    self.store.put('tool_calls', self.s['id'], tool)
        self.checkpoint('recovered')

    def forced_report(self, exp):
        refs = [t['result'].get('log', t['id']) for t in exp['tools'] if t.get('result')] or [exp['id']]
        return {'hypothesis_id': exp['hypothesis_id'], 'experiment_id': exp['id'], 'status': 'tool_error',
                'observations': [{'text': exp['forced_error'], 'sources': refs}],
                'artifacts': exp['artifacts'], 'signal': 'inconclusive', 'changes': [],
                'next_step': 'Inspect tool failure; collect new data or change the procedure before retrying.'}

    def finish_experiment(self, exp, report):
        failures = [t for t in exp['tools'] if t['action']['kind'] in ('run', 'http') and
                    (t.get('result', {}).get('status') != 'completed' or t.get('result', {}).get('exit_code') != 0)]
        if failures:
            exp['forced_error'] = 'Tool failed or timed out; hypothesis signal is inconclusive.'
            report = self.forced_report(exp)
        h = self.get_hypothesis(exp['hypothesis_id'])
        if report['hypothesis_id'] != h['id'] or report['experiment_id'] != exp['id']:
            raise ProtocolError('report IDs do not match approved experiment')
        allowed = {exp['id']} | set(exp['artifacts']) | {t['id'] for t in exp['tools']}
        for t in exp['tools']:
            allowed.update(v for k, v in t.get('result', {}).items() if k in ('log', 'source'))
        for ob in report['observations']:
            if not set(ob['sources']) <= allowed:
                raise ProtocolError('report must cite this experiment sources')
        if not set(report['artifacts']) <= set(exp['artifacts']):
            raise ProtocolError('report invented artifacts')
        for change in report['changes']:
            if not set(change['evidence']) <= allowed:
                raise ProtocolError('delta must cite this experiment')
        informative = report['status'] == 'informative'
        if informative and report['signal'] == 'inconclusive':
            raise ProtocolError('informative report needs a positive or negative signal')
        if not informative and (report['signal'] != 'inconclusive' or report['changes']):
            raise ProtocolError('tool failures and uninformative results must be neutral')
        # Atomic finalization: hypotheses, observations, vectors, checks and completed marker.
        before = copy.deepcopy(self.s)
        self.store.db.execute('BEGIN IMMEDIATE')
        try:
            for ob in report['observations']:
                self.observe(ob['text'], ob['sources'], 'analysis' if informative else 'tool_error')
            self.apply_changes(report['changes'], exp['id'], allow=informative)
            h['status'] = ('confirmed' if report['signal'] == 'positive' else 'rejected') if informative else 'pending'
            if not informative:
                h['uninformative'] += 1
                if h['uninformative'] >= self.cfg.max_uninformative or h['attempts_since_data'] >= self.cfg.max_attempts:
                    h['status'] = 'suspended'
            exp['report'] = report
            exp['status'] = 'completed'
            self.s['active'] = None
            self.check_candidates(exp)
            self.store.put('attempts', self.s['id'], {'id': exp['id'], 'hypothesis': h['id'], 'status': 'completed', 'signal': report['signal']})
            self.store.put('hypotheses', self.s['id'], h)
            self.store.put('experiments', self.s['id'], exp)
            self.store.put('events', self.s['id'], {'event': 'experiment_completed', 'details': {'id': exp['id'], 'signal': report['signal'], 'hypothesis_status': h['status'], 'p': self.s['p']}, 'time': time.time()})
            self.store.save(self.s)
            self.store.db.execute('COMMIT')
        except BaseException:
            self.store.db.execute('ROLLBACK')
            self.s = before
            raise
        self.store.write_json(f'artifacts/{exp["id"]}/report.json', report)
        self.store.export_journal()

    def check_candidates(self, exp):
        sources = []
        for tool in exp['tools']:
            result = tool.get('result', {})
            # Only actual process output, never model-provided candidate claims.
            if tool['action']['kind'] in ('run', 'http') and result.get('status') == 'completed':
                sources.append(result['log'])
        sources += exp['artifacts']
        for source in sources:
            path = safe_path(self.store.root, source)
            if path.stat().st_size > self.cfg.material_bytes:
                continue
            for match in FLAG.finditer(path.read_text(errors='replace')):
                flag = match.group(0)
                if any(c['candidate'] == flag for c in self.s['checks']):
                    continue
                expected = self.cfg.expected_flag_sha256
                verified = bool(expected and digest(flag) == expected)
                check = {'id': uid('check'), 'candidate': flag, 'source': source, 'verified': verified,
                         'criterion': 'user_supplied_sha256' if expected else 'none',
                         'status': 'verified' if verified else 'candidate_unverified'}
                self.s['checks'].append(check)
                self.store.put('checks', self.s['id'], check)

    def continue_experiment(self):
        exp = next(e for e in self.s['experiments'] if e['id'] == self.s['active'])
        if exp['status'] == 'executing':
            if len(exp['tools']) >= self.cfg.max_actions:
                exp['forced_error'] = 'experiment action budget exhausted'
                exp['status'] = 'reporting'
                self.checkpoint('action_budget_exhausted')
                return
            action = self.call('executor', ACTION, self.experiment_context(exp))
            if action['kind'] == 'finish':
                if not any(t['action']['kind'] in ('run', 'http', 'read') for t in exp['tools']):
                    raise ProtocolError('cannot finish without an observed experiment')
                exp['status'] = 'reporting'
                self.checkpoint('execution_finished', {'summary': action['summary']})
            else:
                self.execute_action(exp, action)
        else:
            report = self.forced_report(exp) if exp.get('forced_error') else self.call('reporter', REPORT, self.experiment_context(exp))
            self.finish_experiment(exp, report)

    def step(self):
        self.remaining()
        if self.s['active']:
            self.continue_experiment()
            return
        decision = self.call('orchestrator', DECISION, self.context())
        self.evidence(decision['evidence'])
        self.checkpoint('decision', decision)
        kind, payload = decision['kind'], decision['payload']
        if kind in ('inspect', 'gather_info'):
            self.inspect(payload['path'])
        elif kind == 'interpret':
            if not self.s['category'] and not self.s.get('inferred_category'):
                category = payload['category']
                self.s['inferred_category'] = category
                self.s['initial'] = initial_vector(category)
                self.s['u'], self.s['p'] = dict(self.s['initial']), dict(self.s['initial'])
                self.store.put('vectors', self.s['id'], {'u': self.s['u'], 'p': self.s['p'], 'changes': [], 'reason': 'gemini_initial_interpretation', 'evidence': decision['evidence']})
                self.checkpoint('initial_direction_interpreted', {'category': category})
            self.propose(payload['assignment'])
        elif kind == 'experiment':
            self.start_experiment(self.get_hypothesis(payload['hypothesis_id']))
        elif kind == 'reprioritize':
            self.apply_changes(payload['changes'], 'orchestrator')
            self.checkpoint('reprioritized')
        elif kind == 'suspend':
            h = self.get_hypothesis(payload['hypothesis_id'])
            h['status'], h['resume_when'] = 'suspended', payload['resume_when']
            self.checkpoint('hypothesis_suspended', {'id': h['id']})
        else:
            status = kind
            if kind == 'finish':
                status = 'verified' if any(c['verified'] for c in self.s['checks']) else ('candidate_unverified' if self.s['checks'] else 'partial')
            self.s['status'], self.s['reason'] = status, payload['summary']
            self.checkpoint('finished', {'status': status, 'reason': payload['summary']})

    def run(self, stop_after=None):
        if self.s['status'] in TERMINAL:
            return self.summary()
        initial_done = sum(e['status'] == 'completed' for e in self.s['experiments'])
        try:
            self.recover()
            self.s['status'] = 'running'
            self.checkpoint('running')
            while self.s['status'] == 'running':
                self.step()
                done = sum(e['status'] == 'completed' for e in self.s['experiments'])
                if stop_after and done - initial_done >= stop_after and not self.s['active']:
                    self.s['status'] = 'paused'
                    self.checkpoint('paused_after_experiment')
            return self.summary()
        except BudgetExhausted as e:
            self.s['status'], self.s['reason'] = 'partial', str(e)
            self.checkpoint('budget_exhausted', {'reason': str(e)})
            return self.summary()
        except (ModelError, ProtocolError, ValueError, OSError) as e:
            self.s['status'] = 'blocked'
            self.s['reason'] = self.redact(f'{getattr(e, "code", type(e).__name__)}: {e}')
            self.checkpoint('blocked', {'reason': self.s['reason']})
            return self.summary()
        except KeyboardInterrupt:
            self.s['status'], self.s['reason'] = 'paused', 'Interrupted by user; uncertain actions will not be replayed.'
            self.checkpoint('interrupted')
            return self.summary()
        finally:
            self.store.export_journal()

    def summary(self):
        return {'run_id': self.s['id'], 'status': self.s['status'], 'reason': self.s['reason'],
                'budget_used': self.s['budget'], 'priorities': self.s['p'],
                'hypotheses': [{'id': h['id'], 'category': h['category'], 'hypothesis': h['hypothesis'],
                                'status': h['status'], 'attempts': h['attempts']} for h in self.s['hypotheses']],
                'checks': self.s['checks'], 'directory': str(self.store.root)}
