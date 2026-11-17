import argparse
import getpass
import os
from dataclasses import fields
import json
import math
from pathlib import Path
import sys
import tempfile
import time

from .adapters import LiveModels, ModelError
from .config import CATEGORIES, Config, parse_target
from .engine import Engine, create_run
from .protocol import obj, string, parse
from .storage import Store, digest, safe_path

DEMO_FLAG = 'flag{separate_roles_real_experiments}'


def demo_materials(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'cipher.hex').write_text(bytes(x ^ 73 for x in DEMO_FLAG.encode()).hex() + '\n')
    return directory


def print_json(value):
    print(json.dumps(value, ensure_ascii=False, indent=2))


def run_engine(directory, stop_after=None):
    engine = Engine(directory)
    try:
        return engine.run(stop_after)
    finally:
        engine.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description='Sequential authorized CTF agent (Python + Docker + SQLite).')
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('run', help='Create a run and start the research loop')
    run.add_argument('--task-file', required=True)
    run.add_argument('--materials', required=True)
    run.add_argument('--run-dir', required=True)
    run.add_argument('--category', choices=CATEGORIES)
    run.add_argument('--target', action='append', default=[], metavar='IP:PORT')
    run.add_argument('--config')
    run.add_argument('--prompt-key', action='store_true', help='Read Gemini key without echo for this process')
    run.add_argument('--mode', choices=('fake', 'live'))
    run.add_argument('--stop-after', type=int)
    for name in ('max_experiments', 'max_model_calls', 'max_seconds', 'command_seconds'):
        run.add_argument('--' + name.replace('_', '-'), type=float if 'seconds' in name else int)
    run.add_argument('--expected-flag-sha256')
    demo = sub.add_parser('demo', help='Real Docker experiment with fake models, no external model service')
    demo.add_argument('--run-dir', required=True)
    demo.add_argument('--unverified', action='store_true', help='Do not supply the demo verification criterion')
    demo.add_argument('--stop-after', type=int)
    resume = sub.add_parser('resume', help='Resume without replaying completed/uncertain effects')
    resume.add_argument('run_dir')
    resume.add_argument('--stop-after', type=int)
    resume.add_argument('--prompt-key', action='store_true', help='Read Gemini key without echo for this process')
    resume.add_argument('--extend-seconds', type=float, default=0,
                        help='Explicitly add user-authorized wall-time budget, measured from now if expired')
    sub.add_parser('status', help='Read durable run summary').add_argument('run_dir')
    sub.add_parser('journal', help='Export and print the hypothesis/vector journal').add_argument('run_dir')
    add = sub.add_parser('add-material', help='Append immutable new evidence and wake related suspended hypotheses')
    add.add_argument('run_dir')
    add.add_argument('file')
    add.add_argument('--related-to', required=True, help='Existing object ID, for example originals/cipher.hex')
    live = sub.add_parser('live-check', help='Explicit, optional account-dependent text-only smoke check')
    live.add_argument('--config')
    live.add_argument('--provider', choices=('gemini', 'bridge'), required=True)
    live.add_argument('--prompt-key', action='store_true', help='Read Gemini key without echo; keep only in process memory')
    bridge = sub.add_parser('bridge', help='Manage separately installed Tsuev DeepSeek service')
    bridge.add_argument('action', choices=('status', 'login', 'start'))
    bridge.add_argument('--directory')
    bridge.add_argument('--port', type=int, default=8000)
    args = parser.parse_args(argv)
    old_key = os.environ.get('GEMINI_API_KEY')
    prompted = False
    try:
        if getattr(args, 'prompt_key', False):
            if (args.command == 'live-check' and args.provider != 'gemini') or not sys.stdin.isatty():
                raise ValueError('--prompt-key requires Gemini and an interactive terminal')
            os.environ['GEMINI_API_KEY'] = getpass.getpass('Gemini API key (hidden): ')
            prompted = True
        if getattr(args, 'stop_after', None) is not None and args.stop_after <= 0:
            raise ValueError('stop-after must be positive')
        if args.command == 'bridge':
            from .bridge import DEFAULT_DIRECTORY, launch, status
            directory = args.directory or DEFAULT_DIRECTORY
            if args.action == 'status':
                if not 1 <= args.port <= 65535:
                    raise ValueError('port must be 1..65535')
                print_json(status(f'http://127.0.0.1:{args.port}/v1', directory))
                return 0
            return launch(args.action, directory, args.port)
        if args.command == 'run':
            cfg = Config.load(args.config)
            for field in fields(cfg):
                value = getattr(args, field.name, None)
                if value is not None:
                    setattr(cfg, field.name, value)
            cfg.validate()
            create_run(args.run_dir, Path(args.task_file).read_text(), args.materials, args.category,
                       [parse_target(t) for t in args.target], cfg)
            result = run_engine(args.run_dir, args.stop_after)
        elif args.command == 'demo':
            cfg = Config(expected_flag_sha256='' if args.unverified else digest(DEMO_FLAG))
            with tempfile.TemporaryDirectory(prefix='ctf-demo-input-') as tmp:
                create_run(args.run_dir, 'Crypto: recover the flag from cipher.hex. Suspect single-byte XOR.',
                           demo_materials(tmp), 'crypto', [], cfg)
            result = run_engine(args.run_dir, args.stop_after)
        elif args.command == 'resume':
            engine = Engine(args.run_dir)
            try:
                if not math.isfinite(args.extend_seconds) or args.extend_seconds < 0:
                    raise ValueError('extend-seconds must be nonnegative')
                if args.extend_seconds:
                    engine.s['deadline'] = max(engine.s['deadline'], time.time()) + args.extend_seconds
                    if engine.s['status'] == 'partial' and 'budget exhausted' in engine.s['reason']:
                        engine.s['status'] = 'paused'
                    engine.checkpoint('user_extended_time', {'seconds': args.extend_seconds})
                result = engine.run(args.stop_after)
            finally:
                engine.close()
        elif args.command in ('status', 'journal'):
            if not (Path(args.run_dir) / 'state.sqlite3').is_file():
                raise ValueError('run does not exist')
            store = Store(args.run_dir)
            try:
                state = store.load()
                if args.command == 'journal':
                    store.export_journal()
                    result = json.loads((store.root / 'journal.json').read_text())
                else:
                    result = {k: state[k] for k in ('id', 'status', 'reason', 'budget', 'p', 'hypotheses', 'checks', 'active')}
            finally:
                store.close()
        elif args.command == 'add-material':
            engine = Engine(args.run_dir)
            try:
                source = Path(args.file)
                if source.is_symlink() or not source.is_file():
                    raise ValueError('new material must be a regular file')
                if args.related_to not in {o['id'] for o in engine.s['objects']}:
                    raise ValueError('unknown related object')
                if source.stat().st_size + sum(o['size'] for o in engine.s['objects']) > engine.cfg.material_bytes:
                    raise ValueError('material budget exceeded')
                data = source.read_bytes()
                name = f'originals/added-{digest(data)[:16]}-{source.name}'
                if any(o['id'] == name or o['sha256'] == digest(data) for o in engine.s['objects']):
                    raise ValueError('unchanged material already exists')
                target = safe_path(engine.store.root, name)
                target.write_bytes(data)
                target.chmod(0o444)
                item = {'id': name, 'sha256': digest(data), 'size': len(data), 'source': name}
                engine.s['objects'].append(item)
                engine.store.put('objects', engine.s['id'], item)
                engine.store.put('relations', engine.s['id'], {'from': name, 'to': args.related_to, 'kind': 'user_supplied_related_data'})
                engine.observe(data[:engine.cfg.output_bytes].decode('utf-8', 'replace') or '(empty)', [name], 'new_data', args.related_to)
                engine.store.write_json('manifest.json', engine.s['objects'])
                if engine.s['status'] in ('blocked', 'partial'):
                    engine.s['status'] = 'paused'
                engine.checkpoint('new_material_added', {'path': name, 'related_to': args.related_to})
                result = engine.summary()
            finally:
                engine.close()
        else:
            cfg = Config.load(args.config)
            role = 'orchestrator' if args.provider == 'gemini' else 'interpreter'
            cfg.gemini_output_tokens = 2048
            cfg.gemini_thinking_level = 'low'
            schema = obj(status=string(['ok']))
            raw, _ = LiveModels(cfg).complete(role, schema, {'instruction': 'Return {"status":"ok"}.'}, None, cfg.model_seconds)
            result = {'provider': args.provider, 'result': parse(raw, schema)}
        print_json(result)
        return 2 if result.get('status') == 'blocked' else 0
    except (ValueError, OSError, ModelError) as error:
        print(f'Error: {error}', file=sys.stderr)
        return 2
    finally:
        if prompted:
            if old_key is None:
                os.environ.pop('GEMINI_API_KEY', None)
            else:
                os.environ['GEMINI_API_KEY'] = old_key


if __name__ == '__main__':
    raise SystemExit(main())
