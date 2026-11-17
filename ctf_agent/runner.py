import hashlib
import os
from pathlib import Path
import selectors
import stat
import subprocess
import time
from .storage import safe_path


class DockerRunner:
    def __init__(self, root, config, targets):
        self.root = Path(root).resolve()
        self.config = config
        self.targets = targets

    @staticmethod
    def container_name(tool_id):
        return 'ctf-' + hashlib.sha256(tool_id.encode()).hexdigest()[:24]

    def cleanup(self, tool_id):
        try:
            subprocess.run(['docker', 'rm', '-f', self.container_name(tool_id)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            pass

    def command(self, action, tool_id, timeout):
        cfg = self.config
        network = 'none'
        mounts = []
        if action['kind'] == 'http':
            if {'host': action['host'], 'port': action['port']} not in self.targets:
                raise ValueError('HTTP endpoint is outside explicitly authorized targets')
            path = action['path']
            if not path.startswith('/') or path.startswith('//') or any(c in path for c in '\r\n'):
                raise ValueError('HTTP path must be an origin-relative path without CR/LF')
            # Only the fixed image-baked helper gets network; no writable mounts or model code.
            network = 'bridge'
            argv = ['python3', '-I', '/opt/ctf/http_fetch.py', action['host'], str(action['port']), path,
                    str(action['tls']).lower(), str(timeout), str(cfg.log_bytes)]
        else:
            argv = action['argv']
            mounts = ['--mount', f'type=bind,src={self.root / "originals"},dst=/materials,readonly',
                      '--mount', f'type=bind,src={self.root / "work"},dst=/work']
        # Container-side watchdog also survives an abrupt controller crash.
        argv = ['/usr/bin/timeout', '--signal=TERM', '--kill-after=1', f'{timeout}s', *argv]
        return ['docker', 'run', '--rm', '--pull=never', '--name', self.container_name(tool_id),
                '--network', network, '--user', '10001:10001', '--read-only', '--cap-drop=ALL',
                '--security-opt=no-new-privileges', '--pids-limit=64', '--cpus', str(cfg.cpus),
                '--memory', f'{cfg.memory_mb}m', '--memory-swap', f'{cfg.memory_mb}m',
                '--ulimit', 'nofile=128:128', '--ulimit', f'fsize={cfg.log_bytes}:{cfg.log_bytes}',
                '--tmpfs', '/tmp:rw,nosuid,nodev,noexec,size=32m', '--workdir', '/work',
                '--env', 'PYTHONDONTWRITEBYTECODE=1', *mounts, cfg.runner_image, *argv]

    def run(self, action, tool_id, remaining):
        timeout = min(action.get('timeout', self.config.command_seconds), self.config.command_seconds, remaining)
        if timeout <= 0:
            raise ValueError('no command budget left')
        cmd = self.command(action, tool_id, timeout)
        log = 'logs/' + tool_id + '.log'
        target = safe_path(self.root, log)
        start = time.monotonic()
        status, code, total = 'completed', None, 0
        preview = bytearray()
        process = None
        try:
            with open(target, 'wb') as output:
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
                with selectors.DefaultSelector() as selector:
                    selector.register(process.stdout, selectors.EVENT_READ)
                    while selector.get_map():
                        left = timeout - (time.monotonic() - start)
                        if left <= 0:
                            status = 'timeout'
                            break
                        ready = selector.select(min(left, 0.1))
                        for key, _ in ready:
                            chunk = os.read(key.fileobj.fileno(), 16384)
                            if not chunk:
                                selector.unregister(key.fileobj)
                                continue
                            space = self.config.log_bytes - total
                            saved = chunk[:space]
                            output.write(saved)
                            total += len(saved)
                            preview.extend(saved[:max(0, self.config.output_bytes - len(preview))])
                            if len(chunk) > space:
                                status = 'output_limit'
                                break
                        if status != 'completed':
                            break
                if status != 'completed':
                    self.cleanup(tool_id)
                    if process.poll() is None:
                        process.kill()
                try:
                    code = process.wait(timeout=max(0.01, timeout - (time.monotonic() - start)))
                except subprocess.TimeoutExpired:
                    status = 'timeout'
                    self.cleanup(tool_id)
                    process.kill()
                    code = process.wait(timeout=5)
                if status == 'completed' and code == 124:
                    status = 'timeout'
                if status == 'completed' and code in (125, 126, 127, 137):
                    status = 'tool_error'
        except OSError:
            status = 'tool_error'
            target.write_text('Docker executable or output file unavailable.\n')
            preview = bytearray(b'Docker executable or output file unavailable.')
        finally:
            if process and process.poll() is None:
                self.cleanup(tool_id)
                process.kill()
                process.wait(timeout=5)
            if process and process.stdout:
                process.stdout.close()
        return {'status': status, 'exit_code': code, 'output': preview.decode('utf-8', 'replace'),
                'log': log, 'bytes': total, 'truncated': total > len(preview) or status == 'output_limit',
                'seconds': time.monotonic() - start}


def regular_files(root):
    """Never follow links or read devices/FIFOs created by an experiment."""
    root = Path(root)
    for directory, folders, files in os.walk(root, followlinks=False):
        folders[:] = [n for n in folders if not (Path(directory) / n).is_symlink()]
        for name in files:
            p = Path(directory) / name
            if stat.S_ISREG(p.lstat().st_mode):
                yield p
