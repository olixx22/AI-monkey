from dataclasses import asdict, dataclass, fields
import ipaddress
import math
import re
import tomllib
from urllib.parse import urlsplit

CATEGORIES = ('crypto', 'web', 'reverse', 'pwn', 'forensics', 'misc')


@dataclass
class Config:
    mode: str = 'fake'
    gemini_model: str = 'gemini-4-hydrogen'
    gemini_output_tokens: int = 8192
    gemini_thinking_level: str = ''
    bridge_provider: str = 'tsuev'
    bridge_url: str = 'http://127.0.0.1:8000/v1'
    bridge_model: str = 'deepseek-chat'
    runner_image: str = 'ctf-agent-runner:local'
    max_experiments: int = 8
    max_model_calls: int = 60
    max_actions: int = 10
    max_seconds: float = 600
    command_seconds: float = 15
    model_seconds: float = 120
    output_bytes: int = 8192
    log_bytes: int = 1048576
    model_bytes: int = 131072
    material_bytes: int = 10485760
    max_uninformative: int = 2
    max_attempts: int = 4
    eta: float = 0.25
    delta_bound: float = 0.25
    cpus: float = 1
    memory_mb: int = 256
    # Explicit user verification criterion; empty means candidates only.
    expected_flag_sha256: str = ''

    def validate(self):
        if self.gemini_thinking_level not in ('', 'low', 'medium', 'high', 'xhigh'):
            raise ValueError('unsupported Gemini thinking level')
        if self.bridge_provider != 'tsuev':
            raise ValueError('bridge_provider must be tsuev; no silent bridge fallback')
        if self.mode not in ('fake', 'live'):
            raise ValueError('mode must be fake or live')
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, (float, int)) and (not math.isfinite(v) or v <= 0):
                raise ValueError(f'{f.name} must be positive and finite')
        for name in ('gemini_output_tokens', 'max_experiments', 'max_model_calls', 'max_actions', 'output_bytes',
                     'log_bytes', 'model_bytes', 'material_bytes', 'max_uninformative',
                     'max_attempts', 'memory_mb'):
            if type(getattr(self, name)) is not int:
                raise ValueError(f'{name} must be an integer')
        if self.eta > 1 or self.delta_bound > 1 or self.output_bytes > self.log_bytes:
            raise ValueError('eta/delta_bound <= 1 and output_bytes <= log_bytes required')
        if not re.fullmatch(r'[a-zA-Z0-9._-]+', self.gemini_model):
            raise ValueError('invalid Gemini model identifier')
        if not isinstance(self.runner_image, str) or self.runner_image.startswith('-'):
            raise ValueError('invalid runner image')
        u = urlsplit(self.bridge_url)
        if u.scheme != 'http' or u.hostname not in ('127.0.0.1', '::1', 'localhost') or u.username or u.password or u.query or u.fragment:
            raise ValueError('bridge_url must be a local loopback HTTP service, without credentials')
        if self.expected_flag_sha256 and not re.fullmatch('[0-9a-f]{64}', self.expected_flag_sha256):
            raise ValueError('expected_flag_sha256 must be a lowercase SHA-256 digest')
        return self

    @classmethod
    def load(cls, path=None):
        if path is None:
            return cls().validate()
        with open(path, 'rb') as f:
            data = tomllib.load(f)
        if set(data) - {f.name for f in fields(cls)}:
            raise ValueError('unknown configuration key')
        return cls(**data).validate()

    def to_dict(self):
        return asdict(self)


def parse_target(value):
    """Numeric IP:port only. No DNS rebinding, CIDRs, or implied port ranges."""
    host, port = value.rsplit(':', 1)
    host = str(ipaddress.ip_address(host.strip('[]')))
    port = int(port)
    if not 1 <= port <= 65535:
        raise ValueError('target port must be 1..65535')
    return {'host': host, 'port': port}


def initial_vector(category=None):
    return {c: (0.8 if c == category else 0.04) if category else 1 / 6 for c in CATEGORIES}


def update_vector(u, delta, initial, eta, bound):
    v = {c: min(1., max(0., u[c] + eta * max(-bound, min(bound, delta.get(c, 0))))) for c in CATEGORIES}
    if sum(v.values()) == 0:
        v = dict(initial)
    total = sum(v.values())
    return v, {c: v[c] / total for c in CATEGORIES}
