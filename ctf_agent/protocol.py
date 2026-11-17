"""Small strict JSON-schema subset, shared by prompts and runtime validation."""
import json
import math
from .config import CATEGORIES


class ProtocolError(ValueError):
    pass


def string(enum=None, max_length=8000):
    d = {'type': 'string', 'minLength': 1, 'maxLength': max_length}
    if enum:
        d['enum'] = list(enum)
    return d


def array(items, minimum=0, maximum=64):
    return {'type': 'array', 'items': items, 'minItems': minimum, 'maxItems': maximum}


def obj(**props):
    return {'type': 'object', 'properties': props, 'required': list(props), 'additionalProperties': False}


def number(low, high):
    return {'type': 'number', 'minimum': low, 'maximum': high}


REFS = array(string(max_length=300), 1)
VECTOR = obj(**{c: number(-1, 1) for c in CATEGORIES})
CHANGE = obj(category=string(CATEGORIES), delta=number(-1, 1), reason=string(), evidence=REFS)
EXPERIMENT = obj(procedure=string(), inputs=array(string(max_length=300), 1),
                 positive_signal=string(), negative_signal=string(),
                 uninformative_if=array(string(), 1), constraints=array(string(), 1))
HYPOTHESIS = obj(object_id=string(), hypothesis=string(), category=string(CATEGORIES),
                 evidence=REFS, alternatives=array(string(), 1), experiment=EXPERIMENT,
                 local_priority=number(0, 1), resume_when=string())
DECISION = obj(kind=string(('inspect', 'gather_info', 'interpret', 'experiment', 'reprioritize',
                           'suspend', 'finish', 'blocked', 'partial')),
               evidence=REFS, explanation=string(), payload={'type': 'object'})
ACTION_SCHEMAS = {
    'read': obj(kind=string(['read']), path=string()),
    'write': obj(kind=string(['write']), path=string(), content=string(max_length=60000)),
    'run': obj(kind=string(['run']), argv=array(string(), 1, 64), timeout=number(0.01, 3600)),
    'http': obj(kind=string(['http']), host=string(), port={'type': 'integer', 'minimum': 1, 'maximum': 65535},
                path=string(), tls={'type': 'boolean'}),
    'status': obj(kind=string(['status']), tool_id=string()),
    'finish': obj(kind=string(['finish']), summary=string()),
}
ACTION = {'oneOf': list(ACTION_SCHEMAS.values())}
OBS = obj(text=string(), sources=REFS)
REPORT = obj(hypothesis_id=string(), experiment_id=string(),
             status=string(('informative', 'uninformative', 'tool_error')),
             observations=array(OBS, 1), artifacts=array(string()),
             signal=string(('positive', 'negative', 'inconclusive')),
             changes=array(CHANGE, 0, 6), next_step=string())
DECISION_PAYLOADS = {
    'inspect': obj(path=string()), 'gather_info': obj(path=string()),
    'interpret': obj(assignment=string(), category=string(CATEGORIES)),
    'experiment': obj(hypothesis_id=string()),
    'reprioritize': obj(changes=array(CHANGE, 1, 6)),
    'suspend': obj(hypothesis_id=string(), resume_when=string()),
    'finish': obj(summary=string()), 'blocked': obj(summary=string()), 'partial': obj(summary=string()),
}


def validate(value, schema, path='$'):
    if 'oneOf' in schema:
        good = 0
        for variant in schema['oneOf']:
            try:
                validate(value, variant, path)
                good += 1
            except ProtocolError:
                pass
        if good != 1:
            raise ProtocolError(f'{path}: must match exactly one action schema')
        return value
    typ = schema.get('type')
    ok = {'object': isinstance(value, dict), 'array': isinstance(value, list),
          'string': isinstance(value, str), 'number': type(value) in (int, float),
          'integer': type(value) is int, 'boolean': type(value) is bool}.get(typ, False)
    if not ok:
        raise ProtocolError(f'{path}: expected {typ}')
    if 'enum' in schema and value not in schema['enum']:
        raise ProtocolError(f'{path}: unsupported value')
    if typ == 'object':
        props = schema.get('properties', {})
        if any(k not in value for k in schema.get('required', [])):
            raise ProtocolError(f'{path}: missing required field')
        if schema.get('additionalProperties') is False and set(value) - set(props):
            raise ProtocolError(f'{path}: unexpected fields')
        for key in value.keys() & props.keys():
            validate(value[key], props[key], f'{path}.{key}')
    elif typ == 'array':
        if not schema.get('minItems', 0) <= len(value) <= schema.get('maxItems', 1000):
            raise ProtocolError(f'{path}: invalid array length')
        for i, item in enumerate(value):
            validate(item, schema['items'], f'{path}[{i}]')
    elif typ == 'string':
        if not schema.get('minLength', 0) <= len(value) <= schema.get('maxLength', 100000):
            raise ProtocolError(f'{path}: invalid string length')
    elif typ in ('number', 'integer'):
        if not math.isfinite(value) or not schema.get('minimum', -math.inf) <= value <= schema.get('maximum', math.inf):
            raise ProtocolError(f'{path}: invalid number')
    return value


def parse(raw, schema):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ProtocolError('duplicate JSON key')
            result[key] = value
        return result
    try:
        value = json.loads(raw, object_pairs_hook=unique,
                           parse_constant=lambda _: (_ for _ in ()).throw(ProtocolError('non-finite JSON')))
    except (ValueError, TypeError, RecursionError) as e:
        raise ProtocolError('invalid model JSON') from e
    validate(value, schema)
    if schema is DECISION:
        validate(value['payload'], DECISION_PAYLOADS[value['kind']])
    return value
