"""Shared, strict YAML configuration for the local ACT training project."""
from __future__ import annotations

from copy import deepcopy
import math
import os
from pathlib import Path
import re

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / 'configs/train.yaml'
SCHEMA = {
    'dataset': 'source root split_file repo_id train_ratio seed video_backend',
    'policy': 'type device vision_backbone pretrained_backbone_weights chunk_size n_action_steps optimizer_lr optimizer_lr_backbone kl_weight',
    'training': 'run_name output_root output_dir steps batch_size num_workers prefetch_factor mixed_precision seed resume_config',
    'logging': 'every_steps', 'checkpoint': 'every_steps',
    'plot': 'enabled formats dpi smoothing_window',
    'monitor': 'interval_seconds temperature_limit_c min_host_free_mib',
    'evaluation': 'enabled samples_per_episode batch_size num_workers all_frames',
    'benchmark': 'enabled report_dir batches warmup steps headroom_gib',
}


def resolve_path(value: str | Path) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    return (path if path.is_absolute() else ROOT / path).resolve()


def _merge(base: dict, overrides: dict, prefix: str = '') -> dict:
    if not isinstance(overrides, dict):
        raise ValueError(f'{prefix or "Configuration"} must be a mapping')
    result = deepcopy(base)
    for key, value in overrides.items():
        location = f'{prefix}.{key}' if prefix else str(key)
        if key not in base:
            raise ValueError(f'Unknown configuration key: {location}')
        if isinstance(base[key], dict):
            result[key] = _merge(base[key], value, location)
        else:
            result[key] = value
    return result


def _number(value, name: str, *, integer: bool = False, minimum: float = 0, inclusive: bool = False):
    valid = isinstance(value, int if integer else (int, float)) and not isinstance(value, bool)
    if not valid or not math.isfinite(value) or (value < minimum if inclusive else value <= minimum):
        comparison = '>=' if inclusive else '>'
        raise ValueError(f'{name} must be a finite {"integer" if integer else "number"} {comparison} {minimum}')


def validate_config(config: dict) -> dict:
    expected = set(SCHEMA) | {'schema_version'}
    if set(config) != expected:
        raise ValueError(f'Configuration keys differ: unknown={set(config)-expected}, missing={expected-set(config)}')
    for section, names in SCHEMA.items():
        if not isinstance(config[section], dict):
            raise ValueError(f'{section} must be a mapping')
        expected_fields = set(names.split())
        if set(config[section]) != expected_fields:
            raise ValueError(f'Invalid keys in {section}: unknown={set(config[section])-expected_fields}, '
                             f'missing={expected_fields-set(config[section])}')
    if config['schema_version'] != 1:
        raise ValueError('Only schema_version: 1 is supported')
    ds, policy, train = config['dataset'], config['policy'], config['training']
    for section, fields in (('dataset', ('source', 'root', 'split_file', 'repo_id', 'video_backend')),
                            ('training', ('run_name', 'output_root')),
                            ('benchmark', ('report_dir',))):
        for field in fields:
            if not isinstance(config[section][field], str) or not config[section][field].strip():
                raise ValueError(f'{section}.{field} must be a nonempty string')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', train['run_name']):
        raise ValueError('training.run_name may contain only letters, digits, underscore, dot and hyphen')
    if train['run_name'] in {'.', '..'}:
        raise ValueError('training.run_name must name a run, not a directory traversal')
    for field in ('output_dir', 'resume_config'):
        value = train[field]
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f'training.{field} must be null or a nonempty path string')
    _number(ds['train_ratio'], 'dataset.train_ratio')
    if ds['train_ratio'] >= 1:
        raise ValueError('dataset.train_ratio must be between 0 and 1')
    for section, field in (('dataset', 'seed'), ('training', 'seed'), ('training', 'num_workers'),
                           ('evaluation', 'num_workers')):
        _number(config[section][field], f'{section}.{field}', integer=True, inclusive=True)
    for section, field in (('training', 'steps'), ('training', 'prefetch_factor'),
                           ('logging', 'every_steps'), ('checkpoint', 'every_steps'),
                           ('policy', 'chunk_size'), ('policy', 'n_action_steps'),
                           ('plot', 'dpi'), ('plot', 'smoothing_window'),
                           ('evaluation', 'samples_per_episode'), ('evaluation', 'batch_size'),
                           ('benchmark', 'warmup'), ('benchmark', 'steps')):
        _number(config[section][field], f'{section}.{field}', integer=True)
    if train['batch_size'] != 'auto':
        _number(train['batch_size'], 'training.batch_size', integer=True)
    if train['mixed_precision'] not in {'bf16', 'fp16', 'no'}:
        raise ValueError('training.mixed_precision must be bf16, fp16, or "no" (quoted in YAML)')
    if policy['type'] != 'act' or policy['device'] not in {'cuda', 'cuda:0'}:
        raise ValueError('This project supports ACT on a single CUDA GPU (cuda or cuda:0)')
    if policy['n_action_steps'] > policy['chunk_size']:
        raise ValueError('policy.n_action_steps must not exceed policy.chunk_size')
    if not isinstance(policy['vision_backbone'], str) or not policy['vision_backbone']:
        raise ValueError('policy.vision_backbone must be a model name')
    weights = policy['pretrained_backbone_weights']
    if weights is not None and not isinstance(weights, str):
        raise ValueError('policy.pretrained_backbone_weights must be null or a torchvision weights name')
    for field in ('optimizer_lr', 'optimizer_lr_backbone'):
        _number(policy[field], f'policy.{field}')
    _number(policy['kl_weight'], 'policy.kl_weight', inclusive=True)
    for section, field in (('monitor', 'interval_seconds'), ('monitor', 'temperature_limit_c'),
                           ('monitor', 'min_host_free_mib'), ('benchmark', 'headroom_gib')):
        _number(config[section][field], f'{section}.{field}')
    for section, field in (('plot', 'enabled'), ('evaluation', 'enabled'),
                           ('evaluation', 'all_frames'), ('benchmark', 'enabled')):
        if not isinstance(config[section][field], bool):
            raise ValueError(f'{section}.{field} must be true or false')
    formats = config['plot']['formats']
    if not isinstance(formats, list) or not formats or any(x not in {'png', 'pdf', 'svg'} for x in formats):
        raise ValueError('plot.formats must be a nonempty list drawn from png, pdf, svg')
    if len(set(formats)) != len(formats):
        raise ValueError('plot.formats must not contain duplicates')
    batches = config['benchmark']['batches']
    if not isinstance(batches, list) or not batches:
        raise ValueError('benchmark.batches must be a nonempty list')
    for batch in batches:
        _number(batch, 'benchmark.batches item', integer=True)
    return config


def load_config(path: str | Path | None = None) -> dict:
    """Load defaults, then merge an optional partial override, rejecting typos."""
    defaults = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding='utf-8'))
    if not isinstance(defaults, dict):
        raise ValueError(f'Invalid default configuration: {DEFAULT_CONFIG}')
    chosen = DEFAULT_CONFIG if path is None else resolve_path(path)
    if chosen == DEFAULT_CONFIG.resolve():
        result = deepcopy(defaults)
    else:
        overrides = yaml.safe_load(chosen.read_text(encoding='utf-8'))
        result = _merge(defaults, overrides)
    return validate_config(result)


def dump_config(config: dict, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding='utf-8')
