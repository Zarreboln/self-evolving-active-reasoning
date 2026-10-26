"""Execute named, unmodified upstream definitions without provider imports.

Every source file is byte-identical to a pinned Git blob. This loader selects
definitions, not rewritten algorithms; dependencies are explicitly injected.
No upstream CLI, provider client, environment loader, or model is imported.
"""
from __future__ import annotations

import __future__
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def verified_source(relative_path):
    manifest = json.loads((ROOT / 'vendor-manifest.json').read_text(encoding='utf-8'))
    entry = next(x for x in manifest['files'] if x['path'] == relative_path)
    path = ROOT / 'vendor' / relative_path
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != entry['sha256']:
        raise ValueError('Pinned upstream source changed: ' + relative_path)
    return path, raw.decode('utf-8')


def definitions(relative_path, names, namespace=None, *, class_name=None):
    """Load exactly named functions/classes/constants; leave their AST unchanged."""
    path, source = verified_source(relative_path)
    tree = ast.parse(source, filename=str(path))
    body = tree.body
    if class_name is not None:
        body = next(n for n in body if isinstance(n, ast.ClassDef)
                    and n.name == class_name).body
    found = {}
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            found[node.name] = node
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    found[target.id] = node
    missing = set(names) - found.keys()
    if missing:
        raise ValueError('Missing upstream definitions: ' + repr(missing))
    selected = [found[name] for name in names]
    env = dict(namespace or {})
    module = ast.Module(body=selected, type_ignores=[])
    exec(compile(module, str(path), 'exec',
                 flags=__future__.annotations.compiler_flag), env)
    return env


def source_hashes():
    manifest = json.loads((ROOT / 'vendor-manifest.json').read_text(encoding='utf-8'))
    return {x['path']: x['sha256'] for x in manifest['files']}
