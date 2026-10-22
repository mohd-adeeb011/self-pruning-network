"""Small helpers for reproducible experiment records."""
import hashlib
import json
from pathlib import Path


def source_digest():
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for package in ('engine', 'nn', 'prune', 'train'):
        for path in sorted((root / package).glob('*.py')):
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
