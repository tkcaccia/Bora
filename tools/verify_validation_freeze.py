#!/usr/bin/env python3
"""Verify every source byte bound by the 2026-11-12 validation freeze."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'docs/VALIDATION_FREEZE_2026-11-12.json'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(8 << 20):
            h.update(block)
    return h.hexdigest()


def main():
    spec = json.loads(MANIFEST.read_text())
    assert spec['format'] == 'bora-validation-code-freeze-v1'
    assert spec['tag'] == 'validation-freeze-2026-11-12'
    files = spec['source_sha256']
    assert len(files) >= 80
    for rel, expected in files.items():
        path = (ROOT / rel).resolve(strict=True)
        if not path.is_relative_to(ROOT) or digest(path) != expected:
            raise ValueError('Frozen source differs: ' + rel)
    print('VALIDATION_FREEZE_SOURCE_VERIFIED', len(files))


if __name__ == '__main__':
    main()
