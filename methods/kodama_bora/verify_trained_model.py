#!/usr/bin/env python3
"""Accept a fastPLS model only if it uses Benchmark 1's exact training groups."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
FROZEN = HERE / 'TRAINING_SET_FROZEN.json'


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def verify(manifest_path: Path, model_path: Path) -> dict:
    frozen = read(FROZEN)
    manifest = read(manifest_path)
    if (frozen['status'] != 'TRAINING_MEMBERSHIP_FROZEN_FEATURES_PENDING' or
            manifest.get('format') != 'bora-training-v1' or
            manifest.get('classes') != frozen['classes']):
        raise ValueError('Benchmark 1 training contract or classifier classes differ')
    expected = {case['case']: case for case in frozen['cases']}
    rows = manifest.get('cases', [])
    if (len(rows) != frozen['admitted_training_images'] or
            {row.get('case') for row in rows} != set(expected)):
        raise ValueError('Classifier training case membership differs from Benchmark 1')
    descriptors, targets, descriptor_hashes = [], [], {}
    for row in rows:
        case = expected[row['case']]
        if row.get('split') != 'train' or row.get('source_group') != case['source_group']:
            raise ValueError('Training source group differs: ' + row['case'])
        labels = {str(group['anonymous_group']): group['class_id'] for group in case['groups']
                  if group['class_id'] is not None}
        if row.get('labels') != labels:
            raise ValueError('Training group targets differ: ' + row['case'])
        for key in ('spatial_tokens', 'bora_geojson', 'calibration_receipt'):
            source = Path(row[key]).resolve(strict=True)
            expected_sha = (case['calibration_receipt_sha256'] if key == 'calibration_receipt'
                            else row[key + '_sha256'])
            if sha(source) != expected_sha:
                raise ValueError('Training ' + key + ' source changed: ' + row['case'])
        path = (manifest_path.parent / row['descriptors']).resolve(strict=True)
        digest = sha(path)
        if digest != row.get('sha256'):
            raise ValueError('Training descriptor hash differs: ' + row['case'])
        with np.load(path, allow_pickle=False) as saved:
            if not {'x', 'group_ids', 'metadata'} <= set(saved.files):
                raise ValueError('Incomplete training descriptor: ' + row['case'])
            meta = json.loads(str(saved['metadata'].item()))
            if (meta.get('format') != 'bora-neighbor-context-descriptors-v1' or
                    meta.get('benchmark1_training_set_sha256') != sha(FROZEN) or
                    meta.get('case') != row['case'] or
                    meta.get('calibrated_image_sha256') != case['calibrated_image_sha256'] or
                    meta.get('source_group') != case['source_group'] or
                    meta.get('spatial_tokens_sha256') != row.get('spatial_tokens_sha256') or
                    meta.get('bora_geojson_sha256') != row.get('bora_geojson_sha256')):
                raise ValueError('Training descriptor source identity differs: ' + row['case'])
            ids = saved['group_ids'].astype(np.int64)
            x = saved['x'].astype(np.float64)
            if (ids.tolist() != sorted(map(int, labels)) or x.shape != (len(ids), 2560) or
                    not np.isfinite(x).all()):
                raise ValueError('Training descriptor group geometry differs: ' + row['case'])
            descriptors.append(x)
            targets.append(np.asarray([labels[str(int(g))] for g in ids], np.int64))
        descriptor_hashes[row['case']] = digest
    if sum(len(y) for y in targets) != frozen['class_eligible_groups']:
        raise ValueError('Training group count differs')
    with np.load(model_path, allow_pickle=False) as model:
        if set(model.files) != {'x', 'y', 'metadata'}:
            raise ValueError('Not a native Bora fastPLS classifier artifact')
        info = json.loads(str(model['metadata'].item()))
        if (info.get('format') != 'bora-neighbor-context-v1' or
                info.get('fastpls_version') != '0.3.0' or
                info.get('backend') != 'cpu' or
                info.get('classes') != frozen['classes'] or
                info.get('training_provenance', {}).get('manifest_sha256') != sha(manifest_path) or
                not np.array_equal(model['x'], np.concatenate(descriptors)) or
                not np.array_equal(model['y'], np.concatenate(targets))):
            raise ValueError('Fitted PLS artifact does not match exact training descriptors')
    return dict(status='ACCEPTED', accepted_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                model_sha256=sha(model_path), model_path=str(model_path),
                training_set_sha256=sha(FROZEN), training_manifest_sha256=sha(manifest_path),
                training_manifest_path=str(manifest_path), training_images=len(rows),
                training_groups=frozen['class_eligible_groups'],
                classes=frozen['classes'], descriptor_sha256=descriptor_hashes,
                fit_scope='Benchmark 1 admitted training images only',
                benchmark2_use='whole anonymous Bora group tissue-type prediction; '
                               'do not claim validated biological accuracy on a WSI without ground truth')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('manifest', type=Path)
    parser.add_argument('model', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    receipt = verify(args.manifest, args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + '.partial')
    temporary.write_text(json.dumps(receipt, indent=2) + '\n')
    os.replace(temporary, args.output)
    print(json.dumps({key: receipt[key] for key in ('status', 'training_images',
                                                   'training_groups', 'model_sha256')}, indent=2))


if __name__ == '__main__':
    main()
