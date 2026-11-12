#!/usr/bin/env python3
"""Independent bounded, streaming audit of the 13 public ridge WSI outputs."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

ROOT = Path('/mnt/sata_ssd/ChatGPT/Bora')
DATA = Path('/mnt/sata_ssd/ChatGPT/Bora-Data')
HERE = Path(__file__).resolve().parent
CAMPAIGN = ROOT / 'migration/benchmark2-unified-wsi-panels-2026-11-11-v17/CAMPAIGN_VERIFIED.json'
FINAL = ROOT / 'migration/benchmark2-supervised-ridge-2026-11-12-v6/CAMPAIGN_COMPLETE.json'
OUT = DATA / 'benchmark2-supervised-ridge-2026-11-12-v5'
RECEIPT = HERE / 'FULL_SLIDE_AUDIT_VERIFIED.json'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def audit_slide(slug, campaign_row):
    folder = OUT / slug
    receipt_path = folder / 'FULL_SLIDE_VERIFIED.json'
    info = json.loads(receipt_path.read_text())
    assert info['status'] == 'SUPERVISED_BORA_RIDGE_FULL_SLIDE_VECTOR_VERIFIED'
    assert info['slug'] == slug and info['pm016_included'] is False
    assert info['whole_slide_expert_accuracy'] is None
    assert sha(receipt_path) == campaign_row['receipt_sha256']
    assert info['class_ids'] == info['class_selection']['retained_class_ids']
    assert info['class_selection']['minimum_fraction'] == .02
    assert info['prevector_cleanup']['minimum_area_um2'] == 2000
    assert info['postvector_cleanup']['minimum_area_um2'] == 2000
    assert info['contour_smoothing']['sigma_um'] == 6
    geometry = info['independent_geometry_audit']
    assert geometry['native_geojson_roundtrip_different_pixels'] == 0
    assert geometry['outside_reviewed_support_pixels'] == 0
    assert all(row['below_2000um2'] == 0 for row in geometry['positive_and_hole_components'])
    assert info['native_vector']['minimum_polygon_um2'] >= 2000
    names = {'mask_tiff': 'supervised-bora.mask-0p5um.tif',
             'native_geojson': 'supervised-bora.native.geojson',
             'exact_mask': 'supervised-bora.final-2000um2.uint8.dat',
             'working_geojson': 'supervised-bora.working.geojson'}
    for key, name in names.items():
        assert sha(folder / name) == info['output_sha256'][key], (slug, key)
    assert info['output_sha256']['mask_tiff'] == campaign_row['mask_tiff_sha256']
    assert info['output_sha256']['native_geojson'] == campaign_row['native_geojson_sha256']
    height, width = info['calibrated_shape_yx']
    mask_path = folder / names['exact_mask']
    support_path = folder / 'reviewed-support.uint8.dat'
    assert mask_path.stat().st_size == support_path.stat().st_size == height * width
    mask = np.memmap(mask_path, mode='r', dtype=np.uint8, shape=(height, width))
    support = np.memmap(support_path, mode='r', dtype=np.uint8, shape=(height, width))
    counts = {int(i): 0 for i in [0] + info['class_ids']}
    outside = 0
    with rasterio.open(folder / names['mask_tiff']) as tif:
        assert tif.count == 1 and tif.dtypes == ('uint8',)
        assert (tif.height, tif.width) == (height, width)
        for y in range(0, height, 512):
            h = min(512, height - y)
            for x in range(0, width, 512):
                w = min(512, width - x)
                raw = mask[y:y+h, x:x+w]
                decoded = tif.read(1, window=Window(x, y, w, h))
                if not np.array_equal(decoded, raw):
                    raise ValueError(f'{slug}: TIFF mismatch at ({x},{y})')
                outside += int(np.count_nonzero((raw > 0) & (support[y:y+h, x:x+w] == 0)))
                unique, n = np.unique(raw, return_counts=True)
                for value, count in zip(unique, n):
                    if int(value) not in counts:
                        raise ValueError(f'{slug}: unselected class {value}')
                    counts[int(value)] += int(count)
    assert outside == 0 and sum(counts.values()) == height * width
    assert all(counts[i] > 0 for i in info['class_ids'])
    return {'slug': slug, 'receipt_sha256': sha(receipt_path),
            'shape_yx': [height, width], 'class_pixel_counts': counts,
            'tiff_exact_pixel_differences': 0,
            'positive_pixels_outside_reviewed_support': 0,
            'native_geojson_sha256': info['output_sha256']['native_geojson']}


def main():
    if RECEIPT.exists():
        raise FileExistsError('Audit receipt already exists')
    campaign = json.loads(CAMPAIGN.read_text())
    final = json.loads(FINAL.read_text())
    assert final['status'] == 'THIRTEEN_PUBLIC_RIDGE_FULL_SLIDES_COMPLETE'
    assert final['campaign_sha256'] == sha(CAMPAIGN)
    assert final['pm016_included'] is False and final['whole_slide_expert_accuracy'] is None
    pages = [r['slug'] for r in campaign['pages']]
    assert len(set(pages)) == len(pages) == 13
    assert all('pm016' not in slug.lower() for slug in pages)
    rows = {r['slug']: r for r in final['slides']}
    assert set(rows) == set(pages)
    audited = []
    for slug in pages:
        row = audit_slide(slug, rows[slug])
        audited.append(row)
        print('AUDITED', slug, row['class_pixel_counts'], flush=True)
    obj = {'status': 'THIRTEEN_PUBLIC_RIDGE_FULL_SLIDES_INDEPENDENTLY_AUDITED',
           'campaign_sha256': sha(CAMPAIGN), 'controller_receipt_sha256': sha(FINAL),
           'audit_code_sha256': sha(__file__), 'slides': audited,
           'tiff_exact_pixel_differences': 0,
           'positive_pixels_outside_reviewed_support': 0,
           'pm016_included': False, 'whole_slide_expert_accuracy': None}
    part = RECEIPT.with_suffix('.partial.json')
    part.write_text(json.dumps(obj, indent=2) + '\n')
    os.replace(part, RECEIPT)


if __name__ == '__main__':
    main()
