#!/usr/bin/env python3
"""Complete the selected ridge whole-slide mask/GeoJSON route for public slides.

This version composes the audited training-only projection, physical contour
and global vector functions. It fixes the v30/vector one-pixel gap failure by
repairing only uncovered geometry, as independently verified on ovarian v32.
All work is CPU-local on Chiamaka and one complete slide at a time.
"""
from __future__ import annotations

import gc
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from shapely.geometry import shape
from shapely.ops import transform

ROOT = Path('/mnt/sata_ssd/ChatGPT/Bora')
DATA = Path('/mnt/sata_ssd/ChatGPT/Bora-Data')
SITE = Path('/tmp/bora-analysis-bw106-v2')
HERE = Path(__file__).resolve().parent
OUT = DATA / 'benchmark2-supervised-ridge-2026-11-12-v5'
V30 = ROOT / 'migration/benchmark2-supervised-ridge-2026-11-12-v5/supervised_wsi_pipeline.py'
V32 = ROOT / 'migration/benchmark2-supervised-final-2026-11-11-v32/repair_unoccupied_gaps.py'
for name, path in [('prior', V30), ('gap_repair', V32)]:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    globals()[name] = module


def sha(path: Path) -> str:
    return prior.sha(path)


def pinned_inputs(slug: str, slide: Path) -> dict:
    """Verify the five public 10x sources with archived Bora vectors."""
    support_receipts = list(DATA.glob(
        f'benchmark2-unified-wsi-panels-2026-11-11-v8/shared-bora-support/{slug}/SHARED_SUPPORT_VERIFIED.json'))
    if len(support_receipts) != 1:
        raise ValueError('Expected pinned Bora source geometry receipt')
    support_info = json.loads(support_receipts[0].read_text())
    version = 'v9' if slug in {'visium_hd_human_kidney_ffpe',
                               'visium_hd_human_lung_cancer_fixed_frozen'} else 'v8'
    calibration_path = (DATA / f'benchmark2-unified-wsi-panels-2026-11-11-{version}' /
                        'calibrated' / slug / 'CALIBRATION_VERIFIED.json')
    calibration = json.loads(calibration_path.read_text())
    if (calibration['status'] != 'VERIFIED_LOCAL_CALIBRATED_PUBLIC_HE' or
            calibration['slug'] != slug):
        raise ValueError('Pinned local H&E calibration differs')
    source_path = Path(calibration['new_calibrated_path'])
    if sha(source_path) != calibration['new_calibrated_sha256']:
        raise ValueError('Pinned local H&E SHA256 differs')
    commit, relative = support_info['pinned_geojson_commit'], support_info['pinned_geojson_path']
    compressed = subprocess.check_output(['git', 'show', f'{commit}:{relative}'], cwd=SITE)
    if hashlib.sha256(compressed).hexdigest() != support_info['pinned_geojson_gzip_sha256']:
        raise ValueError('Pinned compressed Bora GeoJSON SHA256 differs')
    payload = gzip.decompress(compressed)
    if hashlib.sha256(payload).hexdigest() != support_info['pinned_geojson_sha256']:
        raise ValueError('Pinned Bora GeoJSON SHA256 differs')
    geo_path = slide / 'pinned-bora-support-source.geojson'
    with geo_path.open('xb') as f:
        f.write(payload)
    geoj = json.loads(payload)
    if (geoj['coordinate_space'] != 'original_WSI_native_pixel_edges' or
            geoj['native_origin_xy'] != [0, 0] or
            geoj['full_slide_input_identity']['shape_yx'] != calibration['working_shape_yx'] or
            geoj['working_mpp_xy'] != calibration['working_mpp_xy']):
        raise ValueError('Pinned Bora working geometry differs from calibration')
    sx, sy = geoj['working_pixel_to_native_scale_xy']
    if min(sx, sy) <= 0:
        raise ValueError('Invalid pinned Bora coordinate mapping')
    working = [transform(lambda x, y, z=None: (x/sx, y/sy),
                         shape(f['geometry'])) for f in geoj['features']]
    if not working or any(not g.is_valid for g in working):
        raise ValueError('Invalid pinned Bora support polygon')
    score_dir = prior.SCORES / slug
    score_receipt = score_dir / 'RIDGE_SCORE_GRID_VERIFIED.json'
    info = json.loads(score_receipt.read_text())
    score_path = score_dir / 'smoothed-scores.npy'
    if (info['status'] != 'VERIFIED_EIGHT_CLASS_RIDGE_CPU_SCORE_GRID' or
            info['model_receipt_sha256'] != sha(prior.MODEL) or
            info['ontology_sha256'] != sha(prior.ONTOLOGY) or
            sha(score_path) != info['smoothed_scores_sha256']):
        raise ValueError('Pinned eight-class supervised score grid differs')
    scores = np.load(score_path, mmap_mode='r', allow_pickle=False)
    if scores.shape[0] != 8 or scores.dtype != np.float32:
        raise ValueError('Expected eight float32 score channels')
    if list(scores.shape[1:]) != [calibration['padded_shape_yx'][0] // 14,
                                   calibration['padded_shape_yx'][1] // 14]:
        raise ValueError('Feature token grid and H&E calibration differ')
    source = {'slug': slug, 'path': str(source_path),
              'sha256': calibration['new_calibrated_sha256'],
              'height': calibration['padded_shape_yx'][0],
              'width': calibration['padded_shape_yx'][1]}
    presence = json.loads(prior.PRESENCE.read_text())
    return {'source': source, 'geoj': geoj, 'geo_path': geo_path,
            'working': working, 'scores': scores, 'score_path': score_path,
            'score_receipt': score_receipt,
            'threshold': float(presence['minimum_predicted_tissue_fraction']),
            'pinned_calibration_receipt_sha256': sha(calibration_path),
            'pinned_support_receipt_sha256': sha(support_receipts[0])}


def context_for(slug: str, slide: Path) -> dict:
    allowed = {p['slug'] for p in json.loads(prior.CAMPAIGN.read_text())['pages']}
    if slug not in allowed or 'pm016' in slug.lower():
        raise ValueError('Only thirteen public slides are admitted')
    original = {r['slug'] for r in json.loads(prior.SCOPE.read_text())['slides']}
    return prior.inputs(slug) if slug in original else pinned_inputs(slug, slide)


def changed_pixels(old: np.memmap, new: np.memmap) -> tuple[int, int]:
    changed = prior_positive = 0
    for y in range(0, old.shape[0], 512):
        difference = old[y:y+512] != new[y:y+512]
        changed += int(difference.sum())
        prior_positive += int(np.count_nonzero(np.asarray(old[y:y+512])[difference]))
    return changed, prior_positive


def main(slug: str) -> None:
    start = time.monotonic()
    slide = OUT / slug
    if slide.exists():
        raise FileExistsError('Preserve completed or failed versioned slide output')
    slide.mkdir(parents=True)
    state = slide / 'STATE.json'
    try:
        prior.atomic(state, {'status': 'RUNNING', 'stage': 'verify_inputs'})
        context = context_for(slug, slide)
        source = context['source']
        frame = (source['height'], source['width'])
        mpp = tuple(context['geoj']['working_mpp_xy'])
        support = prior.reviewed_support(context['working'], frame,
                                         slide / 'reviewed-support.uint8.dat')
        classes, selection = prior.select_classes(context['scores'], context['working'], context['threshold'])
        if len(classes) < 2:
            raise ValueError('Global vector requires at least two evidence-supported classes')
        prior.atomic(slide / 'CLASS_SELECTION.json', selection)
        prior.atomic(state, {'status': 'RUNNING', 'stage': 'full_slide_projection', 'classes': classes})
        raw = prior.project_mask(context['scores'], support, classes,
                                 slide / 'raw-mask.uint8.dat')
        pre, pre_report = prior.area_repair(raw, support,
                                            slide / 'precontour-400um2.uint8.dat', mpp, 400)
        prior.atomic(state, {'status': 'RUNNING', 'stage': 'physical_contour'})
        contour_path = slide / 'contour-6um.npy'
        contour_report = prior.smooth_global(pre, support, mpp, contour_path,
                                             sigma_um=6, block_size=1024)
        smoothed = np.load(contour_path, mmap_mode='r')
        final_source, final_report = prior.area_repair(
            smoothed, support, slide / 'prevector-2000um2.uint8.dat', mpp, 2000)
        del raw, pre, smoothed, final_source
        gc.collect()
        assembly = prior.bind_assembly(slug, slide / 'prevector-2000um2.uint8.dat',
                                       context, classes, slide)
        prior.atomic(state, {'status': 'RUNNING', 'stage': 'global_vector'})
        names = {int(k): value['name'].replace('_', ' ')
                 for k, value in json.loads(prior.ONTOLOGY.read_text())['classes'].items()}
        vector = prior.publish_global_wsi_vectors(
            assembly, slide / 'global-vectors', names, minimum_area_um2=2000,
            smoothing_tolerance_um=0, maximum_cleanup_change=.02,
            max_input_regions=250_000, max_coordinates=10_000_000,
            max_boundary_edges=10_000_000, disk_budget_bytes=12 * 1024**3)
        exact = slide / 'global-vectors/exact/mask.uint8.dat'
        old = np.memmap(exact, mode='r', dtype=np.uint8, shape=frame)
        candidate = slide / 'postvector-candidate.uint8.dat'
        cleaned, post_report = prior.area_repair(old, support, candidate, mpp, 2000)
        changed, old_positive = changed_pixels(old, cleaned)
        if old_positive:
            raise ValueError('Global vector has subthreshold positive raster; different recovery required')
        if changed > .001 * int(np.count_nonzero(old)):
            raise ValueError('Vector gap repair would change too much tissue')
        working_path = slide / 'supervised-bora.working.geojson'
        final_mask = slide / 'supervised-bora.final-2000um2.uint8.dat'
        if changed:
            prior.atomic(state, {'status': 'RUNNING', 'stage': 'unoccupied_vector_gap_repair',
                                 'gap_pixels': changed})
            patch = gap_repair.patched_working_geojson(
                slide / 'global-vectors/exact/mask.geojson', old, cleaned,
                working_path, mpp)
            gap_repair.audit_working_raster(
                working_path, cleaned, slide / 'working-roundtrip.partial.dat')
            os.replace(candidate, final_mask)
        else:
            candidate.unlink()
            os.link(exact, final_mask)
            os.link(slide / 'global-vectors/exact/mask.geojson', working_path)
            patch = {'patches': 0, 'source': 'exact global vector unchanged'}
        del old, cleaned, support
        gc.collect()
        native = slide / 'supervised-bora.native.geojson'
        native_report = prior.native_geojson(working_path, context, native)
        tiff = slide / 'supervised-bora.mask-0p5um.tif'
        prior.write_mask_tiff(final_mask, frame, tiff)
        prior.atomic(state, {'status': 'RUNNING', 'stage': 'independent_final_audit'})
        final_audit = prior.audit_final(final_mask, slide / 'reviewed-support.uint8.dat', native, context)
        result = {'status': 'SUPERVISED_BORA_RIDGE_FULL_SLIDE_VECTOR_VERIFIED',
                  'slug': slug, 'source_image_sha256': source['sha256'],
                  'source_bora_geojson_sha256': sha(context['geo_path']),
                  'score_sha256': sha(context['score_path']),
                  'training_model_receipt_sha256': sha(prior.MODEL),
                  'training_presence_sha256': sha(prior.PRESENCE),
                  'ontology_sha256': sha(prior.ONTOLOGY), 'class_ids': classes,
                  'class_selection': selection, 'precontour_cleanup': pre_report,
                  'contour_smoothing': contour_report, 'prevector_cleanup': final_report,
                  'postvector_cleanup': post_report, 'postvector_gap_pixels': changed,
                  'vector_patch': patch, 'native_vector': native_report,
                  'independent_geometry_audit': final_audit,
                  'global_vector_receipt_sha256': sha(slide / 'global-vectors/GLOBAL_VECTOR_COMPLETE.json'),
                  'calibrated_shape_yx': list(frame), 'calibrated_mpp_xy': list(mpp),
                  'native_mpp_xy': context['geoj']['native_mpp_xy'],
                  'output_sha256': {'mask_tiff': sha(tiff), 'native_geojson': sha(native),
                                    'exact_mask': sha(final_mask), 'working_geojson': sha(working_path)},
                  'source_code_sha256': sha(Path(__file__)),
                  'dependencies_sha256': {'v30_pipeline': sha(V30), 'v32_gap_repair': sha(V32)},
                  'gpu_used': False, 'whole_slide_expert_accuracy': None,
                  'pm016_included': False, 'elapsed_seconds': time.monotonic()-start}
        prior.atomic(slide / 'FULL_SLIDE_VERIFIED.json', result)
        prior.atomic(state, {'status': 'COMPLETE',
                             'receipt_sha256': sha(slide / 'FULL_SLIDE_VERIFIED.json')})
        print('SUPERVISED_FULL_SLIDE_VERIFIED', slug, classes,
              'gap_pixels', changed, round(result['elapsed_seconds'], 1), flush=True)
    except BaseException as exc:
        prior.atomic(state, {'status': 'FAILED_PRESERVE_PARTIAL', 'error': repr(exc),
                             'elapsed_seconds': time.monotonic()-start})
        raise


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Usage: finalize_public_wsi.py PUBLIC_SLUG')
    main(sys.argv[1])
