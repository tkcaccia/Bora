"""Joint-image KODAMA staging for the developing R Bora interface.

Every admitted tile contributes one full-channel Virchow2 mean. Rows retain
their source-image and tile-grid identities. KODAMA never sees spatial
coordinates or expert annotations. A strict 90% tissue-pixel rule admits a
tile when an explicit tissue support was supplied; padded pixels count as
background. There is no silent tile subsample.
"""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import resource

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .r_feature_preflight import TILE, sha256
from .r_training import CHANNELS, _verified_scales
from .r_api import _atomic_json


def _admitted_tile_rows(source: dict) -> np.ndarray:
    """Return row-major admitted 224-pixel tile indices from frozen counts."""
    counts_path = Path(source['tissue_pixel_counts_path'])
    eligible_path = Path(source['eligible_path'])
    if (sha256(counts_path) != source['tissue_pixel_counts_sha256'] or
            sha256(eligible_path) != source['eligible_sha256']):
        raise ValueError('Prepared tile counts or eligibility hash differs')
    counts = np.load(counts_path, mmap_mode='r', allow_pickle=False)
    eligible = np.load(eligible_path, mmap_mode='r', allow_pickle=False)
    grid_h, grid_w = map(int, source['tile_grid_shape_yx'])
    if (counts.dtype != np.uint32 or counts.shape != (grid_h, grid_w) or
            eligible.dtype != np.bool_ or
            list(eligible.shape) != source['base_grid_shape_yx'] or
            np.any(counts > TILE * TILE) or
            np.any((counts > 0) != eligible[:grid_h, :grid_w])):
        raise ValueError('Prepared tile counts or geometry differ')
    image_h, image_w = map(int, source['image_shape_yx'])
    rows = []
    for ty in range(grid_h):
        valid_h = min(TILE, image_h - ty * TILE)
        for tx in range(grid_w):
            valid_w = min(TILE, image_w - tx * TILE)
            if valid_h < 1 or valid_w < 1:
                raise ValueError('Tile grid exceeds the source image')
            if int(counts[ty, tx]) > valid_h * valid_w:
                raise ValueError('Prepared tile counts or geometry differ')
            if source['tissue_support'] is None:
                # Without a mask, the source image includes glass; only
                # artificial right/bottom padding is excluded from admission.
                tissue = valid_h * valid_w
                if int(counts[ty, tx]) != tissue:
                    raise ValueError('Full-glass tile counts differ')
            else:
                tissue = int(counts[ty, tx])
            if tissue * 10 > 9 * TILE * TILE:
                rows.append((ty, tx))
    return np.asarray(rows, dtype=np.int32).reshape(-1, 2)


def stage_joint(preparation_dir: Path, output_dir: Path, *,
                minimum_tiles: int = 40, max_joint_tiles: int = 10000,
                minimum_free_gib: float = 1.0) -> dict:
    """Freeze all admitted source tiles and their joint PCA input for R KODAMA."""
    root = Path(preparation_dir).resolve()
    manifest_path = root / 'PREPARATION_COMPLETE.json'
    preparation = json.loads(manifest_path.read_text())
    if preparation.get('format') != 'bora-r-preparation-v1' or \
            preparation.get('status') != 'COMPLETE' or not preparation.get('cases'):
        raise ValueError('Complete Bora preparation required')
    if (not isinstance(minimum_tiles, int) or minimum_tiles < 40 or
            not isinstance(max_joint_tiles, int) or max_joint_tiles < minimum_tiles or
            not math.isfinite(minimum_free_gib) or minimum_free_gib < 0):
        raise ValueError('Finite joint clustering admission limits required')
    output = Path(output_dir).resolve()
    if output.exists() or not output.parent.is_dir():
        raise FileExistsError('Use a fresh joint KODAMA output directory')

    admitted = []
    for index, case in enumerate(preparation['cases']):
        source = case['source']
        if sha256(Path(source['image'])) != source['image_sha256']:
            raise ValueError('Prepared H&E source SHA256 differs')
        if source['tissue_support'] is not None and \
                sha256(Path(source['tissue_support'])) != source['tissue_support_sha256']:
            raise ValueError('Prepared tissue support SHA256 differs')
        tile_yx = _admitted_tile_rows(source)
        admitted.append((index, case, tile_yx))
    ntiles = sum(len(rows) for _, _, rows in admitted)
    if ntiles < minimum_tiles:
        raise ValueError(f'Only {ntiles} strict eligible tiles, below {minimum_tiles}')
    if ntiles > max_joint_tiles:
        raise MemoryError(f'{ntiles} joint tiles exceed declared {max_joint_tiles} cap')
    projected_bytes = ntiles * (CHANNELS * 4 * 4 + 30 * 16 + 256) + (1 << 30)
    stat = os.statvfs(output.parent)
    if stat.f_bavail * stat.f_frsize - projected_bytes < minimum_free_gib * 1024**3:
        raise OSError('Joint KODAMA staging would breach disk reserve')
    with Path('/proc/meminfo').open() as stream:
        available_kib = next(int(line.split()[1]) for line in stream
                             if line.startswith('MemAvailable:'))
    if ntiles * CHANNELS * 4 * 5 > available_kib * 1024 * .7:
        raise MemoryError('Joint tile PCA would breach host-memory admission')

    output.mkdir()
    _atomic_json(output / 'RESOURCE_ADMISSION.json', {
        'format': 'bora-r-joint-kodama-admission-v1',
        'tiles': ntiles, 'images': len(admitted),
        'max_joint_tiles': max_joint_tiles,
        'projected_peak_new_bytes': projected_bytes,
        'minimum_free_gib': minimum_free_gib,
        'available_memory_kib_before': available_kib,
    })
    try:
        means = np.lib.format.open_memmap(output / 'tile-means-1280.npy', mode='w+',
                                           dtype=np.float32, shape=(ntiles, CHANNELS))
        identity = np.lib.format.open_memmap(output / 'tile-identity.npy', mode='w+',
                                              dtype=np.int32, shape=(ntiles, 3))
        case_rows = []
        offset = 0
        for index, case, tile_yx in admitted:
            grids = _verified_scales(preparation, case)
            grid = grids[1]
            source = case['source']
            needed_h = source['base_grid_shape_yx'][0] * 16
            needed_w = source['base_grid_shape_yx'][1] * 16
            if grid.shape[0] < needed_h or grid.shape[1] < needed_w:
                raise ValueError('Factor-1 token grid cannot register source tiles')
            for ty, tx in tile_yx:
                tile = np.asarray(grid[ty * 16:(ty + 1) * 16,
                                       tx * 16:(tx + 1) * 16], np.float32)
                if tile.shape != (16, 16, CHANNELS) or not np.isfinite(tile).all():
                    raise ValueError('Admitted tile has nonfinite Virchow2 features')
                means[offset] = tile.mean(axis=(0, 1), dtype=np.float64)
                identity[offset] = (index, ty, tx)
                offset += 1
            case_rows.append({'case_index': index,
                              'image': source['image'],
                              'image_sha256': source['image_sha256'],
                              'tissue_support_sha256': source['tissue_support_sha256'],
                              'row_start': offset - len(tile_yx), 'row_end': offset,
                              'admitted_tiles': len(tile_yx),
                              'tile_grid_shape_yx': source['tile_grid_shape_yx']})
            means.flush()
            identity.flush()
        if offset != ntiles:
            raise AssertionError('Joint row accounting differs')
        with threadpool_limits(limits=2):
            scaler = StandardScaler().fit(means)
            scaler.scale_ = np.maximum(scaler.scale_, .05)
            z = scaler.transform(means)
            pca = PCA(n_components=30, svd_solver='randomized',
                      random_state=20261114)
            coordinates = pca.fit_transform(z).astype(np.float32)
        if coordinates.shape != (ntiles, 30) or not np.isfinite(coordinates).all():
            raise ValueError('Joint PCA30 coordinates are invalid')
        np.save(output / 'joint-pca30.npy', coordinates, allow_pickle=False)
        with (output / 'joint-pca30.csv').open('x', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow([f'PC{i}' for i in range(1, 31)])
            writer.writerows(coordinates)
        np.savez_compressed(output / 'pca-transform.npz',
                            scaler_mean=scaler.mean_.astype(np.float32),
                            scaler_scale=scaler.scale_.astype(np.float32),
                            components=pca.components_.astype(np.float32),
                            pca_mean=pca.mean_.astype(np.float32))
        del means, identity
        record = {'format': 'bora-r-joint-kodama-stage-v1',
                  'status': 'COMPLETE', 'preparation_manifest_sha256': sha256(manifest_path),
                  'image_count': len(admitted), 'tile_count': ntiles,
                  'strict_tissue_rule': 'tissue_pixels*10 > 9*224*224; padding is background',
                  'source_coordinates_used_in_kodama': False,
                  'expert_annotations_opened': False,
                  'cases': case_rows,
                  'files': {name: sha256(output / name) for name in (
                      'tile-means-1280.npy', 'tile-identity.npy', 'joint-pca30.npy',
                      'joint-pca30.csv', 'pca-transform.npz')},
                  'peak_host_rss_kb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        _atomic_json(output / 'JOINT_STAGE_COMPLETE.json', record)
        return record
    except BaseException as error:
        _atomic_json(output / 'JOINT_STAGE_FAILED.json', {
            'format': 'bora-r-joint-kodama-stage-failure-v1',
            'status': 'FAILED', 'error': repr(error), 'completed_rows': offset,
        })
        raise


def _cluster_labels(path: Path, ntiles: int) -> np.ndarray:
    """Read R's zero-based row identities and positive group IDs exactly."""
    with Path(path).open(newline='') as stream:
        table = list(csv.DictReader(stream))
    if len(table) != ntiles or set(table[0]) != {'row_index', 'group'}:
        raise ValueError('KODAMA group table has changed shape or columns')
    try:
        indices = np.asarray([int(item['row_index']) for item in table], np.int64)
        groups = np.asarray([int(item['group']) for item in table], np.int32)
    except (TypeError, ValueError) as error:
        raise ValueError('KODAMA group table contains noninteger labels') from error
    if not np.array_equal(indices, np.arange(ntiles)) or \
            groups.min() < 1 or groups.max() > 254 or \
            not np.array_equal(np.unique(groups), np.arange(1, groups.max() + 1)):
        raise ValueError('KODAMA rows or group IDs differ from the joint stage')
    return groups


def _group_head(means: np.ndarray, labels: np.ndarray,
                selected_channels: int = 64) -> dict:
    """Fit a deterministic full-channel tile-group head for token projection.

    Nearest standardized centroid has a linear argmax decision. For two groups
    the coefficients are folded into the historical binary score convention.
    This head names anonymous groups only; it is not a tissue classifier.
    """
    ntiles, channels = means.shape
    if channels != CHANNELS or len(labels) != ntiles or \
            not 1 <= selected_channels <= CHANNELS or \
            not np.isfinite(means).all():
        raise ValueError('Full-channel finite tile means are required')
    classes = np.unique(labels).astype(np.uint8)
    if not np.array_equal(classes, np.arange(1, len(classes) + 1)):
        raise ValueError('Anonymous group IDs must be contiguous')
    mean = means.mean(0, dtype=np.float64).astype(np.float32)
    std = np.maximum(means.std(0, dtype=np.float64).astype(np.float32), .05)
    z = (means - mean) / std
    if len(classes) == 1:
        selected = np.arange(selected_channels, dtype=np.int32)
    else:
        group_means = np.stack([z[labels == cls].mean(0) for cls in classes])
        separation = group_means.var(0)
        selected = np.sort(np.argsort(-separation, kind='stable')[:selected_channels]).astype(np.int32)
    centroids = np.stack([z[labels == cls][:, selected].mean(0) for cls in classes])
    if len(classes) == 1:
        coef = np.zeros((1, len(selected)), np.float32)
        intercept = np.zeros(1, np.float32)
    elif len(classes) == 2:
        coef = (centroids[1] - centroids[0])[None, :].astype(np.float32)
        intercept = np.asarray([-.5 * (np.sum(centroids[1] ** 2) -
                                           np.sum(centroids[0] ** 2))], np.float32)
    else:
        coef = (2 * centroids).astype(np.float32)
        intercept = -(centroids ** 2).sum(1).astype(np.float32)
    return {'selected': selected, 'mean': mean[selected], 'std': std[selected],
            'coef': coef, 'intercept': intercept, 'classes': classes,
            'centroids': centroids.astype(np.float32),
            'training_tile_counts': np.bincount(labels)[1:].astype(np.int64)}


def complete_joint(preparation_dir: Path, stage_dir: Path, cluster_dir: Path,
                   output_dir: Path, *, source_mpp_xy: tuple[float, float],
                   output_mpp_xy: tuple[float, float],
                   minimum_area_um2: float = 2000.0,
                   contour_sigma_um: float = 6.0,
                   minimum_free_gib: float = 1.0,
                   max_new_gib: float | None = None) -> dict:
    """Project each joint solution back to every source image and write masks."""
    from .r_prediction import (
        _preview, _resource_admission, _write_mask_tiff,
        finalize_projected_mask, output_shape, output_support,
        project_mask, score_case, source_coordinate_geojson,
    )

    prep_path = Path(preparation_dir).resolve() / 'PREPARATION_COMPLETE.json'
    stage_path = Path(stage_dir).resolve() / 'JOINT_STAGE_COMPLETE.json'
    cluster_path = Path(cluster_dir).resolve() / 'CLUSTERS_COMPLETE.json'
    prep = json.loads(prep_path.read_text())
    stage = json.loads(stage_path.read_text())
    clusters = json.loads(cluster_path.read_text())
    if (prep.get('status') != 'COMPLETE' or stage.get('status') != 'COMPLETE' or
            clusters.get('status') != 'COMPLETE' or
            stage.get('format') != 'bora-r-joint-kodama-stage-v1' or
            clusters.get('format') != 'bora-r-joint-kodama-clusters-v1' or
            stage['preparation_manifest_sha256'] != sha256(prep_path) or
            clusters['stage_sha256'] != sha256(stage_path) or
            clusters['KODAMA_version'] != '0.99.7' or
            clusters['tile_count'] != stage['tile_count'] or
            clusters['image_count'] != stage['image_count'] or
            len(prep['cases']) != stage['image_count'] or
            not clusters['solutions']):
        raise ValueError('Frozen joint preparation, stage or KODAMA clusters differ')
    stage_root = stage_path.parent
    for name, expected in stage['files'].items():
        if sha256(stage_root / name) != expected:
            raise ValueError(f'Joint stage file changed: {name}')
    means = np.load(stage_root / 'tile-means-1280.npy', mmap_mode='r', allow_pickle=False)
    identity = np.load(stage_root / 'tile-identity.npy', mmap_mode='r', allow_pickle=False)
    if (means.shape != (stage['tile_count'], CHANNELS) or means.dtype != np.float32 or
            identity.shape != (stage['tile_count'], 3) or identity.dtype != np.int32):
        raise ValueError('Joint tile feature and identity arrays differ')
    for item in stage['cases']:
        start, end, index = item['row_start'], item['row_end'], item['case_index']
        source = prep['cases'][index]['source']
        if (end - start != item['admitted_tiles'] or
                source['image_sha256'] != item['image_sha256'] or
                not np.all(identity[start:end, 0] == index)):
            raise ValueError('Joint rows no longer map to source images')
    output = Path(output_dir).resolve()
    if output.exists() or not output.parent.is_dir():
        raise FileExistsError('Use a fresh output directory for final masks')
    plans = []
    for item in clusters['solutions']:
        path = Path(item['group_csv']).resolve()
        if path.parent != Path(cluster_dir).resolve() / f"solution-{int(item['index']):03d}":
            raise ValueError('KODAMA solution path escapes cluster directory')
        labels = _cluster_labels(path, stage['tile_count'])
        if len(np.unique(labels)) != item['groups'] or \
                np.bincount(labels)[1:].tolist() != item['group_counts']:
            raise ValueError('KODAMA group counts changed')
        if item['groups'] < 2:
            raise ValueError('Requested KODAMA resolution yielded one group; '
                             'two or more groups are required for tissue boundaries')
        head = _group_head(means, labels)
        admission = _resource_admission(prep, head, source_mpp_xy, output_mpp_xy,
                                        output, minimum_free_gib, None)
        plans.append((item, path, labels, head, admission))
    projected = sum(plan[4]['projected_peak_new_bytes'] for plan in plans)
    free = os.statvfs(output.parent).f_bavail * os.statvfs(output.parent).f_frsize
    if free - projected < minimum_free_gib * 1024**3 or \
            (max_new_gib is not None and projected > max_new_gib * 1024**3):
        raise OSError('Joint multi-solution outputs breach storage admission')
    output.mkdir()
    _atomic_json(output / 'RESOURCE_ADMISSION.json', {
        'format': 'bora-r-joint-output-admission-v1',
        'projected_peak_new_bytes': projected,
        'minimum_free_gib': minimum_free_gib, 'max_new_gib': max_new_gib,
        'solutions': len(plans), 'source_images': len(prep['cases'])})
    preview_cache = output / 'source-preview-rgb'
    preview_cache.mkdir()
    complete = []
    try:
        for item, path, labels, head, admission in plans:
            solution_dir = output / f"solution-{int(item['index']):03d}"
            solution_dir.mkdir()
            model_path = solution_dir / 'anonymous-group-head.npz'
            np.savez_compressed(model_path, **head)
            model_hash = sha256(model_path)
            cases = []
            for case_index, case in enumerate(prep['cases']):
                source = case['source']
                image_path = Path(source['image'])
                if sha256(image_path) != source['image_sha256']:
                    raise ValueError('Prepared H&E source changed before output')
                folder = solution_dir / f'image-{case_index:03d}'
                folder.mkdir()
                score = score_case(prep, case, head, folder / 'scores')
                shape = output_shape(tuple(source['image_shape_yx']),
                                     source_mpp_xy, output_mpp_xy)
                ratio = (output_mpp_xy[0] / source_mpp_xy[0],
                         output_mpp_xy[1] / source_mpp_xy[1])
                support = output_support(source, shape, ratio,
                                         folder / 'output-support.uint8.dat')
                support_array = np.memmap(support['support_path'], mode='r',
                                          dtype=np.uint8, shape=shape)
                smooth = np.load(score['smoothed_scores_path'], mmap_mode='r')
                native = project_mask(smooth, head['classes'], shape, ratio,
                                      folder / 'native-mask.uint8.dat')
                shared = project_mask(smooth, head['classes'], shape, ratio,
                                      folder / 'support-clipped-mask.uint8.dat',
                                      support=support_array)
                final = finalize_projected_mask(
                    Path(shared['mask_path']), Path(support['support_path']), shape,
                    output_mpp_xy, list(map(int, head['classes'])),
                    source['image_sha256'], model_hash, folder / 'cleanup',
                    minimum_area_um2=minimum_area_um2,
                    contour_sigma_um=contour_sigma_um,
                    class_names={str(int(cls)): f"Group {int(cls)}"
                                 for cls in head['classes']})
                vector = source_coordinate_geojson(
                    folder / 'cleanup/vectors-output-grid/exact/mask.geojson',
                    folder / 'class-mask.geojson',
                    tuple(source['image_shape_yx']), source_mpp_xy, output_mpp_xy)
                final_array = np.memmap(final['final_mask_path'], mode='r',
                                        dtype=np.uint8, shape=shape)
                masks = {'native': _write_mask_tiff(
                    np.memmap(native['mask_path'], mode='r', dtype=np.uint8,
                              shape=shape), folder / 'native-mask.tif'),
                         'final': _write_mask_tiff(final_array,
                                                   folder / 'class-mask.tif')}
                binary = {str(int(cls)): _write_mask_tiff(
                    final_array, folder / f'bw-group-{int(cls):03d}.tif',
                    equals=int(cls)) for cls in head['classes']}
                preview = _preview(image_path, Path(masks['final']['path']),
                                   list(map(int, head['classes'])),
                                   folder / 'preview.png',
                                   source_rgb_cache=preview_cache /
                                   f'image-{case_index:03d}.npy')
                record = {'format': 'bora-r-anonymous-prediction-case-v1',
                          'status': 'COMPLETE', 'image': str(image_path),
                          'image_sha256': source['image_sha256'],
                          'source_case_index': case_index,
                          'group_csv_sha256': sha256(path),
                          'anonymous_group_head_sha256': model_hash,
                          'source_mpp_xy': list(source_mpp_xy),
                          'output_mpp_xy': list(output_mpp_xy),
                          'classes_are_anonymous_group_ids': True,
                          'scores': score, 'support': support,
                          'native_projection': native,
                          'support_clipped_projection': shared,
                          'finalization_sha256': sha256(
                              folder / 'cleanup/FINALIZATION_COMPLETE.json'),
                          'source_vector': vector,
                          'masks': masks, 'binary_group_images': binary,
                          'preview': preview}
                case_receipt = folder / 'CASE_COMPLETE.json'
                _atomic_json(case_receipt, record)
                cases.append({'receipt_path': str(case_receipt),
                              'receipt_sha256': sha256(case_receipt)})
            solution_record = {
                'format': 'bora-r-anonymous-solution-v1', 'status': 'COMPLETE',
                'index': item['index'], 'resolution': item['resolution'],
                'groups': item['groups'], 'group_csv_sha256': sha256(path),
                'head_sha256': model_hash,
                'preparation_manifest_sha256': sha256(prep_path),
                'joint_stage_sha256': sha256(stage_path),
                'clusters_sha256': sha256(cluster_path),
                'source_images': len(cases), 'cases': cases,
                'expert_accuracy_qualified': False,
            }
            solution_receipt = solution_dir / 'SOLUTION_COMPLETE.json'
            _atomic_json(solution_receipt, solution_record)
            complete.append({'receipt_path': str(solution_receipt),
                             'receipt_sha256': sha256(solution_receipt)})
        result = {'format': 'bora-r-prediction-v1', 'status': 'COMPLETE',
                  'route': 'joint-image-kodama-anonymous-groups',
                  'source_preview_caches': [
                      {'path': str(preview_cache / f'image-{i:03d}.npy'),
                       'sha256': sha256(preview_cache / f'image-{i:03d}.npy'),
                       'source_image_sha256': case['source']['image_sha256']}
                      for i, case in enumerate(prep['cases'])],
                  'preparation_manifest_sha256': sha256(prep_path),
                  'joint_stage_sha256': sha256(stage_path),
                  'clusters_sha256': sha256(cluster_path),
                  'image_count': len(prep['cases']), 'solution_count': len(complete),
                  'solutions': complete,
                  'peak_host_rss_kb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  'gpu_used': False, 'expert_accuracy_qualified': False}
        _atomic_json(output / 'PREDICTION_COMPLETE.json', result)
        return result
    except BaseException as error:
        _atomic_json(output / 'PREDICTION_FAILED.json', {
            'format': 'bora-r-anonymous-prediction-failure-v1',
            'status': 'FAILED', 'error': repr(error),
            'completed_solutions': complete})
        raise
