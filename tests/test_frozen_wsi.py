"""Regression contracts for the selected complete-slide colorectal route."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from shapely.geometry import shape, mapping, Polygon

from bora import selected_wsi as kernels
from bora.wsi_frozen import read_case, run_wsi, crop_wsi
from bora.slide_responses import fit_slide_responses, project_slide_responses
from bora.compact_geojson import fit_coverage
from bora.wsi_coverage_lookup import fit_coverage_indexed


def write_json(path, data):
    path.write_text(json.dumps(data))


def case_files(folder, height=481, width=601):
    """Odd frame and nonzero native origin expose alignment/crop mistakes."""
    yy, xx = np.indices((height, width))
    image = np.empty((height, width, 3), np.uint8)
    boundary = width / 2 + 35 * np.sin(yy / 45)
    left = xx < boundary
    image[left] = (124, 65, 140)
    image[~left] = (214, 132, 157)
    Image.fromarray(image).save(folder / 'he.png')
    rows,cols = ((height+895)//896)*4,((width+895)//896)*4
    grid = np.ones((rows,cols), np.uint8); grid[:, max(1,(width+447)//448):] = 2
    np.save(folder / 'groups.npy', grid)
    ty, tx = np.indices((rows*16,cols*16))
    p = 1 / (1 + np.exp(np.clip((tx*14+7-width/2-35*np.sin((ty*14+7)/45))/12, -80, 80)))
    np.save(folder / 'prob.npy', np.stack([p, 1-p], -1).astype(np.float16))
    write_json(folder / 'response.json', dict(format='bora-slide-global-responses-v1', status='COMPLETE',
        source_frame_id='odd-frame', source_image_sha256=kernels.sha(folder/'he.png'),
        group_grid_sha256=kernels.sha(folder/'groups.npy'), probability_sha256=kernels.sha(folder/'prob.npy'),
        shape_yxc=[rows*16,cols*16,2], class_ids=[1,2], one_fit_used_across_slide=True,
        independently_fitted_windows=False, model_sha256='a'*64))
    spec = dict(format='bora-wsi-colorectal-v1', source_frame_id='odd-frame', image='he.png',
        group_grid='groups.npy', probabilities='prob.npy', response_receipt='response.json',
        shape_yx=[height,width], working_mpp_xy=[.5,.5], native_mpp_xy=[.25,.25],
        native_shape_yx=[height*2+60,width*2+40],
        analysis_native_bounds_xyxy=[20,30,20+width*2,30+height*2])
    write_json(folder/'case.json', spec)
    return spec


def test_profile_preserves_reference_kernel_hashes():
    import bora
    root = Path(bora.__file__).parent
    profile = json.loads((root/'frozen_colorectal.json').read_text())
    assert profile['version'] == bora.__version__ == '0.2.2'
    assert profile['parameters']['final_minimum_area_um2'] == 400
    assert profile['parameters']['group_count_reference'] == 16
    for name, digest in profile['source_sha256'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == digest, name


def test_no_annotation_or_independent_field_fit_is_accepted(tmp_path):
    spec = case_files(tmp_path)
    read_case(tmp_path/'case.json')
    spec['annotation'] = 'expert.tif'
    write_json(tmp_path/'case.json', spec)
    with pytest.raises(ValueError, match='predictors only'):
        read_case(tmp_path/'case.json')
    spec.pop('annotation'); write_json(tmp_path/'case.json', spec)
    receipt = json.loads((tmp_path/'response.json').read_text())
    receipt['independently_fitted_windows'] = True
    write_json(tmp_path/'response.json', receipt)
    with pytest.raises(ValueError, match='one frozen slide-wide'):
        read_case(tmp_path/'case.json')


def test_global_cleanup_strict_threshold_and_component_crossing_crop_edge(tmp_path):
    labels = np.ones((150,180), np.uint8)
    labels[20:60,20:60] = 2  # exactly 400 um2 at 0.5 um/px
    labels[80:119,20:60] = 2  # strictly below
    labels[20:60,100:140] = 0  # exactly 400 um2 enclosed hole
    labels[80:119,100:140] = 0
    labels[0:3,0:3] = 0  # slide-edge background is not an enclosed hole
    output, report = kernels.clean_global(labels, (.5,.5), tmp_path/'clean.npy')
    assert np.all(output[20:60,20:60] == 2)
    assert np.all(output[80:119,20:60] == 1)
    assert np.all(output[20:60,100:140] == 0)
    assert np.all(output[80:119,100:140] == 1)
    assert np.all(output[0:3,0:3] == 0)
    assert report['parts'] == report['holes'] == 1
    # A small diagnostic fragment of the retained component is preserved.
    assert np.count_nonzero(output[20:60,55:65] == 2) == 200


def test_streamed_diffusion_and_smoothing_equal_one_global_solution(tmp_path):
    pytest.importorskip('pyvips')
    import cv2
    spec = case_files(tmp_path, 1701, 1803)
    image, prob, grid, identity = read_case(tmp_path/'case.json')
    support = np.ones((1701,1803), np.uint8)
    # Globally aligned source support crossing computation cores.
    support[700:720,700:900] = 0
    out = tmp_path/'stream'; out.mkdir()
    streamed = kernels.predict(image, prob, grid, support, out, (.5,.5))
    h,w = support.shape; sh,sw = (h+1)//2,(w+1)//2
    expected = kernels.diffuse_window(image, np.asarray(prob,np.float32), grid, support, (0,0,sw,sh))
    np.testing.assert_array_equal(np.load(out/'diffusion-1um.npy'), expected)
    labels = expected[np.arange(h)[:,None]//2, np.arange(w)[None,:]//2]
    labels[support==0] = 0
    best = np.full(labels.shape,-1.,np.float32); smoothed = np.zeros_like(labels)
    for group in np.unique(labels):
        response = cv2.GaussianBlur((labels==group).astype(np.float32),(0,0),sigmaX=3,sigmaY=3,borderType=cv2.BORDER_REFLECT_101)
        better = response > best; smoothed[better] = group; best[better] = response[better]
    smoothed[support==0] = 0
    np.testing.assert_array_equal(streamed, smoothed)
    checks = json.loads((out/'SEAM_AUDIT.json').read_text())['checks']
    assert checks and all(c['different_pixels']==0 for c in checks)


def test_selected_pipeline_native_contours_and_crops_use_global_result(tmp_path, monkeypatch):
    pytest.importorskip('pyvips')
    import rasterio
    spec = case_files(tmp_path)
    result = tmp_path/'result'
    receipt = run_wsi(tmp_path/'case.json', result, minimum_free_bytes=0, memory_budget_bytes=1024**3)
    assert receipt['status'] == 'COMPLETE'
    assert receipt['entire_slide_refinement'] and not receipt['diagnostic_fields_used_in_inference']
    assert all(c['below_minimum']==0 for c in receipt['global_geometry_audit']['global_component_audits'])
    native = json.loads((result/'BORA_FULL_SLIDE.wsi.geojson').read_text())
    assert native['native_origin_xy'] == [20,30]
    assert native['native_shape_yx'] == spec['native_shape_yx']
    assert native['compact_boundary_report'] if 'compact_boundary_report' in native else native['source_working_grid_vector_processing']
    for feature in native['features']:
        g=shape(feature['geometry'])
        assert g.is_valid
        assert g.bounds[0]>=20-1e-5 and g.bounds[1]>=30-1e-5
        assert feature['properties']['area_um2']==pytest.approx(g.area*.25*.25)
    def forbidden(*args, **kwargs):
        raise AssertionError('Diagnostic crops must not run refinement or cleanup')
    monkeypatch.setattr(kernels,'predict',forbidden); monkeypatch.setattr(kernels,'clean_global',forbidden)
    crop_wsi(result,[250,100,70,200],tmp_path/'crop')
    with rasterio.open(result/'BORA_FULL_SLIDE.mask-0p5um.tif') as source:
        full=source.read(1)
    np.testing.assert_array_equal(np.array(Image.open(tmp_path/'crop/output-mask.png')),full[100:300,250:320])
    clipped=json.loads((tmp_path/'crop/crop.wsi.geojson').read_text())
    assert clipped['crop_only']
    assert all(shape(f['geometry']).is_valid for f in clipped['features'])
    with pytest.raises(FileExistsError): run_wsi(tmp_path/'case.json',result)
    with pytest.raises(ValueError, match='Memory budget'):run_wsi(tmp_path/'case.json',tmp_path/'low-memory',memory_budget_bytes=0)
    # A completed mask changed behind its receipt must never be displayed as verified.
    with (result/'BORA_FULL_SLIDE.wsi.geojson').open('a') as f:f.write(' ')
    with pytest.raises(ValueError,match='bytes changed'):crop_wsi(result,[0,0,10,10],tmp_path/'changed-crop')


def test_indexed_shared_boundaries_match_original_converter():
    first=Polygon([(0,0),(80,0),(80,60),(70,70),(80,80),(80,120),(0,120)])
    second=Polygon([(80,0),(160,0),(160,120),(80,120),(80,80),(70,70),(80,60)])
    data=dict(type='FeatureCollection',features=[dict(type='Feature',properties=dict(value=i+1),geometry=mapping(g)) for i,g in enumerate((first,second))])
    original=fit_coverage(data,[.5,.5],2)
    indexed=fit_coverage_indexed(data,[.5,.5],2)
    for a,b in zip(original['features'],indexed['features']):
        assert shape(a['geometry']).symmetric_difference(shape(b['geometry'])).area==0


def test_one_slide_response_fit_is_reused_for_all_pixels(tmp_path):
    pytest.importorskip('pyvips')
    import pyvips
    rng=np.random.default_rng(430)
    grid=(np.indices((4,4)).sum(0)%3+1).astype(np.uint8);grid[0,0]=0
    np.save(tmp_path/'groups.npy',grid)
    Image.new('RGB',(890,889),(170,90,140)).save(tmp_path/'he.png')
    spec=dict(format='bora-wsi-response-input-v1',source_frame_id='fit-fixture',image='he.png',shape_yx=[889,890],
        group_grid='groups.npy',selected_dimensions=list(range(64)),features={})
    for factor in (1,2,4):
        step=16//factor;ids=grid.repeat(step,0).repeat(step,1)
        values=rng.normal(0,.1,(*ids.shape,64)).astype(np.float32)+ids[...,None]*.8
        values[ids==0]=np.nan
        name=f'factor{factor}.npy';np.save(tmp_path/name,values);spec['features'][str(factor)]=name
    write_json(tmp_path/'input.json',spec)
    # Exercise the real CLI in fresh processes, starting from the two-thread
    # environment that exposed a native optimizer crash on Linux CI.
    env=dict(os.environ,OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
    entry=[sys.executable,'-c','from bora.entrypoint import main; raise SystemExit(main())']
    subprocess.run(entry+['wsi-fit',str(tmp_path/'input.json'),'--output',str(tmp_path/'fit')],env=env,check=True,capture_output=True,text=True)
    subprocess.run(entry+['wsi-project',str(tmp_path/'input.json'),'--fit',str(tmp_path/'fit'),'--output',str(tmp_path/'project')],env=env,check=True,capture_output=True,text=True)
    fitted=json.loads((tmp_path/'fit/GLOBAL_FIT_COMPLETE.json').read_text())
    projected=json.loads((tmp_path/'project/GLOBAL_PROJECTION_COMPLETE.json').read_text())
    assert fitted['training_tiles']==15 and fitted['one_fit_used_across_slide']
    assert projected['model_sha256']==fitted['model_sha256']
    prob=np.load(tmp_path/'project/group-probabilities-token-grid.f16.npy')
    np.testing.assert_allclose(prob[16:,16:].sum(2),1,atol=.003)
    assert not prob[:16,:16].any()
    rgb=pyvips.Image.new_from_file(str(tmp_path/'project/rgb9plus19-fullslide-0p5um.tif'))
    assert (rgb.height,rgb.width)==(889,890)
    spec['source_frame_id']='different-slide';write_json(tmp_path/'input.json',spec)
    with pytest.raises(ValueError,match='different image'):
        project_slide_responses(tmp_path/'input.json',tmp_path/'fit',tmp_path/'wrong-frame')
