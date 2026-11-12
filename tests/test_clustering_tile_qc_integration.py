"""The PCA/KODAMA input indices and anonymous group grid obey the strict cutoff."""
from pathlib import Path
import importlib.util,json,csv
import numpy as np

def test_pca_and_grouping_exclude_ten_percent_tiles(tmp_path,monkeypatch):
    project=Path(__file__).parents[1];path=project/'migration/bora-local-full1280-2026-10-29-v2/prepare_case.py'
    spec=importlib.util.spec_from_file_location('strict_tile_prepare',path);p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
    p.DATA=tmp_path;p.BASE=tmp_path/'analysis';p.ROOT=p.BASE/'local-bora';case='fixture';uid='fixture-image';source=p.BASE/'feature-fixture'/case;cal=source/'calibrated-01';cal.mkdir(parents=True);(p.ROOT/case).mkdir(parents=True)
    def write(path,value):path.write_text(json.dumps(value))
    image=tmp_path/'rgb-fixture';image.write_bytes(b'calibrated-image-fixture');image_sha=p.sha256(image)
    m=dict(target_image=str(image),target_sha256=image_sha,source_sha256='a'*64,grid_shape_yx=[4,8],true_working_size_wh=[1792,896],target_shape_yxc=[896,1792,3],source_mpp_xy=[.5,.5],effective_mpp_xy=[.5,.5],source_original_shape_yx=[896,1792])
    write(cal/'PREPROCESSED_COMPLETE.local.json',m);np.save(cal/'eligible-grid.npy',np.ones((4,8),bool))
    pooled=np.random.default_rng(42).normal(size=(4,8,1280)).astype(np.float32);assets={}
    for row in range(4):
        name=tmp_path/f'pool-{row}.npy';np.save(name,pooled[row]);assets[f'row-{row}']=dict(path=name.name,sha256=p.sha256(name))
    cls=tmp_path/'class-mask';cls.write_bytes(b'model-class-fixture');clean=tmp_path/'clean-mask';clean.write_bytes(b'model-binary-fixture');summary=tmp_path/'summary.json';write(summary,dict(level0_shape_yx=[896,1792],qc_mask_shape_yx=[4,8]))
    qc=dict(outputs={'fixture_grandqc_clean_tissue_mask.tif':dict(path=clean.name,sha256=p.sha256(clean)),'fixture_grandqc_class_mask.tif':dict(path=cls.name,sha256=p.sha256(cls)),'fixture_grandqc_summary.json':dict(path=summary.name,sha256=p.sha256(summary))})
    qc_ref=tmp_path/'qc-cache.json';write(qc_ref,qc)
    features={}
    for factor in (1,2,4):
        a=tmp_path/f'full1280-{factor}.npy';np.save(a,np.zeros((64//factor,128//factor,1280),np.float16));features[str(factor)]=dict(path=str(a),sha256=p.sha256(a))
    write(source/'CACHE_COMPLETE.json',dict(pooled_cache_manifest='pooled-cache',spatial_cache_manifest='spatial-cache',features=features))
    monkeypatch.setattr(p,'entry',lambda _:dict(case=case,image_id=uid,feature_root='feature-fixture',kind='registered-benchmark-image'))
    monkeypatch.setattr(p,'prepared_record',lambda _:dict(preprocessing_cache_manifest='preprocessing-cache',grandqc_cache_manifest=str(qc_ref)))
    monkeypatch.setattr(p,'require_image_inputs',lambda *a,**k:dict(virchow2_pooled=dict(outputs=assets),grandqc=qc))
    folder=p.BASE/'tile-qc-2026-10-29-v1'/uid;folder.mkdir(parents=True);bad=np.zeros((4,8),np.int64);bad[0,:3]=[9,10,11];bad[3,7]=100;eligible=np.ones((4,8),bool);eligible[0,1:3]=False;eligible[3,7]=False
    payloads={'background-or-artifact-area.npy':bad,'unknown-area.npy':np.zeros((4,8),np.int64),'clustering-eligible.npy':eligible,'clean-percent.npy':100-bad,'background-percent.npy':bad,'artifact-percent.npy':np.zeros((4,8)),'background_or_artifact-percent.npy':bad,'unknown-percent.npy':np.zeros((4,8))}
    files={}
    for name,value in payloads.items():np.save(folder/name,value);files[name]=dict(sha256=p.sha256(folder/name))
    (folder/'tile-qc-percentages.tsv').write_text('fixture table\n');files['tile-qc-percentages.tsv']=dict(sha256=p.sha256(folder/'tile-qc-percentages.tsv'))
    write(folder/'TILE_QC_COMPLETE.json',dict(status='COMPLETE',clustering_policy=dict(comparison='<',background_or_artifact_percent=10),input_identity=dict(calibrated_image_sha256=image_sha,class_mask_sha256=p.sha256(cls)),grid_shape_yx=[4,8],tile_area_denominator=100,files=files))
    p.prepare(case);root=p.ROOT/case;indices=np.load(root/'prepare-01/eligible-tile-indices.npy');receipt=json.loads((root/'prepare-01/PREPARE_COMPLETE.json').read_text())
    assert receipt['selected_tiles']==29 and receipt['clustering_background_artifact_percent_lt']==10
    assert 0 in indices and not any(x in indices for x in (1,2,31))
    assert np.array_equal(indices,np.flatnonzero(eligible))
    assert np.array_equal(np.load(root/'prepare-01/all-pooled-1280.f32.npy'),pooled.reshape(-1,1280))
    kodama=root/'kodama-01';kodama.mkdir();write(kodama/'KODAMA_COMPLETE.json',dict(status='COMPLETE',groups=2,selected_tiles=29))
    with (kodama/'clusters.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['row_zero_based','cluster']);writer.writerows((i,1+i%2) for i in range(29))
    p.groups(case);groups=np.load(root/'groups.npy');assert np.array_equal(groups>0,eligible)
    response=json.loads((root/'response-input-full1280.json').read_text());assert response['format']=='bora-wsi-response-input-v2-full1280'
    assert all(np.load(v['path'],mmap_mode='r').shape[-1]==1280 for v in response['features'].values())
