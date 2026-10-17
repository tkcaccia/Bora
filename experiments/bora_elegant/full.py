"""Frozen final methods: full-corpus prediction, scoring and exhaustive rendering."""
import argparse,csv,gc,json,time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import tifffile
from threadpoolctl import threadpool_limits
from experiments.virchow2_inverse.common import digest,save_json
from experiments.bora_strategies.prepare import verify
from experiments.bora_strategies.context import Context
from experiments.bora_strategies.methods import run as old_run
from experiments.bora_strategies.registry import STRATEGIES as OLD,CONVERSIONS,CONSENSUS,REJECTED_STRATEGIES
from .core import predict,TITLES,resize

FINAL_MAP={"B21":"E09","B22":"E11"}


def freeze(args):
    source,root,sample,selection=args;src=source/'cases'/sample['case'];out=root/'cases'/sample['case'];out.mkdir(parents=True,exist_ok=True)
    cv2.setNumThreads(2);verify(src,json.loads((src/'INPUTS_FROZEN.json').read_text()))
    coarse=tifffile.imread(src/'coarse.tif');completed=tifffile.imread(src/'coarse.completed.tif');support=tifffile.imread(src/'tissue.working.tif')>0;ns=tifffile.imread(src/'tissue.native.tif')>0
    he=np.array(Image.open(src/'original.png').convert('RGB'));save_json(out/'SAMPLE.json',sample)
    tiles=resize(completed,sample['native_shape_yx'],True);tiles[~ns]=0;tifffile.imwrite(out/'tiles.native.tif',tiles,compression='deflate');manifests={}
    for m in CONVERSIONS:
        folder=out/m;folder.mkdir(exist_ok=True)
        if (folder/'FROZEN.json').exists():verify(folder,json.loads((folder/'FROZEN.json').read_text()));manifests[sample['case']+'/'+m]=digest(folder/'FROZEN.json');continue
        rgb=np.array(Image.open(src/m/'representation.png').convert('RGB'));records={};files={}
        with threadpool_limits(limits=2):
            for key in selection['methods']:
                start=time.perf_counter()
                if key in OLD:
                    c=Context(rgb,he,coarse,completed,support,sample['working_mpp_xy'])
                    if key=='B19':
                        for component in CONSENSUS:old_run(component,c)
                    analysis=c.finish(old_run(key,c));details=c.diagnostics.get(key,{});ashape=list(c.shape);ampp=c.mpp_xy.tolist();del c
                else:
                    analysis,details=predict(FINAL_MAP.get(key,key),rgb,coarse,support,sample['working_mpp_xy'],palette=sample['palette'])
                    ashape=details.get('analysis_shape_yx',list(analysis.shape));ampp=details.get('analysis_mpp_xy',[sample['working_shape_yx'][1]*sample['working_mpp_xy'][0]/analysis.shape[1],sample['working_shape_yx'][0]*sample['working_mpp_xy'][1]/analysis.shape[0]])
                native=resize(analysis,sample['native_shape_yx'],True);native[~ns]=0;seconds=time.perf_counter()-start
                assert native.shape==tuple(sample['native_shape_yx']) and np.all(native[ns]>0) and np.all(native[~ns]==0) and set(np.unique(native))<=set(np.unique(coarse))|{0}
                target=folder/key;target.mkdir(exist_ok=True)
                for name,array in [('mask.analysis.tif',analysis),('bora.native.tif',native)]:
                    path=target/name;tifffile.imwrite(path,array,compression='deflate');files[str(path.relative_to(folder))]=digest(path)
                records[key]=dict(seconds=seconds,diagnostics=details,analysis_shape_yx=ashape,analysis_mpp_xy=ampp,fine_reference_used=False)
                save_json(target/'details.json',records[key]);files[key+'/details.json']=digest(target/'details.json');gc.collect()
        save_json(folder/'FROZEN.json',dict(files_sha256=files,records=records,tiles_native_sha256=digest(out/'tiles.native.tif'),input_manifest_sha256=digest(src/'INPUTS_FROZEN.json'),rgb_sha256=digest(src/m/'representation.png'),analysis_shape_yx=ashape,analysis_mpp_xy=ampp,native_shape_yx=sample['native_shape_yx'],native_mpp_xy=sample['native_mpp_xy'],fine_reference_read=False))
        manifests[sample['case']+'/'+m]=digest(folder/'FROZEN.json')
    print(json.dumps(dict(predicted=sample['case'],conversions=len(CONVERSIONS),methods=len(selection['methods']))),flush=True);return manifests


def score_one(args):
    source,root,case,protocol=args
    from experiments.bora_strategies import score as scoring
    strategies={k:(v['title'],v['hypothesis']) for k,v in protocol['strategies'].items()}
    with threadpool_limits(limits=2):return scoring.score_case(case,root,source,protocol,strategies=strategies)


def summary(rows,keys,conversion):
    if not rows:return []
    rng=np.random.default_rng(7730);idx=rng.integers(0,len(rows),(3000,len(rows)));ranking=[]
    for key in keys:
        values=np.array([r['conversions'][conversion]['strategies'][key]['score']['macro_iou'] for r in rows]);base=np.array([r['tiles']['macro_iou'] for r in rows]);delta=values-base
        bf=[r['conversions'][conversion]['strategies'][key]['score']['boundary_f1_at_um']['8'] for r in rows];bf=[v for v in bf if v is not None]
        seconds=[r['conversions'][conversion]['strategies'][key]['seconds'] for r in rows]
        ranking.append(dict(strategy=key,title=keys[key]['title'],mean_macro_iou=float(values.mean()),dataset_balanced_macro_iou=float(values.mean()),delta_vs_tiles=float(delta.mean()),paired_delta_bootstrap95=np.percentile(delta[idx].mean(1),[2.5,97.5]).tolist(),boundary_f1_8um=float(np.mean(bf)) if bf else None,mean_seconds=float(np.mean(seconds)),wins=int((delta>1e-8).sum()),losses=int((delta<-1e-8).sum()),ties=int((abs(delta)<=1e-8).sum())))
    ranking.sort(key=lambda r:-r['mean_macro_iou'])
    for i,r in enumerate(ranking):r['rank']=i+1
    return ranking


def render_one(args):
    source,root,row,protocol=args
    from experiments.bora_strategies import publish
    strategies={k:(v['title'],v['hypothesis']) for k,v in protocol['strategies'].items()}
    return publish.render(root,source,row,strategies=strategies)


def report(root,result):
    p=result['protocol'];dev=result['development_ranking'];rest=result['evaluation_ranking'];lines=['# Whole-field Bora: fixed RGB, revised segmentation mathematics','',f"{len(result['cases'])} ROIs, five byte-identical RGB conversions, {len(p['strategies'])} active methods. Development uses nine user-specified ROIs and the fixed equal 9+19 conversion. Primary metric is arithmetic mean native per-image macro IoU; no dataset balancing and no per-case conversion selection.",'','B05/B09/B10/B12 and all earlier rejected solvers are excluded. B19 revision 3 uses only B02/B03/B08. No new Virchow2 inference was performed.','', '## Proposed mathematics','']
    for k,v in p['strategies'].items():lines += [f"- **{k} {v['title']}**: {v['hypothesis']}"]
    lines+=['','## Fixed protocol and interpretation','','All converted RGB files, coarse masks and tissue support retain their original hashes. All detected tissue is labelled. Glass is removed after segmentation. Original reference >0 is the scoring support, including annotated pixels wrongly removed by tissue detection; unannotated tissue is predicted but unscored. Missing fine-reference classes remain in macro metrics. All cases are ROIs; native, working and analysis pixel spacing are recorded. Pixel spacing does not establish optical resolving power. CAMELYON non-tumor reference is automatically detected tissue, not expert stroma.','','The nine development images are tuning data. The other 140 are additional evaluation cases, not an untouched external test. Coarse supervision is derived from reference annotations. Predictions are frozen before fine-reference scoring, but development choices use development scores. This evaluates controlled refinement, not independent tissue classification.','','CPU timings include each method\'s preprocessing, inference and native projection, excluding shared image loading, disk compression, scoring and upstream fixed RGB/Virchow2 computation. Development timings are repeated; full-corpus timings are one pass on the same bounded CPU workers. B19 includes its components. Bootstrap intervals are descriptive paired case resamples.','','## Results: fixed equal 9+19 conversion','']
    for title,ranking in [('Nine development images',dev),('Other 140 images',rest),('Full 149 images',result['ranking'])]:
        lines += ['### '+title,'','| Method | Mean macro IoU | Delta vs tiles | Boundary F1 8 um | Mean seconds |','|---|---:|---:|---:|---:|']
        for r in ranking:lines.append(f"| {r['strategy']} {r['title']} | {r['mean_macro_iou']:.6f} | {r['delta_vs_tiles']:+.6f} | {r['boundary_f1_8um'] or 0:.6f} | {r['mean_seconds']:.4f} |")
    lines+=['','## Development and reproducibility','','DEVELOPMENT.json records all rounds, hypotheses, failures, selection and stopping. SELECTION.json was frozen before full-corpus scoring. Every native TIFF and synchronized overlay is available in gallery.html; all_outputs.html contains all case/conversion sheets. Per-class scores, timings and hashes are in results.json, scores.csv, PROTOCOL.json and AUDIT.json. The legacy field dataset_balanced_macro_iou is retained for viewer compatibility but contains the arithmetic case mean and is never dataset balanced in this experiment.','']
    (root/'REPORT.md').write_text('\n'.join(lines))


def publish(root,source,result):
    protocol=result['protocol']
    with ProcessPoolExecutor(max_workers=2,mp_context=get_context('spawn')) as pool:result['cases']=list(pool.map(render_one,[(source,root,row,protocol) for row in result['cases']]))
    save_json(root/'results.json',result);report(root,result)
    template=Path(__file__).with_name('gallery.html').read_text();(root/'gallery.html').write_text(template.replace('__DATA__',json.dumps(result,allow_nan=False)))
    blocks=[]
    for r in result['cases']:
        blocks.append(f'<section><h2>{r["case"]} · {r["cohort"]}</h2>')
        for m in CONVERSIONS:blocks.append(f'<h3>RGB {m}</h3><a href="gallery.html#{r["case"]}/{m}"><img loading="lazy" src="cases/{r["case"]}/{m}/contact.jpg"></a>')
        blocks.append('</section>')
    (root/'all_outputs.html').write_text('<!doctype html><meta charset="utf-8"><title>Bora whole-field: all outputs</title><style>body{font:15px system-ui;background:#edf3f6;color:#203744;margin:18px}section{background:white;padding:15px;margin:25px 0}img{width:100%;max-width:1120px}a{color:#09688e}</style><h1>149 ROIs · five fixed RGB conversions · whole-field Bora</h1><p><a href="gallery.html">Synchronized viewer</a> · <a href="scores.csv">Scores</a> · <a href="REPORT.md">Report</a></p>'+''.join(blocks))


def audit(source,root,result):
    manifests=json.loads((root/'ALL_PREDICTIONS_FROZEN.json').read_text());native_masks=0;rgb_count=0
    assert digest(root/'PROTOCOL.json')==manifests['protocol_sha256']
    for name,sha in result['protocol']['predictor_sources_sha256'].items():assert digest(Path(__file__).parent/name)==sha
    assert digest(root/'SELECTION.json')==result['protocol']['selection_sha256']
    repo=Path(__file__).resolve().parents[2]
    for name,sha in result['protocol']['runtime_sources_sha256'].items():assert digest(repo/name)==sha
    for row in result['cases']:
        case=row['case'];src=source/'cases'/case;out=root/'cases'/case;verify(src,json.loads((src/'INPUTS_FROZEN.json').read_text()))
        truth=tifffile.imread(src/'reference.native.tif');support=tifffile.imread(src/'tissue.native.tif')>0
        assert digest(src/'reference.native.tif')==row['sample']['reference_native_sha256'];assert row['coverage']['annotated_pixels']==int((truth>0).sum())
        for m in CONVERSIONS:
            folder=out/m;rec=json.loads((folder/'FROZEN.json').read_text());verify(folder,rec);assert digest(folder/'FROZEN.json')==manifests['conversion_manifest_sha256'][case+'/'+m]
            assert rec['rgb_sha256']==digest(src/m/'representation.png');rgb_count+=1
            for key in result['protocol']['strategies']:
                mask=tifffile.imread(folder/key/'bora.native.tif');assert mask.shape==truth.shape and np.all(mask[support]>0) and np.all(mask[~support]==0)
                visual=Image.open(folder/key/'mask.png');assert visual.mode=='P' and np.array_equal(np.array(visual),mask)
                assert np.array_equal(np.array(Image.open(folder/key/'difference.mask.png'))>0,mask!=tifffile.imread(out/'tiles.native.tif'))
                score=row['conversions'][m]['strategies'][key]['score'];assert score['valid_pixels']==int((truth>0).sum()) and set(score['per_class'])==set(map(str,np.unique(truth[truth>0])))
                if key in ('B02','B03','B08'):
                    previous=source.parent/'bora-retained-expanded-2026-10-17'/'cases'/case/m/key/'bora.native.tif'
                    assert np.array_equal(mask,tifffile.imread(previous))
                native_masks+=1
    files={str(f.relative_to(root)):digest(f) for f in root.rglob('*') if f.is_file() and f.suffix in ('.png','.jpg','.html')};save_json(root/'VISUAL_MANIFEST.json',dict(files_sha256=files))
    audit=dict(cases=len(result['cases']),native_masks=native_masks,unchanged_rgb_inputs=rgb_count,unchanged_retained_native_masks=rgb_count*3,visuals=len(files),all_tissue_assigned=True,background_removed=True,quality_support_unchanged=True,all_reference_classes_scored=True,predictions_frozen_before_scoring=True,predictor_sources_verified=True,all_checks_passed=True)
    save_json(root/'AUDIT.json',audit);save_json(root/'COMPLETE.json',audit);print(json.dumps(audit),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--selection',type=Path,required=True);a=p.parse_args();selection=json.loads(a.selection.read_text());a.root.mkdir(parents=True,exist_ok=True)
    plan=json.loads((a.source/'PLAN.json').read_text());dev=selection['development_cases'];samples=[]
    inputs=json.loads((a.source/'ALL_INPUTS_FROZEN.json').read_text());assert inputs['plan_sha256']==digest(a.source/'PLAN.json')
    for sample in plan['samples']:assert inputs['manifests'][sample['case']]==digest(a.source/'cases'/sample['case']/'INPUTS_FROZEN.json')
    for sample in plan['samples']:
        sample=dict(sample);sample['benchmark_cohort']='development' if sample['case'] in dev else 'evaluation';samples.append(sample)
    strategies={k:dict(title=OLD[k][0] if k in OLD else TITLES[FINAL_MAP.get(k,k)],hypothesis=selection['hypotheses'].get(k,OLD.get(k,('',''))[1])) for k in selection['methods']}
    protocol=dict(source_folder=a.source.name,inputs_sha256=digest(a.source/'ALL_INPUTS_FROZEN.json'),source_plan_sha256=digest(a.source/'PLAN.json'),cases=[s['case'] for s in samples],conversions=list(CONVERSIONS),strategies=strategies,development_cases=dev,primary_conversion='ensemble',primary_metric='arithmetic case mean of native macro IoU',selection_sha256=digest(a.selection),predictor_sources_sha256={f.name:digest(f) for f in Path(__file__).parent.glob('*.py')},runtime_sources_sha256={str(f.relative_to(Path(__file__).resolve().parents[2])):digest(f) for folder in ('src','experiments') for f in (Path(__file__).resolve().parents[2]/folder).rglob('*.py')},rejected_strategies=list(REJECTED_STRATEGIES),consensus_components=list(CONSENSUS),consensus_revision=3,compute=dict(workers=3,threads=2))
    if (a.root/'PROTOCOL.json').exists():assert json.loads((a.root/'PROTOCOL.json').read_text())==protocol
    save_json(a.root/'PROTOCOL.json',protocol);save_json(a.root/'SELECTION.json',selection)
    if not (a.root/'ALL_PREDICTIONS_FROZEN.json').exists():
        manifests={}
        with ProcessPoolExecutor(max_workers=3,mp_context=get_context('spawn')) as pool:
            for record in pool.map(freeze,[(a.source,a.root,s,selection) for s in samples]):manifests.update(record)
        save_json(a.root/'ALL_PREDICTIONS_FROZEN.json',dict(protocol_sha256=digest(a.root/'PROTOCOL.json'),conversion_manifest_sha256=manifests,fine_reference_read=False))
    cases=[];csvrows=[]
    with ProcessPoolExecutor(max_workers=3,mp_context=get_context('spawn')) as pool:
        for row,records in pool.map(score_one,[(a.source,a.root,s['case'],protocol) for s in samples]):cases.append(row);csvrows.extend(records)
    development=[r for r in cases if r['cohort']=='development'];evaluation=[r for r in cases if r['cohort']=='evaluation'];per={m:summary(cases,strategies,m) for m in CONVERSIONS};other={m:summary(evaluation,strategies,m) for m in CONVERSIONS}
    result=dict(cases=cases,protocol=protocol,ranking=per['ensemble'],development_ranking=summary(development,strategies,'ensemble'),evaluation_ranking=other['ensemble'],new_image_ranking=other['ensemble'],per_conversion=per,new_per_conversion=other,metric=protocol['primary_metric'])
    save_json(a.root/'results.json',result)
    with (a.root/'scores.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(csvrows[0]));writer.writeheader();writer.writerows(csvrows)
    publish(a.root,a.source,result);audit(a.source,a.root,result)

if __name__=='__main__':main()
