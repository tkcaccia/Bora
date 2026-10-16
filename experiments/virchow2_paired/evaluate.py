"""Post-freeze fidelity, anatomy-proxy and expert segmentation assessment."""
import argparse,json
from itertools import combinations
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import tifffile
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from skimage.color import rgb2hed,rgb2lab
from skimage.filters import threshold_otsu
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.measure import regionprops
from skimage.metrics import structural_similarity,peak_signal_noise_ratio
from experiments.virchow2_inverse.common import digest,save_json
from experiments.virchow2_inverse.finalize import score,font,overlay,rgba_mask
from experiments.virchow2_inverse.protocol import COLOURS
from .reconstruct import GAINS


def fidelity(reference,image):
    r=rgb2hed(reference.astype(float)/255)[...,0];v=rgb2hed(image.astype(float)/255)[...,0]
    gy,gx=np.gradient(ndi.gaussian_filter(r,.8));rg=np.hypot(gy,gx)
    gy,gx=np.gradient(ndi.gaussian_filter(v,.8));vg=np.hypot(gy,gx)
    return dict(psnr_db=float(peak_signal_noise_ratio(reference,image,data_range=255)),
        ssim_rgb=float(structural_similarity(reference,image,channel_axis=-1,data_range=255)),
        hematoxylin_correlation=float(np.corrcoef(r.ravel(),v.ravel())[0,1]),
        hematoxylin_gradient_correlation=float(np.corrcoef(rg.ravel(),vg.ravel())[0,1]))


def nuclear_candidates(image,threshold=None):
    h=ndi.gaussian_filter(rgb2hed(image.astype(float)/255)[...,0],.8)
    threshold=float(threshold_otsu(h)) if threshold is None else threshold
    mask=h>threshold
    dist=ndi.distance_transform_edt(mask)
    peaks=peak_local_max(dist,min_distance=3,threshold_abs=1.2,labels=mask,exclude_border=False)
    markers=np.zeros(mask.shape,np.int32)
    if len(peaks):markers[tuple(peaks.T)]=np.arange(1,len(peaks)+1)
    regions=watershed(-dist,markers,mask=mask)
    accepted=[r for r in regionprops(regions) if 6<=r.area<=250]
    centroids=np.array([r.centroid for r in accepted],dtype=float).reshape(-1,2)
    return centroids,threshold


def match_nuclei(reference,predicted,radius_px=4):
    n,m=len(reference),len(predicted)
    if n==0 or m==0:return dict(reference_count=n,reconstruction_count=m,matches=0,precision=0.,recall=0.,f1=0.,median_displacement_um=None)
    neighbours=cKDTree(predicted).query_ball_point(reference,radius_px)
    rr=[];cc=[];cost=[]
    for row,cols in enumerate(neighbours):
        for col in cols:
            rr.append(row);cc.append(col);cost.append(float(np.linalg.norm(reference[row]-predicted[col]))+1e-6)
    rr.extend(range(n));cc.extend(m+np.arange(n));cost.extend([1e6]*n)
    matrix=coo_matrix((cost,(rr,cc)),shape=(n,m+n)).tocsr()
    rows,cols=min_weight_full_bipartite_matching(matrix)
    real=cols<m;rows=rows[real];cols=cols[real];matches=len(rows)
    distances=np.linalg.norm(reference[rows]-predicted[cols],axis=1)*.5
    return dict(reference_count=n,reconstruction_count=m,matches=matches,precision=matches/m,recall=matches/n,
        f1=2*matches/(n+m),median_displacement_um=float(np.median(distances)) if matches else None,
        matching="one-to-one maximum-cardinality, then minimum-distance; 2um tolerance; fixed original-image H threshold",
        expert_nuclei_annotations=False)


def contrast(image,coarse):
    lab=rgb2lab(image.astype(float)/255);values=[];labels=[]
    for y in range(0,coarse.shape[0],90):
        for x in range(0,coarse.shape[1],90):
            region=coarse[y:y+90,x:x+90]
            if min(region.shape)<72:continue
            ids,count=np.unique(region,return_counts=True);group=int(ids[count.argmax()])
            if group==0 or count.max()/count.sum()<.95:continue
            values.append(lab[y+22:min(y+67,coarse.shape[0]),x+22:min(x+67,coarse.shape[1])].mean((0,1)));labels.append(group)
    values=np.array(values);labels=np.array(labels)
    means={g:values[labels==g].mean(0) for g in range(1,6)}
    variance={g:values[labels==g].var(0) for g in range(1,6)}
    pairs={f"{a}-{b}":float(np.sum((means[a]-means[b])**2)/max(float(np.sum(variance[a]+variance[b])),1e-8))
           for a,b in combinations(range(1,6),2)}
    return dict(worst_pair_fisher=min(pairs.values()),mean_pair_fisher=float(np.mean(list(pairs.values()))),pairs=pairs,
        measurement="central-half Lab colour means per90px coarse tile; descriptive class contrast, not a significance test")


def panel(images,titles,path,subtitles=None,size=448):
    canvas=Image.new("RGB",(len(images)*size, size+108),"white");draw=ImageDraw.Draw(canvas)
    for i,(im,title) in enumerate(zip(images,titles)):
        im=Image.fromarray(im);factor=size/max(im.size)
        im=im.resize((round(im.width*factor),round(im.height*factor)),Image.Resampling.NEAREST)
        canvas.paste(im,(i*size+(size-im.width)//2,86+(size-im.height)//2))
        draw.text((i*size+14,12),title,fill="#172a35",font=font(23))
        if subtitles:draw.text((i*size+14,46),subtitles[i],fill="#405060",font=font(17))
    canvas.save(path)


def main():
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True)
    p.add_argument("--source",type=Path,required=True);p.add_argument("--training",type=Path,required=True)
    p.add_argument("--previous",type=Path,required=True);a=p.parse_args()
    frozen=json.loads((a.root/"PREDICTIONS_FROZEN.json").read_text())
    for filename,expected in frozen["files_sha256"].items():
        if digest(a.root/filename)!=expected:raise ValueError(f"Output changed after freeze: {filename}")
    protocol=json.loads((a.root/"PROTOCOL.json").read_text())
    original=np.array(Image.open(a.root/"original.png"));coarse=tifffile.imread(a.root/"coarse.tif")
    nuclei,threshold=nuclear_candidates(original)
    recon={};tiles={};metrics={}
    for name in GAINS:
        recon[name]=np.array(Image.open(a.root/f"{name}.png"))
        tiles[name]=np.array(Image.open(a.root/f"tile.{name}.png"))
        candidate,_=nuclear_candidates(recon[name],threshold)
        metrics[name]=dict(reconstruction=fidelity(original,recon[name]),nuclei_proxy=match_nuclei(nuclei,candidate),
                           coarse_class_contrast=contrast(recon[name],coarse))
    tile_original=np.array(Image.open(a.root/"tile.original.png"))
    for name in GAINS:metrics[name]["focal_direct_window"]=fidelity(tile_original,tiles[name])
    # Test reconstructions were frozen without consulting any test RGB pixels.
    test=np.load(a.root/"test_reconstructions.npy",mmap_mode="r");ids=np.load(a.root/"test_ids.npy")
    targets=np.load(a.training/"rgb.npy",mmap_mode="r");rows=json.loads((a.training/"corpus.json").read_text())["patches"]
    test_rows=[]
    for image,i in zip(test,ids):
        reference=np.asarray(targets[i]).transpose(1,2,0)
        test_rows.append(dict(index=int(i),case=rows[i]["case"],**fidelity(reference,image)))
    cases=sorted(set(r["case"] for r in test_rows))
    case_metrics={case:{key:float(np.mean([r[key] for r in test_rows if r["case"]==case]))
                  for key in ("psnr_db","ssim_rgb","hematoxylin_correlation","hematoxylin_gradient_correlation")} for case in cases}
    test_summary={key:dict(case_macro_mean=float(np.mean([r[key] for r in case_metrics.values()])),
                      case_median=float(np.median([r[key] for r in case_metrics.values()])),
                      case_q25_q75=np.percentile([r[key] for r in case_metrics.values()],[25,75]).tolist())
                  for key in next(iter(case_metrics.values()))}
    # Only now is the fine expert tissue reference opened.
    truth=tifffile.imread(a.source/"expert_reference.compact.tif")
    segmentation={"coarse":score(coarse,truth),"original":score(tifffile.imread(a.root/"bora.original.tif"),truth)}
    for name in GAINS:segmentation[name]=score(tifffile.imread(a.root/f"bora.{name}.tif"),truth)
    full=metrics["full"];primary=metrics["selective"];gate=protocol["fidelity_gate"]
    full_pass=(full["reconstruction"]["ssim_rgb"]>=gate["full_rgb_ssim_min"] and
        full["focal_direct_window"]["ssim_rgb"]>=gate["full_direct_tile_rgb_ssim_min"] and
        full["nuclei_proxy"]["f1"]>=gate["full_nuclei_proxy_f1_min"] and
        full["nuclei_proxy"]["median_displacement_um"] is not None and
        full["nuclei_proxy"]["median_displacement_um"]<=gate["full_nuclei_median_displacement_um_max"])
    anatomy_preserved=primary["nuclei_proxy"]["f1"]>=full["nuclei_proxy"]["f1"]-gate["selective_nuclei_f1_drop_max"]
    contrast_improved=primary["coarse_class_contrast"]["worst_pair_fisher"]>full["coarse_class_contrast"]["worst_pair_fisher"]
    improvement=(segmentation["selective"]["macro_iou"]-segmentation["original"]["macro_iou"]>gate["selective_macro_iou_improvement_min"])
    conclusion=dict(full_reconstruction_passed_fidelity_screen=bool(full_pass),selective_retains_full_nuclei_proxy=bool(anatomy_preserved),
        selective_improves_worst_class_colour_contrast=bool(contrast_improved),selective_bora_beats_original_he=bool(improvement),
        preprocessing_proof_of_concept_supported=bool(full_pass and anatomy_preserved and contrast_improved and improvement),
        caution="Proxy object matches are not expert nuclear annotations. One familiar ROI, reference-derived coarse labels, no independent segmentation validation.")
    result=dict(metrics=metrics,segmentation=segmentation,test_summary=test_summary,test_cases=case_metrics,test_patches=test_rows,
        test_case_count=len(cases),test_patch_count=len(ids),conclusion=conclusion,original_coarse_contrast=contrast(original,coarse),
        nuclei_threshold=threshold,fine_reference_sha256=digest(a.source/"expert_reference.compact.tif"))
    save_json(a.root/"results.json",result)
    # Fixed examples cover test-case SSIM quartiles, without hiding difficult cases.
    ordered=sorted(test_rows,key=lambda r:r["ssim_rgb"])
    chosen=[ordered[round(q*(len(ordered)-1))] for q in (0,.25,.5,.75,1)]
    save_json(a.root/"test_visual_examples.json",chosen)
    for index,row in enumerate(chosen):
        i=row["index"];j=int(np.where(ids==i)[0][0])
        panel([np.asarray(targets[i]).transpose(1,2,0),np.asarray(test[j])],
              [f"Test case {row['case']}","All-feature reconstruction"],a.root/f"test_example_{index}.png",
              ["Held-out source case",f"SSIM {row['ssim_rgb']:.3f}; PSNR {row['psnr_db']:.2f} dB"],size=336)
    panel([tile_original,tiles["full"],tiles["selective"]],["Original H&E","Full spatial features","Selective features (1.25x)"],
        a.root/"paired_tile.png",["224 px = 112 µm; same field of view",
            f"SSIM {metrics['full']['focal_direct_window']['ssim_rgb']:.3f}",f"SSIM {metrics['selective']['focal_direct_window']['ssim_rgb']:.3f}"],size=448)
    panel([original,recon["full"],recon["selective"]],["Original tissue ROI","Full reconstruction","Selective reconstruction"],a.root/"paired_roi.png",
        ["1224 × 1166 px; 0.5 µm/px",f"SSIM {full['reconstruction']['ssim_rgb']:.3f}",f"SSIM {primary['reconstruction']['ssim_rgb']:.3f}"],size=612)
    images=[overlay(original,truth),overlay(original,coarse),overlay(original,tifffile.imread(a.root/"bora.original.tif")),
            overlay(original,tifffile.imread(a.root/"bora.full.tif")),overlay(original,tifffile.imread(a.root/"bora.selective.tif"))]
    panel(images,["Expert annotation","Coarse tile mask","Bora: original H&E","Bora: full reconstruction","Bora: selective features"],a.root/"segmentation.png",
          ["35% opacity","90 px = 45 µm",*[f"Macro IoU {segmentation[n]['macro_iou']:.3f}" for n in ("original","full","selective")]],size=400)
    Image.fromarray(rgba_mask(truth)).save(a.root/"expert.mask.png");Image.fromarray(rgba_mask(coarse)).save(a.root/"coarse.mask.png")
    for name in ["original",*GAINS]:Image.fromarray(rgba_mask(tifffile.imread(a.root/f"bora.{name}.tif"))).save(a.root/f"bora.{name}.mask.png")
    if (a.root/"CONSISTENCY_FROZEN.json").exists():
        consistency=json.loads((a.root/"CONSISTENCY_FROZEN.json").read_text())
        for filename,expected in consistency["files_sha256"].items():
            if digest(a.root/filename)!=expected:raise ValueError("Consistency image changed after freeze")
        candidate_original,tile_threshold=nuclear_candidates(tile_original)
        diagnostics={};optimized={}
        for name in ("full","selective"):
            optimized[name]=np.array(Image.open(a.root/f"tile.consistency.{name}.png"))
            for label,pixels in [("decoder",tiles[name]),("consistency",optimized[name])]:
                candidates,_=nuclear_candidates(pixels,tile_threshold)
                diagnostics[f"{label}.{name}"]=dict(**fidelity(tile_original,pixels),nuclei_proxy=match_nuclei(candidate_original,candidates))
        save_json(a.root/"consistency_results.json",dict(metrics=diagnostics,
            scope="focal224px tile only; no ROI reconstruction or Bora claim; primary results unchanged",
            protocol=json.loads((a.root/"CONSISTENCY_PROTOCOL.json").read_text())))
        panel([tile_original,optimized["full"],optimized["selective"]],
            ["Original H&E","Full tokens + consistency","Selective + consistency"],a.root/"paired_tile_consistency.png",
            ["224 px = 112 µm; same native tile",f"SSIM {diagnostics['consistency.full']['ssim_rgb']:.3f}",
             f"SSIM {diagnostics['consistency.selective']['ssim_rgb']:.3f}"],size=448)
    save_json(a.root/"COMPLETE.json",dict(completed=True,predictions_verified=True,conclusion=conclusion,
        evaluated_implementation_sha256=digest(__file__)))
    print(json.dumps(dict(test=test_summary,conclusion=conclusion,macro_iou={k:v["macro_iou"] for k,v in segmentation.items()})),flush=True)

if __name__=="__main__":main()
