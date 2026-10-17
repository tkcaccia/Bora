"""Deterministic native-ROI selection and physically calibrated coarse inputs.

Fine annotations are used here ONLY to choose boundary-containing fields and
make controlled majority-tile inputs. Prediction modules cannot open them.
"""
import argparse,colorsys,json,re,shutil
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import tifffile
from experiments.virchow2_inverse.common import digest,save_json,coarse_tile_observations

QUOTAS={"tiger":16,"bcss":16,"camelyon16-subset":8,"beetle":10}
LABELS={
 "tiger":{1:"invasive tumor",2:"tumor-associated stroma",3:"in-situ tumor",4:"healthy glands",5:"necrosis",6:"inflamed stroma",7:"rest"},
 "bcss":{1:"tumor",2:"stroma",3:"lymphocytic infiltrate",4:"necrosis/debris",5:"glandular secretions",6:"blood",8:"metaplasia NOS",9:"fat",10:"plasma cells",11:"other immune infiltrate",12:"mucoid material",13:"normal acinus/duct",14:"lymphatics",16:"nerve",17:"skin adnexa",18:"blood vessel",19:"angioinvasion",20:"DCIS",21:"other"},
 "camelyon16-subset":{1:"automatically detected non-tumor tissue",2:"expert-derived tumor"},
 "beetle":{1:"other",2:"non-invasive epithelium",3:"invasive epithelium",4:"necrosis"}}
RED=[220,45,65];GREEN=[55,165,75];BLUE=[55,110,230];PURPLE=[200,100,220];GOLD=[220,180,55];GREY=[115,120,135]
FIXED={"tiger":{1:RED,2:GREEN,3:[20,190,205],4:BLUE,5:GREY,6:PURPLE,7:GOLD},
 "bcss":{1:RED,2:GREEN,3:PURPLE,4:GREY,9:GOLD,13:BLUE},
 "camelyon16-subset":{1:GREEN,2:RED},"beetle":{1:GOLD,2:BLUE,3:RED,4:GREY}}


def palette(dataset,codes):
    return [FIXED[dataset].get(g,[round(v*255) for v in colorsys.hsv_to_rgb((g*.61803398875)%1,.7,.9)]) for g in codes]


def coarse_mask(truth,tile=90):
    """Majority including ignore0; do not inject fine spatial support afterward."""
    out=np.zeros_like(truth)
    for y in range(0,truth.shape[0],tile):
        for x in range(0,truth.shape[1],tile):
            a=truth[y:y+tile,x:x+tile];codes,count=np.unique(a,return_counts=True)
            out[y:y+tile,x:x+tile]=codes[np.argmax(count)]
    return out


def mpp(row,data):
    if row["dataset"]=="beetle":
        path=data/"raw/beetle"/(row["source_slide"]+"_image.tif")
        with tifffile.TiffFile(path) as tif:
            tags=tif.pages[0].tags;unit=int(tags["ResolutionUnit"].value)
            factor={2:25400.,3:10000.}.get(unit)
            if factor is None:raise ValueError("TIFF has no physical unit")
            rationals=[tags[k].value for k in ("XResolution","YResolution")]
        return [factor*b/a for a,b in rationals],dict(source=str(path),unit=unit,rationals=rationals,status="embedded TIFF axis spacing")
    if row.get("mpp_um") is not None:return [row["mpp_um"]]*2,dict(status="provider/mirror reported",source_dataset=row["dataset"])
    return [row["mpp_x_um"],row["mpp_y_um"]],dict(status="embedded WSI axis spacing",source_slide=row["slide_id"])


def group_key(row):
    if row["dataset"]!="tiger":return row["group_id"]
    name=row["slide_id"]
    return re.match(r"TC_S\d+_P\d+",name).group() if name.startswith("TC_") else re.match(r"\d+",name).group()


def candidate_windows(truth,spacing,max_side=1280):
    height,width=truth.shape
    ch=min(height,round(max_side*.5/spacing[1]));cw=min(width,round(max_side*.5/spacing[0]))
    if min(round(ch*spacing[1]/.5),round(cw*spacing[0]/.5))<896:return []
    def starts(n,s):return sorted(set([*range(0,n-s+1,max(1,s//2)),n-s]))
    ranked=[]
    for y in starts(height,ch):
        for x in starts(width,cw):
            view=truth[y:y+ch:8,x:x+cw:8];ids,count=np.unique(view,return_counts=True)
            good=ids>0;count=count[good];ids=ids[good]
            if len(ids)<2 or count.sum()/view.size<.08:continue
            p=count/count.sum();entropy=-float(np.sum(p*np.log(p)))
            ranked.append((entropy,float(count.sum()/view.size),-y,-x,y,x,ch,cw))
    return sorted(ranked,reverse=True)


def main():
    p=argparse.ArgumentParser();p.add_argument("--data-root",type=Path,required=True);p.add_argument("--root",type=Path,required=True)
    p.add_argument("--max-side",type=int,default=1280);p.add_argument("--previous",type=Path);p.add_argument("--seed",type=int,default=7730);p.add_argument("--quotas",nargs="+");a=p.parse_args();a.root.mkdir(parents=True,exist_ok=True)
    quotas=QUOTAS if a.quotas is None else {k:int(v) for k,v in (item.split("=") for item in a.quotas)}
    if (a.root/"PLAN.json").exists():raise FileExistsError("Use a new output directory; do not silently replace an evaluated corpus")
    rng=np.random.default_rng(a.seed);samples=[];excluded=[]
    if a.previous:
        previous=json.loads((a.previous/"PLAN.json").read_text())
        for old in previous["samples"]:
            row={**old,"benchmark_cohort":"previously_inspected"};folder=a.root/"cases"/row["case"];folder.mkdir(parents=True)
            for name in ("original.png","original.native.png","coarse.tif","reference.native.tif"):
                shutil.copy2(a.previous/"cases"/row["case"]/name,folder/name)
            save_json(folder/"sample.json",row);samples.append(row)
    for dataset,quota in quotas.items():
        manifests=[dataset+".jsonl"] if dataset!="camelyon16-subset" else ["camelyon16-subset.jsonl","camelyon16-extension.jsonl"]
        candidates=[]
        for name in manifests:candidates.extend(json.loads(line) for line in (a.data_root/"prepared"/name).read_text().splitlines() if line)
        if dataset=="tiger":candidates=[r for r in candidates if r.get("cohort")=="non-TCGA" and group_key(r)!="114"]
        candidates=sorted(candidates,key=lambda r:r["id"]);rng.shuffle(candidates)
        prior=[r for r in samples if r["dataset"]==dataset];used={r["group"] for r in prior};accepted=len(prior)
        if accepted>=quota:continue
        for row in candidates:
            group=group_key(row)
            if group in used:continue
            image_path=a.data_root/row["image"];mask_path=a.data_root/row["mask"]
            spacing,resolution=mpp(row,a.data_root)
            if any(not (.1<v<2) for v in spacing):raise ValueError("Implausible spacing")
            native_truth=np.array(Image.open(mask_path))
            if native_truth.ndim!=2:raise ValueError("Categorical native mask required")
            ignore=set(row["ignore_labels"]);mapped=np.where(np.isin(native_truth,list(ignore)),0,native_truth).astype(np.uint8)
            chosen=None
            for _,_,_,_,y,x,ch,cw in candidate_windows(mapped,spacing,a.max_side):
                cropped=mapped[y:y+ch,x:x+cw];shape=[round(ch*spacing[1]/.5),round(cw*spacing[0]/.5)]
                work_truth=cv2.resize(cropped,(shape[1],shape[0]),interpolation=cv2.INTER_NEAREST)
                coarse=coarse_mask(work_truth)
                dummy=np.zeros((int(np.ceil(shape[0]/14)),int(np.ceil(shape[1]/14)),1),np.float32)
                try:_,labels,_,_=coarse_tile_observations(dummy,coarse)
                except ValueError:continue
                if len(np.unique(labels))<2 or min(Counter(labels).values())<1:continue
                chosen=(y,x,ch,cw,shape,cropped,coarse);break
            if chosen is None:
                excluded.append(dict(dataset=dataset,id=row["id"],group=group,reason="No field >=896 working pixels with at least two coarse training classes"));continue
            if digest(image_path)!=row["image_sha256"] or digest(mask_path)!=row["mask_sha256"]:raise ValueError("Source hash mismatch")
            y,x,ch,cw,shape,cropped,coarse=chosen
            raw_codes=sorted(int(g) for g in np.unique(cropped) if g>0)
            if not set(raw_codes)<=set(LABELS[dataset]):raise ValueError("Unreconciled class code")
            mapping={raw:i+1 for i,raw in enumerate(raw_codes)}
            fine=np.zeros(cropped.shape,np.uint8);compact=np.zeros(coarse.shape,np.uint8)
            for raw,g in mapping.items():fine[cropped==raw]=g;compact[coarse==raw]=g
            rgb=np.array(Image.open(image_path).convert("RGB"))[y:y+ch,x:x+cw]
            working=cv2.resize(rgb,(shape[1],shape[0]),interpolation=cv2.INTER_AREA if min(spacing)<.5 else cv2.INTER_LINEAR)
            accepted+=1;case=f"{dataset}_{accepted:02d}";folder=a.root/"cases"/case;folder.mkdir(parents=True)
            Image.fromarray(rgb).save(folder/"original.native.png");Image.fromarray(working).save(folder/"original.png")
            tifffile.imwrite(folder/"reference.native.tif",fine,compression="deflate")
            tifffile.imwrite(folder/"coarse.tif",compact,compression="deflate")
            metadata=dict(case=case,dataset=dataset,group=group,source=row,benchmark_cohort="new_extension" if a.previous else "initial",native_crop_xywh=[x,y,cw,ch],
                source_region_type="ROI" if dataset in ["tiger","bcss"] else "native ROI cropped from WSI",
                native_mpp_xy=spacing,working_mpp_xy=[cw*spacing[0]/shape[1],ch*spacing[1]/shape[0]],
                target_mpp=.5,working_shape_yx=shape,native_shape_yx=[ch,cw],fov_um_xy=[cw*spacing[0],ch*spacing[1]],
                resolution=resolution,optical_resolution_um=None,objective_magnification=None,
                raw_to_compact=mapping,class_names={mapping[g]:LABELS[dataset][g] for g in raw_codes},palette=palette(dataset,raw_codes),
                raw_ignore_codes=sorted(ignore),coarse_reference_derived=True,coarse_tile_px=90,coarse_tile_um=45,
                original_sha256=digest(folder/"original.png"),coarse_sha256=digest(folder/"coarse.tif"),
                reference_native_sha256=digest(folder/"reference.native.tif"))
            save_json(folder/"sample.json",metadata);samples.append(metadata);used.add(group)
            print(json.dumps(dict(prepared=case,group=group,working_shape=shape,classes=len(raw_codes))),flush=True)
            if accepted==quota:break
    save_json(a.root/"PLAN.json",dict(seed=a.seed,requested_quotas=quotas,previous_plan_sha256=digest(a.previous/"PLAN.json") if a.previous else None,samples=samples,exclusions=excluded,
        methods=[9,10,15,19],field_selection="maximum annotated-class entropy among deterministic overlapping windows; no algorithm-score selection",
        max_working_side=a.max_side,min_working_side=896,target_mpp=.5,coarse_tile_um=45,
        source_case114_excluded=True,tiger_bcss_overlap_avoidance="TIGER uses non-TCGA only; BCSS uses TCGA cases",
        exclusions_by_dataset={"sicapv2":"JPEG mask semantics unreconciled","cocahis-vista":"foreground semantics and pixel spacing unreconciled",
            "wsss4luad":"distributed patch pixel spacing unverified","rings":"pixel spacing unverified"},
        model_selection="per-image coarse-tile labels only; fixed presets; fine references unavailable to prediction module",
        annotation_caveat="CAMELYON tumor is expert-derived; its non-tumor tissue is automatically detected, not expert stroma",
        implementation_sha256=digest(__file__)))

if __name__=="__main__":main()
