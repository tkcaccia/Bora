"""Run frozen feature inference and seven Bora inputs without reading references."""
import argparse,gc,json,time,shutil
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import tifffile
import torch
from threadpoolctl import threadpool_limits
from experiments.virchow2_inverse.common import digest,save_json,pad_image,starts,MODEL_SHA256,MODEL_REVISION
from experiments.virchow2_inverse.gpu_ideas import load_model,normalize
from experiments.virchow2_inverse.protocol import bora_rgb
from .methods import Context,METHODS,run_method,align_variant


def encode(model,image,selected=None):
    p=pad_image(image);coords=[(y,x) for y in starts(p.shape[0]) for x in starts(p.shape[1])]
    channels=1280 if selected is None else len(selected)
    accum=np.zeros((p.shape[0]//14,p.shape[1]//14,channels),np.float32);mass=np.zeros(accum.shape[:2],np.float32)
    axis=np.maximum(np.hanning(16),.05);blend=axis[:,None]*axis[None,:]
    with torch.inference_mode():
        for start in range(0,len(coords),4):
            batch=np.stack([p[y:y+224,x:x+224].transpose(2,0,1) for y,x in coords[start:start+4]])
            x=torch.from_numpy(batch.copy()).cuda().float()/255
            with torch.autocast("cuda",dtype=torch.float16):values=model(normalize(x))[:,5:]
            if selected is not None:values=values[:,:,selected]
            values=values.float().cpu().numpy().reshape(-1,16,16,channels)
            for (y,x),v in zip(coords[start:start+4],values):
                accum[y//14:y//14+16,x//14:x//14+16]+=v*blend[...,None];mass[y//14:y//14+16,x//14:x//14+16]+=blend
    result=(accum/mass[...,None]).astype(np.float16)
    if not np.isfinite(result).all():raise FloatingPointError("Nonfinite embeddings")
    return result,dict(window_coordinates_yx=coords,window_px=224,stride_px=112,padded_shape_yx=list(p.shape[:2]),
        spatial_tokens="5:261; CLS and registers excluded",patch_px=14,channels=channels,selected_dimensions=None if selected is None else selected.tolist())


def freeze_case(folder,sample,model,refresh=False):
    previous=None
    if (folder/"FROZEN.json").exists():
        record=json.loads((folder/"FROZEN.json").read_text())
        if not all(digest(folder/f)==sha for f,sha in record["files_sha256"].items()):raise ValueError("Previously frozen output changed")
        if not refresh:return
        if record.get("context_head_revision")==2:return
        previous=record;archive=folder/"diagnostics/rounded_context_head";archive.mkdir(parents=True,exist_ok=True)
        shutil.copy2(folder/"FROZEN.json",archive/"FROZEN.json")
        for name in record["files_sha256"]:
            if name.startswith(("features/","10/","15/")):
                dest=archive/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(folder/name,dest)
        if (folder/"scores.json").exists():shutil.copy2(folder/"scores.json",archive/"scores.json")
    started=time.monotonic();image=np.array(Image.open(folder/"original.png").convert("RGB"));coarse=tifffile.imread(folder/"coarse.tif")
    if digest(folder/"original.png")!=sample["original_sha256"] or digest(folder/"coarse.tif")!=sample["coarse_sha256"]:raise ValueError("Input changed")
    featdir=folder/"features";featdir.mkdir(exist_ok=True)
    if not (featdir/"READY.json").exists():
        f,meta=encode(model,image);np.save(featdir/"base.npy",f)
        save_json(featdir/"READY.json",dict(**meta,sha256=digest(featdir/"base.npy"),image_sha256=sample["original_sha256"]))
    f=np.load(featdir/"base.npy",mmap_mode="r")
    c=Context(image,coarse,f,sample["palette"])
    save_json(folder/"selection.json",c.selection)
    variants={};records=[]
    for name in ["scale_1","scale_2"]:
        path=featdir/(name+".npy");meta_path=featdir/(name+".json")
        factor=2 if name=="scale_1" else 4
        out=np.array(Image.fromarray(image).resize((round(image.shape[1]/factor),round(image.shape[0]/factor)),Image.Resampling.LANCZOS))
        if not path.exists() or not meta_path.exists():
            values,meta=encode(model,out,c.selected);np.save(path,values)
            save_json(meta_path,dict(**meta,shape_yx=list(out.shape[:2]),mpp_xy=[sample["fov_um_xy"][0]/out.shape[1],sample["fov_um_xy"][1]/out.shape[0]],sha256=digest(path)))
        values=np.load(path).astype(np.float32)
        variants[name]=align_variant(values,out.shape[:2],c)
        records.append(dict(name=name,**json.loads(meta_path.read_text())))
    save_json(folder/"inference.json",dict(encoder_revision=MODEL_REVISION,encoder_sha256=MODEL_SHA256,base=json.loads((featdir/"READY.json").read_text()),
        variants=records,model_input_mpp_xy=sample["working_mpp_xy"],nominal_window_fov_um=112,all_weights_frozen=True,fine_reference_read=False))
    files=dict(previous["files_sha256"]) if previous else {};methods=[r for r in previous["methods"] if r["id"] not in (10,15)] if previous else []
    def store(name,rgb,raw=None,details=None,extras=None):
        target=folder/name;target.mkdir(exist_ok=True)
        Image.fromarray(rgb).save(target/"representation.png")
        pred=bora_rgb(rgb,coarse).astype(np.uint8)
        tifffile.imwrite(target/"bora.working.tif",pred,compression="deflate")
        native=cv2.resize(pred,(sample["native_shape_yx"][1],sample["native_shape_yx"][0]),interpolation=cv2.INTER_NEAREST)
        tifffile.imwrite(target/"bora.native.tif",native,compression="deflate")
        if raw is not None:
            tifffile.imwrite(target/"before.working.tif",raw,compression="deflate")
            native_raw=cv2.resize(raw,(sample["native_shape_yx"][1],sample["native_shape_yx"][0]),interpolation=cv2.INTER_NEAREST)
            tifffile.imwrite(target/"before.native.tif",native_raw,compression="deflate")
        for filename in ["representation.png","bora.working.tif","bora.native.tif"]+(["before.working.tif","before.native.tif"] if raw is not None else []):files[f"{name}/{filename}"]=digest(target/filename)
        for key,array in (extras or {}).items():
            if array.dtype==np.uint8 and array.ndim==2:Image.fromarray(array).save(target/(key+".png"))
            else:np.save(target/(key+".npy"),array)
        save_json(target/"details.json",details or {})
    if not refresh:store("he",image,details=dict(control="original H&E resampled to common physical sampling, same Bora settings"))
    feature_rgb=c.probability_rgb(c.native_p);feature_raw=c.classes[c.native_p.argmax(-1)].astype(np.uint8);feature_raw[coarse==0]=0
    store("features",feature_rgb,feature_raw,dict(control="plain selected-feature probability RGB + Bora"))
    for number,title in METHODS.items():
        if refresh and number not in (10,15):continue
        method_start=time.monotonic();rgb,raw,details,extras=run_method(number,c,variants)
        store(f"{number:02d}",rgb,raw,details,extras)
        methods.append(dict(id=number,title=title,seconds=time.monotonic()-method_start))
        print(json.dumps(dict(case=sample["case"],method=number,complete=True,seconds=methods[-1]["seconds"])),flush=True)
    save_json(folder/"FROZEN.json",dict(files_sha256=files,methods=methods,seconds=time.monotonic()-started,
        context_head_revision=2,correction="normalize original float64 tile means before fitting initial head, matching the earlier implementation",
        fine_reference_read=False,coarse_observations=len(c.labels),coarse_classes=c.classes.tolist(),
        source_sha256={p.name:digest(p) for p in Path(__file__).parent.glob("*.py")},timestamp_unix=time.time()))


def main():
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);p.add_argument("--model-dir",type=Path,required=True)
    p.add_argument("--cases",nargs="*");p.add_argument("--refresh-context-head",action="store_true");a=p.parse_args();torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.4)
    plan=json.loads((a.root/"PLAN.json").read_text());model=None if a.refresh_context_head else load_model(a.model_dir)
    with threadpool_limits(limits=2):
        for i,sample in enumerate(plan["samples"]):
            if a.cases and sample["case"] not in a.cases:continue
            freeze_case(a.root/"cases"/sample["case"],sample,model,a.refresh_context_head);gc.collect()
            print(json.dumps(dict(cases_complete=i+1,total=len(plan["samples"]),case=sample["case"])),flush=True)
    if all((a.root/"cases"/r["case"]/"FROZEN.json").exists() for r in plan["samples"]):
        save_json(a.root/"ALL_PREDICTIONS_FROZEN.json",dict(cases=len(plan["samples"]),methods=list(METHODS),
            case_manifest_sha256={r["case"]:digest(a.root/"cases"/r["case"]/"FROZEN.json") for r in plan["samples"]},timestamp_unix=time.time(),fine_references_read_by_predictor=False))

if __name__=="__main__":main()
