"""Decode identical spatial-token grids before and after bounded class edits.

No fine annotation is opened here. All RGB and Bora outputs are hashed before
the separate evaluation program reads that annotation.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import tifffile
from .model import Decoder
from experiments.virchow2_inverse.common import save_json,digest,coarse_tile_observations,pad_image
from experiments.virchow2_inverse.protocol import bora_rgb
from bora.boundary import BoundaryConfig


GAINS={"full":1.,"selective":1.25,"suppression":.75,"stronger":1.5}


def contrast_basis(features,coarse,selected,mean,scale):
    tile_values,labels,_,records=coarse_tile_observations(features,coarse)
    z=(tile_values[:,selected]-mean[selected])/scale[selected]
    means=np.stack([z[labels==group].mean(0) for group in range(1,6)])
    centre=means.mean(0)
    _,singular,vh=np.linalg.svd(means-centre,full_matrices=False)
    rank=int(np.sum(singular>singular[0]*1e-6))
    return centre.astype(np.float32),vh[:rank].T.astype(np.float32),dict(
        rank=rank,singular_values=singular.tolist(),selected_dimensions=selected.tolist(),
        class_tile_counts={str(g):int(np.sum(labels==g)) for g in range(1,6)},coarse_observations=len(labels),
        rule="equal-class mean contrasts, SVD; selected64 coordinates only; all other coordinates retained",
        selected_pool_source="previous experiment's balanced-MRMR coarse-only pool",coarse_tile_records=records)


def edit_tokens(tokens,mean,scale,selected,centre,basis,gain):
    """Bound each edit to 0.5 training SD and 5% of the whole token's norm."""
    z=(tokens.astype(np.float32)-mean)/scale
    part=z[...,selected]
    delta=(gain-1)*((part-centre)@basis)@basis.T
    delta=np.clip(delta,-.5,.5)
    norm=np.linalg.norm(z,axis=-1,keepdims=True)
    factor=np.minimum(1,.05*np.maximum(norm,1e-6)/np.maximum(np.linalg.norm(delta,axis=-1,keepdims=True),1e-12))
    delta*=factor
    changed=tokens.astype(np.float32).copy()
    changed[...,selected]+=delta*scale[selected]
    stats=dict(mean_abs_edit_training_sd=float(np.mean(abs(delta))),max_abs_edit_training_sd=float(abs(delta).max()),
        max_relative_token_norm_edit=float(np.max(np.linalg.norm(delta,axis=-1)/np.maximum(norm[...,0],1e-6))),
        mean_relative_token_norm_edit=float(np.mean(np.linalg.norm(delta,axis=-1)/np.maximum(norm[...,0],1e-6))))
    return changed,stats


def focal_window(coords,coarse):
    candidates=[]
    for index,(y,x) in enumerate(coords):
        if y+224>coarse.shape[0] or x+224>coarse.shape[1]:continue
        ids,count=np.unique(coarse[y:y+224,x:x+224],return_counts=True)
        keep=ids>0;count=count[keep];ids=ids[keep]
        if len(ids)<3:continue
        p=count/count.sum();entropy=-float(np.sum(p*np.log(p)))
        candidates.append((entropy,-index,index,y,x,ids.tolist()))
    if not candidates:raise ValueError("No complete native tile contains at least three coarse classes")
    _,_,index,y,x,classes=max(candidates)
    return dict(index=index,y=y,x=x,size_px=[224,224],mpp_um=.5,classes=classes,
                selection="maximum coarse-class entropy among nonpadded windows with >=3 classes; first index breaks ties")


def main():
    p=argparse.ArgumentParser();p.add_argument("--training",type=Path,required=True)
    p.add_argument("--source",type=Path,required=True);p.add_argument("--previous",type=Path,required=True)
    p.add_argument("--root",type=Path,required=True);a=p.parse_args();a.root.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.4)
    checkpoint=torch.load(a.training/"decoder.pt",map_location="cpu",weights_only=True)
    state=checkpoint["state"];mean=state["mean"].numpy().ravel();scale=state["scale"].numpy().ravel()
    decoder=Decoder(mean,scale);decoder.load_state_dict(state);decoder.cuda().eval().requires_grad_(False)
    image=np.array(Image.open(a.source/"image.original.png").convert("RGB"));coarse=tifffile.imread(a.source/"tile_90/coarse.tif")
    info=json.loads((a.previous/"features/inference.json").read_text())
    if digest(a.source/"image.original.png")!=info["image_sha256"]:raise ValueError("Wrong evaluation image")
    features=np.load(a.previous/"features/dense_patch_features.npy",mmap_mode="r")
    selected=np.array(json.loads((a.previous/"initial_feature_selection.json").read_text())["indices"])
    centre,basis,basis_info=contrast_basis(features,coarse,selected,mean,scale)
    np.savez(a.root/"contrast_basis.npz",centre=centre,basis=basis,selected=selected,mean=mean,scale=scale)
    coords=info["window_coordinates_yx"];focal=focal_window(coords,coarse)
    config=BoundaryConfig(**json.loads((a.previous/"protocol.json").read_text())["boundary_config"])
    protocol=dict(input_type="TIGER breast ROI, not WSI",native_shape_yx=list(image.shape[:2]),native_mpp_xy=[.5,.5],
        optical_resolution_um=None,objective_magnification=None,window_px=224,window_fov_um=112,token_px=14,token_fov_um=7,
        input="256 spatial tokens x1280, CLS/register tokens excluded",encoder=info["repository"],encoder_revision=info["revision"],
        encoder_sha256=info["checkpoint_sha256"],decoder_sha256=digest(a.training/"decoder.pt"),focal=focal,gains=GAINS,
        edit_rule="z'=z+clip((gain-1)*(z_selected-balanced_class_mean)Q Q^T, +/-0.5 training SD), then norm-cap to 5% per token",
        basis=basis_info,source_rgb_skip=False,same_decoder_for_all=True,decoder_frozen=True,
        fine_reference_used=False,coarse_reference_derived=True,coarse_tile_px=90,bora_config=vars(config),
        fidelity_gate=dict(full_rgb_ssim_min=.8,full_direct_tile_rgb_ssim_min=.8,full_nuclei_proxy_f1_min=.8,full_nuclei_median_displacement_um_max=2,
            selective_nuclei_f1_drop_max=.02,selective_macro_iou_improvement_min=0),
        gate_scope="exploratory screen, not clinical validation; no threshold tuning after fine-mask scoring",
        code_sha256={f.name:digest(f) for f in Path(__file__).parent.glob("*.py")})
    save_json(a.root/"PROTOCOL.json",protocol)
    Image.fromarray(image).save(a.root/"original.png");tifffile.imwrite(a.root/"coarse.tif",coarse)
    y,x=focal["y"],focal["x"];Image.fromarray(image[y:y+224,x:x+224]).save(a.root/"tile.original.png")
    raw=np.load(a.previous/"features/window_patch_tokens.npy",mmap_mode="r")
    padded=pad_image(image);axis=np.maximum(np.hanning(224),.05);blend=axis[:,None]*axis[None,:]
    hashes={};summary={}
    with torch.inference_mode():
        for name,gain in GAINS.items():
            accum=np.zeros_like(padded,dtype=np.float32);mass=np.zeros(padded.shape[:2],np.float32);edit_stats=[]
            for start in range(0,len(coords),16):
                original=np.array(raw[start:start+16],dtype=np.float32)
                values,stat=edit_tokens(original,mean,scale,selected,centre,basis,gain);edit_stats.append(stat)
                batch=torch.from_numpy(values.transpose(0,3,1,2).copy()).cuda()
                with torch.autocast("cuda",dtype=torch.float16):decoded=decoder(batch)
                rgb=decoded.float().permute(0,2,3,1).cpu().numpy()
                for j,(yy,xx) in enumerate(coords[start:start+16]):
                    accum[yy:yy+224,xx:xx+224]+=rgb[j]*blend[...,None];mass[yy:yy+224,xx:xx+224]+=blend
                    if start+j==focal["index"]:Image.fromarray(np.rint(rgb[j]*255).clip(0,255).astype(np.uint8)).save(a.root/f"tile.{name}.png")
            result=np.rint(accum/mass[...,None]*255).clip(0,255).astype(np.uint8)[:image.shape[0],:image.shape[1]]
            Image.fromarray(result).save(a.root/f"{name}.png")
            prediction=bora_rgb(result,coarse,config);tifffile.imwrite(a.root/f"bora.{name}.tif",prediction.astype(np.uint8),compression="deflate")
            for filename in [f"tile.{name}.png",f"{name}.png",f"bora.{name}.tif"]:hashes[filename]=digest(a.root/filename)
            summary[name]=dict(gain=gain,edit_batches=edit_stats)
            print(json.dumps(dict(reconstruction=name,complete=True)),flush=True)
    prediction=bora_rgb(image,coarse,config);tifffile.imwrite(a.root/"bora.original.tif",prediction.astype(np.uint8),compression="deflate")
    hashes["bora.original.tif"]=digest(a.root/"bora.original.tif")
    # Decode all test patches with the full-feature control only. Never select
    # model, editing directions or gains on test pixels.
    rows=json.loads((a.training/"corpus.json").read_text())["patches"]
    ids=np.array([i for i,r in enumerate(rows) if r["split"]=="test"])
    cached=np.load(a.training/"tokens.npy",mmap_mode="r")
    out=np.lib.format.open_memmap(a.root/"test_reconstructions.npy",mode="w+",shape=(len(ids),224,224,3),dtype="uint8")
    with torch.inference_mode():
        for start in range(0,len(ids),16):
            batch=torch.from_numpy(cached[ids[start:start+16]].copy()).cuda().float()
            with torch.autocast("cuda",dtype=torch.float16):prediction=decoder(batch)
            out[start:start+len(batch)]=np.rint(prediction.float().permute(0,2,3,1).cpu().numpy()*255).clip(0,255).astype(np.uint8)
    out.flush();np.save(a.root/"test_ids.npy",ids)
    save_json(a.root/"PREDICTIONS_FROZEN.json",dict(files_sha256=hashes,edits=summary,
        timestamp_unix=time.time(),fine_reference_opened=False,decoder_sha256=protocol["decoder_sha256"],
        test_reconstructions_sha256=digest(a.root/"test_reconstructions.npy")))

if __name__=="__main__":main()
