"""Frozen-encoder RGB decoding; checkpoint selection uses other validation cases."""
import argparse,json,time,random
from pathlib import Path
import numpy as np
import torch
from .model import Decoder,reconstruction_loss
from experiments.virchow2_inverse.common import save_json,digest


def main():
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True)
    p.add_argument("--epochs",type=int,default=100);p.add_argument("--batch",type=int,default=32)
    a=p.parse_args();torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.4)
    torch.manual_seed(711);np.random.seed(711);random.seed(711)
    torch.backends.cudnn.benchmark=False
    rows=json.loads((a.root/"corpus.json").read_text())["patches"]
    splits={s:np.array([i for i,r in enumerate(rows) if r["split"]==s]) for s in ("train","validation","test")}
    if any(set(rows[i]["case"] for i in splits[s])&set(rows[i]["case"] for i in splits[t])
           for s,t in (("train","validation"),("train","test"),("validation","test"))):raise ValueError("Case leakage")
    if any(r["case"]=="114" for r in rows):raise ValueError("Evaluation source included")
    features=np.load(a.root/"tokens.npy",mmap_mode="r");images=np.load(a.root/"rgb.npy",mmap_mode="r")
    sums=np.zeros(1280);squares=np.zeros(1280);count=0
    for start in range(0,len(splits["train"]),32):
        x=features[splits["train"][start:start+32]].astype(np.float64)
        sums+=x.sum((0,2,3));squares+=(x*x).sum((0,2,3));count+=x.shape[0]*256
    mean=(sums/count).astype(np.float32);scale=np.sqrt(np.maximum(squares/count-mean.astype(float)**2,.05**2)).astype(np.float32)
    model=Decoder(mean,scale).cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=.0004,weight_decay=.0001)
    scaler=torch.amp.GradScaler("cuda");best=float("inf");history=[];rng=np.random.default_rng(711);started=time.monotonic()
    save_json(a.root/"TRAIN_PROTOCOL.json",dict(epochs=a.epochs,batch=a.batch,seed=711,optimizer="AdamW",lr=.0004,
        loss="RGB L1 + 0.2 native-gradient L1 + 0.15 (1-local SSIM)",
        parameters=sum(p.numel() for p in model.parameters()),normalization="training tokens only; channel std floor 0.05",
        encoder_frozen=True,input="all 1280 final spatial channels; 16x16",decoder="1x1 projection to256,3 residual blocks,1x1 to588,PixelShuffle14,sigmoid",
        source_rgb_skip=False,evaluation_case_excluded="114",selection="lowest validation loss; no target-ROI or test-set selection",
        stopping="100 epochs maximum; after epoch30, stop after20 epochs without improvement",
        split_patch_counts={k:len(v) for k,v in splits.items()},model_source_sha256=digest(Path(__file__).with_name("model.py")),train_source_sha256=digest(__file__)))
    best_epoch=0
    for epoch in range(a.epochs):
        lr=.0004*(.1+.9*(1+np.cos(np.pi*epoch/a.epochs))/2)
        for pg in optimizer.param_groups:pg["lr"]=lr
        model.train();order=rng.permutation(splits["train"]);train_total=0
        for start in range(0,len(order),a.batch):
            ids=order[start:start+a.batch]
            x=torch.from_numpy(features[ids].copy()).cuda().float();target=torch.from_numpy(images[ids].copy()).cuda().float()/255
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.float16):prediction=model(x)
            loss,_=reconstruction_loss(prediction.float(),target)
            if not torch.isfinite(loss):raise FloatingPointError("Invalid decoder loss")
            scaler.scale(loss).backward();scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),5)
            scaler.step(optimizer);scaler.update();train_total+=float(loss.detach())*len(ids)
        model.eval();val_total=0;val_parts=np.zeros(3)
        with torch.inference_mode():
            for start in range(0,len(splits["validation"]),a.batch):
                ids=splits["validation"][start:start+a.batch]
                x=torch.from_numpy(features[ids].copy()).cuda().float();target=torch.from_numpy(images[ids].copy()).cuda().float()/255
                with torch.autocast("cuda",dtype=torch.float16):prediction=model(x)
                loss,parts=reconstruction_loss(prediction.float(),target)
                val_total+=float(loss)*len(ids);val_parts+=np.array([float(v) for v in parts])*len(ids)
        validation=val_total/len(splits["validation"])
        if validation<best:
            best=validation;best_epoch=epoch+1
            torch.save(dict(state=model.cpu().state_dict(),epoch=epoch+1,validation_loss=best),a.root/"decoder.pt");model.cuda()
        record=dict(epoch=epoch+1,train_loss=train_total/len(order),validation_loss=validation,
                    validation_l1=val_parts[0]/len(splits["validation"]),validation_ssim_training=val_parts[2]/len(splits["validation"]),
                    best_epoch=best_epoch,seconds=time.monotonic()-started,lr=lr)
        history.append(record);save_json(a.root/"training_history.json",history);print(json.dumps(record),flush=True)
        if epoch+1>=30 and epoch+1-best_epoch>=20:break
    save_json(a.root/"TRAIN_COMPLETE.json",dict(best_epoch=best_epoch,best_validation_loss=best,epochs=len(history),
        checkpoint_sha256=digest(a.root/"decoder.pt"),seconds=time.monotonic()-started,
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated()))

if __name__=="__main__":main()
