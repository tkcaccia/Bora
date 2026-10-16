"""Case-disjoint TIGER RGB/token pairs at the provider's 0.5 um/pixel."""
import argparse, json, re, time
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from experiments.virchow2_inverse.common import digest, save_json, MODEL_SHA256
from experiments.virchow2_inverse.gpu_ideas import load_model, normalize


def case_key(filename):
    name = Path(filename).stem.split("_[")[0]
    if name.startswith("TCGA-"):
        return name[:12]
    if name.startswith("TC_"):
        return re.match(r"TC_S\d+_P\d+", name).group()
    match = re.match(r"\d+", name)
    if not match:
        raise ValueError(f"Unrecognized case identifier: {name}")
    return match.group()


def corpus_plan(data, seed=711):
    groups = defaultdict(list)
    for path in sorted(data.glob("*/images/*.png")):
        with Image.open(path) as im:
            width, height = im.size
        if min(width, height) >= 224 and case_key(path.name) != "114":
            groups[case_key(path.name)].append((path, width, height))
    rng = np.random.default_rng(seed)
    assignments = {}
    for cohort in ("TCGA", "other"):
        keys = sorted(k for k in groups if k.startswith("TCGA-") == (cohort == "TCGA"))
        rng.shuffle(keys)
        held = 15 if cohort == "TCGA" else 6
        for i, key in enumerate(keys):
            assignments[key] = "test" if i < held else "validation" if i < 2*held else "train"
    rows = []
    for key in sorted(groups):
        split = assignments[key]
        count = (16 if key.startswith("TCGA-") else 64) if split == "train" else (8 if key.startswith("TCGA-") else 16)
        used = set()
        for index in range(count):
            for attempt in range(200):
                path, width, height = groups[key][int(rng.integers(len(groups[key])))]
                y, x = int(rng.integers(height-223)), int(rng.integers(width-223))
                if (str(path), y, x) not in used:
                    break
            else:
                raise RuntimeError("Could not draw unique patch")
            used.add((str(path), y, x))
            rows.append(dict(case=key, split=split, image=str(path.relative_to(data)), y=y, x=x,
                             native_mpp_xy=[.5, .5], size_px=[224, 224]))
    return rows, assignments


def main():
    p=argparse.ArgumentParser();p.add_argument("--data",type=Path,required=True)
    p.add_argument("--root",type=Path,required=True);p.add_argument("--model-dir",type=Path,required=True)
    a=p.parse_args();a.root.mkdir(parents=True,exist_ok=True)
    if (a.root/"CACHE_READY.json").exists():
        print("Cache already complete",flush=True);return
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.4)
    rows, assignments=corpus_plan(a.data)
    save_json(a.root/"corpus.json",dict(seed=711, patches=rows, case_assignments=assignments,
        excluded_case_keys=["114"], case_identifiers_are_patient_proxies=True,
        source="https://zenodo.org/records/6014422", native_mpp_xy=[.5,.5],
        image_type="native ROI crops, not WSI; no resampling", fine_masks_used=False))
    n=len(rows)
    tokens=np.lib.format.open_memmap(a.root/"tokens.npy",mode="w+",dtype="float16",shape=(n,1280,16,16))
    rgb=np.lib.format.open_memmap(a.root/"rgb.npy",mode="w+",dtype="uint8",shape=(n,3,224,224))
    model=load_model(a.model_dir);started=time.monotonic();sources={}
    image_cache={}
    with torch.inference_mode():
        for start in range(0,n,4):
            batch=[]
            for row in rows[start:start+4]:
                filename=row["image"]
                if filename not in image_cache:
                    if len(image_cache)>6:image_cache.clear()
                    image_cache[filename]=np.array(Image.open(a.data/filename).convert("RGB"))
                patch=image_cache[filename][row["y"]:row["y"]+224,row["x"]:row["x"]+224]
                batch.append(patch.transpose(2,0,1))
                if filename not in sources:sources[filename]=digest(a.data/filename)
            batch=np.stack(batch).copy();rgb[start:start+len(batch)]=batch
            x=torch.from_numpy(batch).cuda().float()/255
            with torch.autocast("cuda",dtype=torch.float16):features=model(normalize(x))[:,5:]
            if features.shape[1:] != (256,1280):raise ValueError("Incorrect token indexing")
            tokens[start:start+len(batch)]=features.transpose(1,2).reshape(-1,1280,16,16).cpu().numpy()
            if start%128==0:print(json.dumps(dict(extracted=start+len(batch),total=n,seconds=time.monotonic()-started)),flush=True)
    tokens.flush();rgb.flush()
    save_json(a.root/"CACHE_READY.json",dict(patches=n,source_sha256=sources,encoder_sha256=MODEL_SHA256,
        tokens_sha256=digest(a.root/"tokens.npy"),rgb_sha256=digest(a.root/"rgb.npy"),
        seconds=time.monotonic()-started,spatial_tokens_only=True,implementation_sha256=digest(__file__)))

if __name__=="__main__":main()
