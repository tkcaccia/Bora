"""Verify provenance and frozen outputs, and retain the executed source files."""
import argparse,json,platform,shutil
from pathlib import Path
from importlib.metadata import version
import numpy as np
import tifffile
from PIL import Image
from experiments.virchow2_inverse.common import digest,save_json


def main():
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True)
    p.add_argument("--training",type=Path,required=True);a=p.parse_args()
    corpus=json.loads((a.training/"corpus.json").read_text());checks={}
    split={s:set(r["case"] for r in corpus["patches"] if r["split"]==s) for s in ("train","validation","test")}
    checks["case_splits_disjoint"]=not any(split[x]&split[y] for x,y in [("train","validation"),("train","test"),("validation","test")])
    checks["all114_cases_excluded"]=all("114" not in ids for ids in split.values())
    train=json.loads((a.training/"TRAIN_COMPLETE.json").read_text())
    protocol=json.loads((a.root/"PROTOCOL.json").read_text())
    checks["decoder_hash_matches_training_and_inference"]=(digest(a.training/"decoder.pt")==train["checkpoint_sha256"]==protocol["decoder_sha256"])
    for manifest in ("PREDICTIONS_FROZEN.json","CONSISTENCY_FROZEN.json"):
        files=json.loads((a.root/manifest).read_text())["files_sha256"]
        checks[manifest]=all(digest(a.root/name)==sha for name,sha in files.items())
    shape=tuple(protocol["native_shape_yx"])
    checks["native_image_geometry"]=all(np.array(Image.open(a.root/f"{name}.png")).shape==(*shape,3)
        for name in ("original","full","selective","suppression","stronger"))
    checks["native_mask_geometry_and_labels"]=all((lambda x:x.shape==shape and set(np.unique(x))<=set(range(6)))(tifffile.imread(path)) for path in a.root.glob("bora.*.tif"))
    checks["fp32_gradient_audit"]=json.loads((a.root/"GRADIENT_AUDIT.json").read_text())["passed"]
    previous=a.root.parent/"virchow2-inverse-2026-10-16"
    earlier=[]
    for index in range(1,21):
        folder=previous/f"{index:02d}";record=json.loads((folder/"result.json").read_text())
        earlier.append(digest(folder/"bora.tif")==record["prediction_sha256"] and
                       digest(folder/"representation.png")==record["image_sha256"])
    checks["all20_earlier_images_and_predictions_match_recorded_hashes"]=all(earlier)
    executed=a.root/"runtime_sources";executed.mkdir(exist_ok=True)
    for f in Path(__file__).parent.glob("*.py"):shutil.copy2(f,executed/f.name)
    # Shared encoder, sampling, Bora wrapper and scoring helpers are also used.
    helpers=Path(__file__).parent.parent/"virchow2_inverse"
    for name in ("gpu_ideas.py","common.py","protocol.py","finalize.py"):
        shutil.copy2(helpers/name,executed/("shared_"+name))
    report=dict(checks=checks,passed=all(checks.values()),python=platform.python_version(),
        packages={name:version(name) for name in ("torch","torchvision","timm","numpy","scipy","scikit-image","safetensors","Pillow","tifffile")},
        executed_sources_sha256={f.name:digest(f) for f in executed.glob("*.py")},
        training_code_sha256=json.loads((a.training/"TRAIN_PROTOCOL.json").read_text())["train_source_sha256"],
        decoder_sha256=train["checkpoint_sha256"])
    save_json(a.root/"AUDIT.json",report);print(json.dumps(report),flush=True)
    if not report["passed"]:raise RuntimeError("Experiment audit failed")

if __name__=="__main__":main()
