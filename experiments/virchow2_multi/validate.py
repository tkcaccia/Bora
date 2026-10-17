"""Compare generalized implementations to the original five-class outputs."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
import tifffile
from threadpoolctl import threadpool_limits
from experiments.virchow2_inverse.common import save_json,digest
from experiments.virchow2_inverse.protocol import COLOURS,bora_rgb
from experiments.virchow2_inverse.cpu_ideas import run_idea,additional_grid
from .methods import Context,METHODS,run_method


def main():
    p=argparse.ArgumentParser();p.add_argument("--previous",type=Path,required=True);p.add_argument("--source",type=Path,required=True)
    p.add_argument("--root",type=Path,required=True);a=p.parse_args()
    image=np.array(Image.open(a.source/"image.original.png").convert("RGB"));coarse=tifffile.imread(a.source/"tile_90/coarse.tif")
    features=np.load(a.previous/"features/dense_patch_features.npy",mmap_mode="r");rows=[]
    with threadpool_limits(limits=2):
        c=Context(image,coarse,features,COLOURS);c.root=a.previous
        expected=json.loads((a.previous/"initial_feature_selection.json").read_text())["indices"]
        if c.selected.tolist()!=expected:raise ValueError("Feature selection changed on the original five-class case")
        c.output=lambda number,rgb,*args:rgb
        variants={name:additional_grid(c,name)[0] for name in ("scale_1","scale_2")}
        for n in METHODS:
            rgb,_,_,_=run_method(n,c,variants);original_formula=run_idea(n,c)
            reference=np.array(Image.open(a.previous/f"{n:02d}/representation.png").convert("RGB"))
            difference=abs(rgb.astype(np.int16)-reference.astype(np.int16))
            mask=bora_rgb(rgb,coarse);reference_mask=tifffile.imread(a.previous/f"{n:02d}/bora.tif")
            row=dict(id=n,max_abs_rgb_difference=int(difference.max()),mean_abs_rgb_difference=float(difference.mean()),
                fraction_rgb_values_changed=float(np.mean(difference>0)),exact_published_rgb_match=bool(np.array_equal(rgb,reference)),
                exact_old_formula_match_with_same_context=bool(np.array_equal(rgb,original_formula)),
                bora_mask_agreement=float(np.mean(mask==reference_mask)))
            rows.append(row);print(json.dumps(row),flush=True)
    passed=all(r["exact_published_rgb_match"] and r["bora_mask_agreement"]==1 for r in rows)
    save_json(a.root/"FIVE_CLASS_REGRESSION.json",dict(rows=rows,source_image_sha256=digest(a.source/"image.original.png"),
        fine_reference_read=False,earlier_results_written=False,
        criterion="exact RGB and Bora-mask match with all seven original published five-class outputs",
        passed=passed))
    if not passed:raise RuntimeError("Original-example regression failed")

if __name__=="__main__":main()
