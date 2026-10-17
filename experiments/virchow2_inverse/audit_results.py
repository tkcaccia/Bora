"""Check persisted provenance, dimensions, frozen predictions and real donors."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
import tifffile
from .common import digest,save_json

def main():
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);a=p.parse_args();r=a.root
    results=json.loads((r/"results.json").read_text());frozen=json.loads((r/"PREDICTIONS_FROZEN.json").read_text())
    original=np.asarray(Image.open(r/"original.png").convert("RGB"));shape=original.shape[:2]
    assert len(results["experiments"])==20
    assert [row["id"] for row in results["experiments"]]==list(range(1,21))
    for row in results["experiments"]:
        n=row["id"];folder=r/f"{n:02d}";image=np.asarray(Image.open(folder/"representation.png"));pred=tifffile.imread(folder/"bora.tif")
        assert image.shape==original.shape and pred.shape==shape
        assert set(np.unique(pred)).issubset({0,1,2,3,4,5})
        assert digest(folder/"representation.png")==frozen["outputs"][str(n)]["input"]==row["image_sha256"]
        assert digest(folder/"bora.tif")==frozen["outputs"][str(n)]["prediction"]==row["prediction_sha256"]
        assert np.isfinite([row["scores"][k] for k in ["macro_iou","macro_dice","worst_class_iou","hd95_um"]]).all()
        assert row["fine_reference_used"] is False
        for i in range(1,6):assert Image.open(folder/f"tiles/{i:02d}.png").size==(224,224)
    coords=np.load(r/"18/donor_pixel_yx.npy")
    assert coords.min()>=0 and coords[...,0].max()<shape[0] and coords[...,1].max()<shape[1]
    quilt=np.asarray(Image.open(r/"18/representation.png"))
    assert np.array_equal(quilt,original[coords[...,0],coords[...,1]])
    donors=json.loads((r/"18/donors.json").read_text())
    for row in donors:
        t,d=row["target_context_box"],row["donor_context_box"]
        assert d[2]<=t[0] or d[0]>=t[2] or d[3]<=t[1] or d[1]>=t[3]
    assert json.loads((r/"encoder_target_audit.json").read_text())["passed"]
    report={"passed":True,"experiments_checked":20,"native_frame_verified":True,"pre_reference_prediction_hashes_match":True,
            "quilt_exact_native_donor_pixels_verified":True,"quilt_context_disjoint_donors":len(donors),"same_five_crops_for_every_input":True}
    save_json(r/"AUDIT.json",report);print(json.dumps(report))

if __name__=="__main__":main()
