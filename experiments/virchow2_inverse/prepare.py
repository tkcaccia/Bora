"""Prepare coarse-only feature selection and the fixed class head."""
import argparse,json
from pathlib import Path
import numpy as np
import tifffile
from threadpoolctl import threadpool_limits
from .common import coarse_tile_observations,save_json
from .selection import select_features,blocked_validation
from .protocol import Context

def main():
    p=argparse.ArgumentParser();p.add_argument("--source",type=Path,required=True);p.add_argument("--root",type=Path,required=True);a=p.parse_args()
    info=json.loads((a.source/"results.json").read_text())
    if not np.isclose(info["sample"]["mpp_um"],.5):raise ValueError("This preset is fixed to the native 0.5 um/px TIGER ROI")
    values,labels,groups,records=coarse_tile_observations(np.load(a.root/"features/dense_patch_features.npy"),tifffile.imread(a.source/"tile_90/coarse.tif"))
    with threadpool_limits(limits=2):
        save_json(a.root/"initial_feature_selection.json",select_features(values,labels,"balanced_mrmr",64))
        save_json(a.root/"initial_class_validation.json",blocked_validation(values,labels,groups,"balanced_mrmr",64))
    save_json(a.root/"selection_scope.json",{"observations":records,"labels":labels.tolist(),"spatial_groups":groups.tolist(),"fine_annotation_used":False,"label_source":"reference-derived 90px coarse tile mask"})
    Context(a.source,a.root)

if __name__=="__main__":main()
