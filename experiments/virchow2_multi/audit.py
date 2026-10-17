"""Verify scope, feature caches, frozen masks, original-case parity and gallery."""
import argparse,json,platform,shutil
from pathlib import Path
from importlib.metadata import version
import numpy as np
from PIL import Image
import tifffile
from experiments.virchow2_inverse.common import digest,save_json
from .methods import METHODS


def main():
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);a=p.parse_args()
    plan=json.loads((a.root/"PLAN.json").read_text());results=json.loads((a.root/"results.json").read_text())
    frozen=json.loads((a.root/"ALL_PREDICTIONS_FROZEN.json").read_text());checks={};sample_checks=[]
    checks["one_region_per_source_case_group"]=len({(r["dataset"],r["group"]) for r in plan["samples"]})==len(plan["samples"])
    checks["case114_excluded"]=all(r["group"]!="114" for r in plan["samples"] if r["dataset"]=="tiger")
    checks["tiger_non_tcga_only"]=all(r["source"].get("cohort")=="non-TCGA" for r in plan["samples"] if r["dataset"]=="tiger")
    checks["original_seven_method_regression"]=json.loads((a.root/"FIVE_CLASS_REGRESSION.json").read_text())["passed"]
    for sample in plan["samples"]:
        folder=a.root/"cases"/sample["case"];manifest=json.loads((folder/"FROZEN.json").read_text())
        row=dict(case=sample["case"],frozen_manifest=digest(folder/"FROZEN.json")==frozen["case_manifest_sha256"][sample["case"]],
            frozen_files=all(digest(folder/name)==sha for name,sha in manifest["files_sha256"].items()),
            native_reference_hash=digest(folder/"reference.native.tif")==sample["reference_native_sha256"],
            original_working_hash=digest(folder/"original.png")==sample["original_sha256"],
            working_geometry=list(Image.open(folder/"original.png").size)==sample["working_shape_yx"][::-1],
            native_geometry=list(Image.open(folder/"original.native.png").size)==sample["native_shape_yx"][::-1],
            physical_fov_matches=bool(np.allclose(np.array(sample["working_shape_yx"])[::-1]*sample["working_mpp_xy"],sample["fov_um_xy"])),
            seven_methods=set(m["id"] for m in manifest["methods"])==set(METHODS),context_head_revision=manifest.get("context_head_revision")==2)
        meta=json.loads((folder/"features/READY.json").read_text());row["base_feature_hash"]=digest(folder/"features/base.npy")==meta["sha256"]
        row["variant_feature_hashes"]=all(digest(folder/f"features/{name}.npy")==json.loads((folder/f"features/{name}.json").read_text())["sha256"] for name in ("scale_1","scale_2"))
        row["native_mask_shapes"]=True;row["working_mask_shapes"]=True;row["all_visual_outputs"]=True
        for n in METHODS:
            key=f"{n:02d}"
            with tifffile.TiffFile(folder/key/"bora.native.tif") as tif:row["native_mask_shapes"] &= list(tif.pages[0].shape)==sample["native_shape_yx"]
            with tifffile.TiffFile(folder/key/"bora.working.tif") as tif:row["working_mask_shapes"] &= list(tif.pages[0].shape)==sample["working_shape_yx"]
            row["all_visual_outputs"] &= all((folder/name).exists() for name in (f"{key}/representation.png",f"{key}/mask.png",f"{key}/before.mask.png",f"thumbs/{key}.input.jpg",f"thumbs/{key}.overlay.jpg","contact.jpg"))
        sample_checks.append(row)
    checks["all_case_checks"]=all(all(v for k,v in r.items() if k!="case") for r in sample_checks)
    matrix=(a.root/"all_outputs.html").read_text()
    checks["complete_output_matrix"]=matrix.count('/representation.png"')==len(plan["samples"])*7 and matrix.count('/bora.native.tif"')==len(plan["samples"])*7
    checks["all_native_scores_retained"]=len(results["cases"])==len(plan["samples"]) and all(set(c["methods"])=={f"{n:02d}" for n in METHODS} for c in results["cases"])
    checks["gallery_contact_sheets_and_report"]=all((a.root/name).exists() for name in ("gallery.html","REPORT.md","all_inputs.jpg","all_overlays.jpg","ranking.png","scores.csv","ranking.csv"))
    visuals={}
    for path in a.root.rglob("*"):
        relative=path.relative_to(a.root)
        if path.is_file() and path.suffix.lower() in (".png",".jpg",".html") and not {"diagnostics","runtime_sources"}.intersection(relative.parts):
            visuals[str(relative)]=digest(path)
    save_json(a.root/"VISUAL_MANIFEST.json",dict(files_sha256=visuals,scope="all current gallery images and HTML; historical diagnostics excluded"))
    source=a.root/"runtime_sources";source.mkdir(exist_ok=True)
    package=Path(__file__).parent
    for path in package.iterdir():
        if path.suffix in (".py",".html"):shutil.copy2(path,source/path.name)
    # Preserve the exact shared helper code, including the production RGB wand.
    shared=source/"virchow2_inverse";shared.mkdir(exist_ok=True)
    for name in ("common.py","selection.py","protocol.py","gpu_ideas.py","cpu_ideas.py","algorithms.py","finalize.py"):
        shutil.copy2(package.parent/"virchow2_inverse"/name,shared/name)
    import bora.boundary
    shutil.copy2(Path(bora.boundary.__file__),source/"bora_boundary.py")
    record=dict(passed=all(checks.values()),checks=checks,cases=sample_checks,python=platform.python_version(),
        visual_manifest_sha256=digest(a.root/"VISUAL_MANIFEST.json"),
        packages={name:version(name) for name in ("torch","timm","numpy","scipy","scikit-learn","scikit-image","Pillow","tifffile")},
        sources_sha256={str(path.relative_to(source)):digest(path) for path in source.rglob("*") if path.is_file()})
    save_json(a.root/"AUDIT.json",record);print(json.dumps(dict(passed=record["passed"],checks=checks)),flush=True)
    if not record["passed"]:raise RuntimeError("Final benchmark audit failed")

if __name__=="__main__":main()
