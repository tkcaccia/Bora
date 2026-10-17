"""Assemble all349 inputs and freeze unchanged conversions; never run Bora.

Native reference files are transferred and hashed as input acquisition, but
conversion functions never decode them. Historical149 input manifests and RGB
files are copied exactly. New conversions use the audited200 feature caches.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gc
import json
from multiprocessing import get_context
from pathlib import Path
import shutil
import time

import cv2
import numpy as np
from PIL import Image
import tifffile
from threadpoolctl import threadpool_limits

from experiments.virchow2_inverse.common import digest, save_json
from experiments.virchow2_multi.methods import Context, align_variant, run_method
from experiments.virchow2_ensemble.run import mean_evidence
from experiments.virchow2_tissue.support import tissue_support, complete_tiles
from .features import verify_case
from .task_adapter import geometry, report_units

CONVERSIONS = ("09", "10", "15", "19", "ensemble")


def json_safe(value):
    """Undefined diagnostic correlations are null, not invented zeros."""
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def convert_maps(image, coarse, features, palette, support, selection, variants):
    classes = np.asarray(selection["observed_classes"], dtype=int)
    if len(classes) == 1:
        # A one-class probability is identically one; no classifier can be fit.
        colour = np.asarray(palette, np.uint8)[classes[0]-1]
        rgb = np.broadcast_to(colour, (*coarse.shape, 3)).copy()
        details = dict(degenerate_single_eligible_class=True, observed_classes=classes.tolist(),
            rule="One observed-class probability is identically1; every conversion is constant",
            fine_reference_used=False, biological_missing_classes_inferred=False)
        return {k: (rgb.copy(), dict(details)) for k in CONVERSIONS}, np.ones((*coarse.shape, 1), np.float32), classes
    c = Context(image, coarse, features, palette, analysis_support=support)
    if c.selected.tolist() != selection["selection"]["indices"] or c.classes.tolist() != classes.tolist():
        raise ValueError("Conversion/feature selection differs from frozen cache")
    aligned = {name: align_variant(values, metadata["shape_yx"], c)
               for name, (values, metadata) in variants.items()}
    outputs, evidence = {}, {}
    for number in (9, 10, 15, 19):
        rgb, _, details, _, p = run_method(number, c, aligned, return_probabilities=True)
        if not np.isfinite(p).all() or rgb.shape != (*coarse.shape, 3):
            raise ValueError("Invalid unchanged-conversion output")
        # Legacy labels described nominal0.5um sampling, not the actual axes.
        details.pop("nominal_sampling_um", None)
        details.pop("object_scales_um", None)
        if number == 15:
            details["object_gaussian_sigma_px"] = [12, 40, 24]
            details["object_distance_cap_px"] = 100
            details["object_size_limits_px2"] = dict(nuclei=[12, 1200], lumens=[40, 12000])
        outputs[f"{number:02d}"] = rgb, json_safe(details)
        if number in (9, 19):
            evidence[number] = p
    p = mean_evidence([evidence[9], evidence[19]], [c.classes, c.classes])
    outputs["ensemble"] = c.probability_rgb(p), dict(components=[9, 19], weights=[.5, .5], fine_reference_used=False)
    return outputs, c.native_p, c.classes


def copy_verified(source, destination, expected):
    if digest(source) != expected:
        raise ValueError("Acquisition source changed: " + str(source))
    if destination.exists():
        if digest(destination) != expected:
            raise ValueError("Existing assembled input changed: " + str(destination))
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    if digest(destination) != expected:
        raise ValueError("Input copy mismatch")


def transfer_native_files(source, root, sample):
    """Acquisition-only copying, with no categorical reference decoding."""
    src, dst = source/"cases"/sample["case"], root/"cases"/sample["case"]
    dst.mkdir(parents=True, exist_ok=True)
    for name, expected in (("original.png", sample["original_sha256"]),
            ("coarse.tif", sample["coarse_sha256"]),
            ("reference.native.tif", sample["reference_native_sha256"]),
            ("original.native.png", digest(src/"original.native.png"))):
        copy_verified(src/name, dst/name, expected)


def prepare_case(task):
    root, feature_root, sample = task
    root, feature_root = Path(root), Path(feature_root)
    folder = root/"cases"/sample["case"]
    frozen = folder/"INPUTS_FROZEN.json"
    if frozen.exists():
        record = json.loads(frozen.read_text())
        for name, expected in record["files_sha256"].items():
            if digest(folder/name) != expected:
                raise ValueError("Frozen conversion input changed")
        return sample["case"]
    started = time.monotonic()
    cv2.setNumThreads(2)
    with threadpool_limits(limits=2):
        cache = feature_root/"cases"/sample["case"]
        feature_record = verify_case(cache, sample)
        image = np.array(Image.open(folder/"original.png").convert("RGB"))
        native = np.array(Image.open(folder/"original.native.png").convert("RGB"))
        coarse = tifffile.imread(folder/"coarse.tif")
        if list(image.shape[:2]) != sample["working_shape_yx"] or list(native.shape[:2]) != sample["native_shape_yx"]:
            raise ValueError("Assembled RGB geometry mismatch")
        native_coordinates = geometry(sample, native=True)
        native_support, policy = tissue_support(native, native_coordinates["algorithm_spacing_xy"])
        policy = report_units(policy, native_coordinates["units"])
        policy["coordinate_system"] = native_coordinates
        support = cv2.resize(native_support.astype(np.float32), tuple(image.shape[:2][::-1]), interpolation=cv2.INTER_AREA) > .2
        if not support.any():
            raise ValueError("No original-image tissue support")
        selection = json.loads((cache/"FEATURE_SELECTION.json").read_text())
        f = np.load(cache/"features/base.npy", mmap_mode="r")
        variants, feature_manifest = {}, {}
        for name in ("base", "scale_1", "scale_2"):
            path = cache/"features"/(name+".npy")
            meta = json.loads((cache/"features"/("READY.json" if name == "base" else name+".json")).read_text())
            feature_manifest[name] = dict(path=str(path), sha256=meta["sha256"], sampling=meta["sampling"])
            if name != "base":
                variants[name] = np.load(path).astype(np.float32), meta
        outputs, plain_prob, classes = convert_maps(image, coarse, f, sample["palette"], support, selection, variants)
        completed, completion = complete_tiles(coarse, plain_prob, classes, support)
        save_json(folder/"selection.json", selection)
        save_json(folder/"completion.json", completion)
        save_json(folder/"tissue_policy.json", policy)
        base_meta = json.loads((cache/"features/READY.json").read_text())
        save_json(folder/"encoder_coverage.json", dict(**base_meta["coverage"],
            feature_manifest_sha256=digest(cache/"FEATURES_FROZEN.json"),
            sampling=feature_record["sampling"]))
        for name, array in (("coarse.completed.tif", completed),
                ("tissue.native.tif", native_support.astype(np.uint8)),
                ("tissue.working.tif", support.astype(np.uint8))):
            tifffile.imwrite(folder/name, array, compression="deflate")
        for key, (rgb, details) in outputs.items():
            target = folder/key
            target.mkdir(exist_ok=True)
            Image.fromarray(rgb).save(target/"representation.png")
            save_json(target/"details.json", dict(**details,
                original_conversion_mechanism_unchanged=True,
                actual_base_sampling=feature_record["sampling"],
                actual_scale_sampling={n: m["sampling"] for n, (_, m) in variants.items()},
                pixel_shape_yx=list(image.shape[:2])))
        names = ["original.png", "original.native.png", "coarse.tif", "coarse.completed.tif",
                 "selection.json", "completion.json", "tissue.native.tif", "tissue.working.tif",
                 "tissue_policy.json", "encoder_coverage.json"]
        names += [key+"/"+name for key in CONVERSIONS for name in ("representation.png", "details.json")]
        record = dict(files_sha256={n: digest(folder/n) for n in names},
            feature_cache=feature_manifest, conversions=list(CONVERSIONS),
            feature_manifest_sha256=digest(cache/"FEATURES_FROZEN.json"),
            single_eligible_class=selection["single_coarse_class"],
            fine_reference_read=False, fine_reference_decoded_by_converter=False,
            reference_transfer="separate acquisition-only byte copy; no mask values used",
            bora_run=False, seconds=time.monotonic()-started)
        save_json(frozen, record)
    gc.collect()
    print(json.dumps(dict(converted=sample["case"], seconds=round(record["seconds"], 3))), flush=True)
    return sample["case"]


def assemble(historical, sources, features, root):
    sources = [Path(s) for s in sources]
    old = json.loads((historical/"PLAN.json").read_text())
    additions = [(source, row) for source in sources
                 for row in json.loads((source/"PLAN.json").read_text())["samples"]]
    feature_freeze = json.loads((features/"ALL_FEATURES_FROZEN.json").read_text())
    if feature_freeze["cases"] != len(additions) or set(feature_freeze["case_manifest_sha256"]) != {s["case"] for _, s in additions}:
        raise ValueError("The full added feature corpus is required")
    old_freeze = json.loads((historical/"ALL_INPUTS_FROZEN.json").read_text())
    if old_freeze["plan_sha256"] != digest(historical/"PLAN.json"):
        raise ValueError("Historical plan changed")
    for _, sample in additions:
        path = features/"cases"/sample["case"]/"FEATURES_FROZEN.json"
        if digest(path) != feature_freeze["case_manifest_sha256"][sample["case"]]:
            raise ValueError("Audited feature case manifest changed")
    samples = old["samples"]+[s for _, s in additions]
    if len({s["case"] for s in samples}) != len(samples):
        raise ValueError("Duplicate benchmark case")
    plan = dict(samples=samples, historical_cases=[s["case"] for s in old["samples"]],
        additional_cases=[s["case"] for _, s in additions],
        source_plan_sha256={str(p): digest(p/"PLAN.json") for p in [historical]+sources},
        historical_inputs_sha256=digest(historical/"ALL_INPUTS_FROZEN.json"),
        added_features_sha256=digest(features/"ALL_FEATURES_FROZEN.json"),
        all_images_are_development=True, all_analysis_images_are_ROIs=True,
        conversions=list(CONVERSIONS), historical_conversions_modified=False,
        primary_conversion="ensemble", physical_metrics_for_uncalibrated=False)
    root.mkdir(parents=True, exist_ok=True)
    if (root/"PLAN.json").exists() and json.loads((root/"PLAN.json").read_text()) != plan:
        raise ValueError("Expanded assembly plan changed")
    save_json(root/"PLAN.json", plan)
    for sample in old["samples"]:
        src, dst = historical/"cases"/sample["case"], root/"cases"/sample["case"]
        transfer_native_files(historical, root, sample)
        manifest = json.loads((src/"INPUTS_FROZEN.json").read_text())
        if digest(src/"INPUTS_FROZEN.json") != old_freeze["manifests"][sample["case"]]:
            raise ValueError("Historical input manifest changed")
        for name, expected in manifest["files_sha256"].items():
            copy_verified(src/name, dst/name, expected)
        copy_verified(src/"INPUTS_FROZEN.json", dst/"INPUTS_FROZEN.json", digest(src/"INPUTS_FROZEN.json"))
    for source, sample in additions:
        transfer_native_files(source, root, sample)
    return plan, [s for _, s in additions]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--historical", type=Path, required=True)
    p.add_argument("--sources", type=Path, nargs="+", required=True)
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--cases", nargs="*")
    a = p.parse_args()
    if not 1 <= a.workers <= 3:
        raise ValueError("One to three low-priority image workers required")
    if shutil.disk_usage(a.root.parent).free < 20*1024**3:
        raise RuntimeError("Preserve20GiB free before conversion assembly")
    plan, samples = assemble(a.historical, a.sources, a.features, a.root)
    repo = Path(__file__).resolve().parents[2]
    files = [f for name in ("bora_extension", "virchow2_multi", "virchow2_inverse", "virchow2_tissue", "virchow2_ensemble")
             for f in sorted((repo/"experiments"/name).glob("*.py"))]
    protocol = dict(plan_sha256=digest(a.root/"PLAN.json"),
        source_sha256={str(f.relative_to(repo)): digest(f) for f in files},
        fine_reference_decoding_by_converter=False, bora_run=False,
        physical_coordinates="Measured axes where available; otherwise explicit numerical pixels",
        numeric_conversion_mechanisms_unchanged=True)
    if (a.root/"ASSEMBLY_PROTOCOL.json").exists() and json.loads((a.root/"ASSEMBLY_PROTOCOL.json").read_text()) != protocol:
        raise ValueError("Frozen assembly implementation changed")
    save_json(a.root/"ASSEMBLY_PROTOCOL.json", protocol)
    if a.cases:
        if not set(a.cases) <= {s["case"] for s in samples}:
            raise ValueError("Unknown added pilot case")
        samples = [s for s in samples if s["case"] in a.cases]
    tasks = [(a.root, a.features, s) for s in samples]
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=get_context("spawn")) as pool:
        list(pool.map(prepare_case, tasks))
    if all((a.root/"cases"/s["case"]/"INPUTS_FROZEN.json").exists() for s in plan["samples"]):
        save_json(a.root/"ALL_INPUTS_FROZEN.json", dict(plan_sha256=digest(a.root/"PLAN.json"),
            cases=len(plan["samples"]), conversions=list(CONVERSIONS),
            manifests={s["case"]: digest(a.root/"cases"/s["case"]/"INPUTS_FROZEN.json") for s in plan["samples"]},
            all_original_fields_encoded=True, fine_references_decoded_by_converter=False,
            historical_inputs_modified=False, bora_run=False))
