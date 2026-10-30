"""Frozen colorectal WSI workflow and retained compatibility commands."""
import argparse
import json
import os
from pathlib import Path
import tempfile
import cv2
import numpy as np
import tifffile
from threadpoolctl import threadpool_limits
from . import __version__
from .classifier import NeighborContextClassifier
from .annotations import read_annotation, group_targets, _class_map
from .pipeline import load_case, prepare, run, json_file, digest, _mask


def parser():
    p = argparse.ArgumentParser(prog="bora", description="Frozen global RGB9+19/H&E WSI refinement and Neighbor context fastPLS region identity")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("wsi-fit", help="fit one slide-wide RGB9+19 response model (selected workflow)")
    a.add_argument("manifest", type=Path); a.add_argument("--output", type=Path, required=True)
    a = sub.add_parser("wsi-project", help="apply the frozen response model across the complete slide")
    a.add_argument("manifest", type=Path); a.add_argument("--fit", type=Path, required=True)
    a.add_argument("--output", type=Path, required=True)
    a = sub.add_parser("wsi", help="selected default: one global mask and compact diagonal GeoJSON")
    a.add_argument("manifest", type=Path); a.add_argument("--output", type=Path, required=True)
    a.add_argument("--minimum-free-gib", type=float, default=1.0)
    a.add_argument("--memory-gib", type=float, default=12.0)
    a = sub.add_parser("wsi-polished", help="whole-slide result with configurable physical contour smoothing and minimum geometry")
    a.add_argument("manifest", type=Path); a.add_argument("--output", type=Path, required=True)
    a.add_argument("--minimum-free-gib", type=float, default=1.0)
    a.add_argument("--memory-gib", type=float, default=12.0)
    a.add_argument("--minimum-area-um2", type=float, default=2000.0,
                   help="strict minimum component and enclosed-hole area in µm² (default: 2000)")
    a.add_argument("--contour-sigma-um", "--contour-smoothing-um", type=float, default=6.0,
                   help="Gaussian class-contour smoothing sigma in µm (default: 6)")
    a = sub.add_parser("wsi-crop", help="crop an existing full-slide mask and geometry; never refit")
    a.add_argument("result", type=Path); a.add_argument("--box", type=int, nargs=4, metavar=("X","Y","W","H"), required=True)
    a.add_argument("--output", type=Path, required=True)
    a = sub.add_parser("prepare", help="prepare whole-group geometry/descriptors for training")
    a.add_argument("case", type=Path); a.add_argument("--output", type=Path, required=True)
    a = sub.add_parser("train", help="fit the single Neighbor context classifier on declared training groups")
    a.add_argument("manifest", type=Path); a.add_argument("--output", type=Path, required=True)
    a = sub.add_parser("refine", help="retained single-field RGB9+19/Bora-AF route")
    a.add_argument("case", type=Path); a.add_argument("--model", type=Path, required=True)
    a.add_argument("--output", type=Path, required=True)
    a = sub.add_parser("kodama-contours", help="legacy three-group contour compatibility command")
    a.add_argument("assembly", type=Path, help="complete frozen native-mask assembly folder")
    a.add_argument("native_he", type=Path, help="registered native RGB H&E TIFF")
    a.add_argument("group_names", type=Path, help="JSON object mapping IDs 1, 2, 3 to group names")
    a.add_argument("--output", type=Path, required=True)
    a.add_argument("--minimum-area-um2", type=float, default=1000.0)
    a.add_argument("--minimum-free-gib", type=float, default=1.0)
    a = sub.add_parser("kodama-fit", help="legacy three-group PCA32 fitting compatibility command")
    a.add_argument("manifest", type=Path)
    a.add_argument("--output", type=Path, required=True)
    a = sub.add_parser("kodama-score", help="legacy three-group PCA32 projection compatibility command")
    a.add_argument("manifest", type=Path)
    a.add_argument("--model", type=Path, required=True)
    a.add_argument("--output", type=Path, required=True)
    a.add_argument("--sigma-tokens", type=float, required=True)
    a.add_argument("--prior-exponent", type=float, required=True)
    a = sub.add_parser("kodama-native", help="legacy three-group native-mask compatibility command")
    a.add_argument("manifest", type=Path)
    a.add_argument("--output", type=Path, required=True)
    a = sub.add_parser("classify-regions", help="name existing tissue polygons with frozen Neighbor-context PLS")
    a.add_argument("contours", type=Path, help="accepted native-coordinate tissue GeoJSON")
    a.add_argument("registration", type=Path, help="bora-region-registration-v1 JSON")
    a.add_argument("--tokens", type=Path, required=True)
    a.add_argument("--model", type=Path, required=True)
    a.add_argument("--output", type=Path, required=True)
    a.add_argument("--group-field", default="value")
    a.add_argument("--stripe-rows", type=int, default=256)
    a.add_argument("--maximum-working-mib", type=int, default=256)
    return p


def prepare_file(case_path, output):
    if output.exists():
        raise FileExistsError(output)
    case = load_case(case_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".bora-", dir=output.parent) as temp:
        folder = Path(temp) / "prepared"
        folder.mkdir()
        data = prepare(case)
        _mask(folder / "groups.ome.tif", data["regions"].astype(np.uint32), case.native_mpp_xy)
        meta = dict(format="bora-neighbor-context-descriptors-v1", provenance=case.provenance,
            mpp_xy=case.mpp_xy.tolist(), native_mpp_xy=case.native_mpp_xy.tolist(),
            prediction_unit="Whole anonymous group including disconnected pieces")
        np.savez_compressed(folder / "descriptors.npz", x=data["descriptors"],
            group_ids=data["group_ids"], area_pixels=data["area_pixels"], metadata=json.dumps(meta))
        json_file(folder / "report.json", dict(data["details"], **meta,
            descriptors_sha256=digest(folder / "descriptors.npz")))
        if output.exists():
            raise FileExistsError(output)
        os.rename(folder, output)
    return dict(output=str(output), groups=len(data["group_ids"]))


def train_file(manifest_path, output):
    if output.exists():
        raise FileExistsError(output)
    spec = json.loads(manifest_path.read_text())
    if spec.get("format") != "bora-training-v1" or not spec.get("cases"):
        raise ValueError("A nonempty bora-training-v1 manifest is required")
    names = {int(k): v for k, v in spec["classes"].items()}
    xs, ys, seen, records = [], [], set(), []
    for row in spec["cases"]:
        if row.get("split") != "train" or not row.get("source_group"):
            raise ValueError("Every descriptor file needs split=train and a source_group identity")
        path = (manifest_path.parent / row["descriptors"]).resolve()
        sha = digest(path)
        if row.get("sha256", sha) != sha or sha in seen:
            raise ValueError("Changed or duplicated training descriptors")
        seen.add(sha)
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata"].item()))
            if meta.get("format") != "bora-neighbor-context-descriptors-v1":
                raise ValueError("Use descriptors produced by bora prepare")
            ids = data["group_ids"].copy()
            if ("labels" in row) == ("annotation" in row):
                raise ValueError("Supply either explicit labels or a training annotation, not both")
            if "labels" in row:
                labels = row["labels"]
                if set(labels) != set(map(str, ids.tolist())):
                    raise ValueError("Supply one tissue target for every whole anonymous group")
                target = np.asarray([labels[str(int(g))] for g in ids])
                keep = np.ones(len(ids), bool)
                excluded = []
                annotation_sha = None
                groups_sha = None
            else:
                groups_path = (manifest_path.parent / row["groups"]).resolve()
                annotation_path = (manifest_path.parent / row["annotation"]).resolve()
                groups_sha, annotation_sha = digest(groups_path), digest(annotation_path)
                if (row.get("groups_sha256", groups_sha) != groups_sha or
                        row.get("annotation_sha256", annotation_sha) != annotation_sha):
                    raise ValueError("Changed training groups or annotation")
                groups = tifffile.imread(groups_path)
                mapping = _class_map(row["annotation_values"], names)
                annotated = read_annotation(annotation_path, groups.shape, mapping,
                    field=row.get("annotation_field", "value"),
                    ignore_values=row.get("ignore_values", ["0"]),
                    coordinate_space=row.get("annotation_coordinate_space"))
                retained, target, excluded = group_targets(groups, ids, annotated,
                    min_coverage=float(row.get("min_coverage", .5)),
                    min_purity=float(row.get("min_purity", .8)))
                keep = np.isin(ids, retained)
                if not np.array_equal(ids[keep], retained):
                    raise ValueError("Training group IDs changed order")
            xs.append(data["x"][keep].copy()); ys.append(target)
            records.append(dict(source_group=row["source_group"], descriptor_sha256=sha,
                groups_sha256=groups_sha, annotation_sha256=annotation_sha,
                training_groups=int(np.count_nonzero(keep)),
                excluded_groups=excluded))
    if not any(len(x) for x in xs):
        raise ValueError("No sufficiently annotated training groups remain")
    model = NeighborContextClassifier().fit(np.concatenate(xs), np.concatenate(ys), spec["classes"])
    meta = model.save(output, training_provenance=dict(manifest_sha256=digest(manifest_path), cases=records))
    return dict(output=str(output), training_groups=len(model.y_), model=meta,
                fit_seconds=model.fit_seconds_, training_manifest_sha256=digest(manifest_path),
                excluded_groups=sum(len(r["excluded_groups"]) for r in records))


def main(argv=None):
    a = parser().parse_args(argv)
    cv2.setNumThreads(2)
    try:
        with threadpool_limits(limits=4 if a.command in {"wsi-fit", "wsi-project", "wsi", "wsi-polished", "wsi-crop"} else 2):
            if a.command == "wsi-fit":
                from .slide_responses import fit_slide_responses
                report = fit_slide_responses(a.manifest, a.output)
            elif a.command == "wsi-project":
                from .slide_responses import project_slide_responses
                report = project_slide_responses(a.manifest, a.fit, a.output)
            elif a.command == "wsi":
                from .wsi_frozen import run_wsi
                if not np.isfinite([a.minimum_free_gib,a.memory_gib]).all() or a.minimum_free_gib < 0 or a.memory_gib <= 0:
                    raise ValueError("Resource limits must be finite and nonnegative")
                report = run_wsi(a.manifest,a.output,minimum_free_bytes=int(a.minimum_free_gib*1024**3),memory_budget_bytes=int(a.memory_gib*1024**3))
            elif a.command == "wsi-polished":
                from .wsi_polished import run_wsi
                if not np.isfinite([a.minimum_free_gib,a.memory_gib]).all() or a.minimum_free_gib < 0 or a.memory_gib <= 0:
                    raise ValueError("Resource limits must be finite and nonnegative")
                report = run_wsi(a.manifest,a.output,minimum_free_bytes=int(a.minimum_free_gib*1024**3),memory_budget_bytes=int(a.memory_gib*1024**3),minimum_area_um2=a.minimum_area_um2,contour_sigma_um=a.contour_sigma_um)
            elif a.command == "wsi-crop":
                from .wsi_frozen import crop_wsi
                report = crop_wsi(a.result,a.box,a.output)
            elif a.command == "prepare":
                report = prepare_file(a.case, a.output)
            elif a.command == "train":
                report = train_file(a.manifest, a.output)
            elif a.command == "kodama-contours":
                from .kodama_bora import publish_three_group_contours
                if not np.isfinite(a.minimum_free_gib) or a.minimum_free_gib < 0:
                    raise ValueError("Free-space reserve must be a nonnegative GiB value")
                names = json.loads(a.group_names.read_text())
                report = publish_three_group_contours(
                    a.assembly, a.native_he, a.output, names,
                    minimum_area_um2=a.minimum_area_um2,
                    minimum_free_bytes=int(a.minimum_free_gib * 1024**3))
            elif a.command == "kodama-fit":
                from .kodama_bora import fit_three_group_model
                report = fit_three_group_model(a.manifest, a.output)
            elif a.command == "kodama-score":
                from .kodama_bora import score_three_group_model
                report = score_three_group_model(a.manifest, a.model, a.output,
                    sigma_tokens=a.sigma_tokens,
                    prior_exponent=a.prior_exponent)
            elif a.command == "kodama-native":
                from .kodama_native import refine_global_score_to_native
                report = refine_global_score_to_native(a.manifest, a.output)
            elif a.command == "classify-regions":
                from .region_identity import classify_regions
                report = classify_regions(a.contours, a.registration, a.tokens,
                    a.model, a.output, group_field=a.group_field, stripe_rows=a.stripe_rows,
                    maximum_working_bytes=a.maximum_working_mib * 1024**2)
            else:
                case = load_case(a.case)
                model = NeighborContextClassifier.load(a.model)
                report = run(case, model, a.output)
    except (OSError, ValueError, KeyError, ImportError) as exc:
        raise SystemExit(f"bora {a.command}: {exc}") from exc
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
