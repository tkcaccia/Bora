"""Command line interface for the sole supported Bora pipeline."""
import argparse
import json
import os
from pathlib import Path
import tempfile
import cv2
import numpy as np
from threadpoolctl import threadpool_limits
from . import __version__
from .classifier import NeighborContextClassifier
from .pipeline import load_case, prepare, run, json_file, digest, _mask


def parser():
    p = argparse.ArgumentParser(prog="bora", description="Bora-AF + Neighbor context (native fastPLS CPU)")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("prepare", help="prepare whole-group geometry/descriptors for training")
    a.add_argument("case", type=Path); a.add_argument("--output", type=Path, required=True)
    a = sub.add_parser("train", help="fit the single Neighbor context classifier on declared training groups")
    a.add_argument("manifest", type=Path); a.add_argument("--output", type=Path, required=True)
    a = sub.add_parser("refine", help="RGB9+19, Bora-AF, Neighbor context identity and compact native GeoJSON")
    a.add_argument("case", type=Path); a.add_argument("--model", type=Path, required=True)
    a.add_argument("--output", type=Path, required=True)
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
    xs, ys, seen = [], [], set()
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
            labels = row["labels"]
            if set(labels) != set(map(str, data["group_ids"].tolist())):
                raise ValueError("Supply one tissue target for every whole anonymous group")
            target = np.asarray([labels[str(int(g))] for g in data["group_ids"]])
            xs.append(data["x"].copy()); ys.append(target)
    model = NeighborContextClassifier().fit(np.concatenate(xs), np.concatenate(ys), spec["classes"])
    meta = model.save(output)
    return dict(output=str(output), training_groups=len(model.y_), model=meta,
                fit_seconds=model.fit_seconds_, training_manifest_sha256=digest(manifest_path))


def main(argv=None):
    a = parser().parse_args(argv)
    cv2.setNumThreads(2)
    try:
        with threadpool_limits(limits=2):
            if a.command == "prepare":
                report = prepare_file(a.case, a.output)
            elif a.command == "train":
                report = train_file(a.manifest, a.output)
            else:
                case = load_case(a.case)
                model = NeighborContextClassifier.load(a.model)
                report = run(case, model, a.output)
    except (OSError, ValueError, KeyError, ImportError) as exc:
        raise SystemExit(f"bora {a.command}: {exc}") from exc
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
