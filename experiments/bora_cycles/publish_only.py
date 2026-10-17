"""Render frozen outputs in a separate folder; never rerun segmentation."""
import argparse
import json
from pathlib import Path
import time
from .run import digest, publish, save


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, choices=(1, 2, 3), default=3)
    args = p.parse_args()
    if args.input.resolve() == args.output.resolve():
        p.error("Use a separate publication folder to preserve existing review evidence")
    if args.output.exists():
        p.error("Publication output must be a new directory")
    result = json.loads((args.input/"RESULTS.json").read_text())
    args.output.mkdir(parents=True)
    for row in result["cases"]:
        for method, entry in row["methods"].items():
            target = args.output/"cases"/row["case"]/method
            target.mkdir(parents=True)
            if entry["status"] == "ok":
                (target/"native.tif").symlink_to((args.input/"cases"/row["case"]/method/"native.tif").resolve())
    start = time.perf_counter()
    publish(args.source, args.output, result, workers=args.workers)
    save(args.output/"PUBLICATION_BUILD.json", dict(
        input_results_sha256=digest(args.input/"RESULTS.json"),
        publication_source_sha256=digest(Path(__file__).with_name("run.py")),
        method_predictions_modified=False, scores_modified=False,
        segmentation_rerun=False, workers=args.workers,
        seconds=time.perf_counter()-start,
        separate_publication_folder=True))


if __name__ == "__main__":
    main()
