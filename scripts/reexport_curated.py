#!/usr/bin/env python3
"""Re-export cached curated predictions without retraining Bora."""
import argparse
import json
from pathlib import Path

import numpy as np

from bora.curated_training import geojson_properties, write_scaled_geojson


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("optimization_dir", type=Path)
    parser.add_argument("--sample", action="append", default=[])
    args = parser.parse_args()
    for report_path in sorted(args.optimization_dir.glob("*/optimization_report.json")):
        report = json.loads(report_path.read_text())
        if args.sample and report["sample"] not in args.sample:
            continue
        prediction = np.load(report_path.parent / "calibration_prediction.npz")["prediction"]
        count = write_scaled_geojson(
            prediction, report["geojson"], report["config"]["downsample"],
            geojson_properties(report["curated_target"]))
        report["geojson_feature_count"] = count
        report["geojson_class_id_encoding"] = "explicit_properties.value"
        report_path.write_text(json.dumps(report, indent=2))
        print(f"{report['sample']}: {count} features")


if __name__ == "__main__":
    main()
