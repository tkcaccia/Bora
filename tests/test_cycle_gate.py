"""The smoothing gate measures absolute tile remnants and border accuracy."""
import json
import subprocess
import sys
from pathlib import Path


def _write_cycle(root: Path, vector: Path, *, new: bool) -> None:
    for index in range(4):
        sample = f"sample_{index}"
        folder = root / "216" / sample
        score = vector / "216" / sample
        folder.mkdir(parents=True)
        score.mkdir(parents=True)
        (folder / "comparison.json").write_text(json.dumps({
            "sample": sample, "baseline": {}, "bora": {},
            "refinement_seconds": 1.0,
        }))
        (score / "vector_comparison.json").write_text(json.dumps({
            "vector_vs_ground_truth": {
                "per_class": [{"tp": 10, "support": 10}],
            },
        }))
        metrics = {"macro_iou": .62 if new else .60,
                   "accuracy": .82 if new else .80,
                   "boundary_f1_8px": .18 if new else .20,
                   "false_tile_imprint_edge_pixels": 80 if new else 100,
                   "false_tile_imprint_fraction": .08 if new else .06}
        (score / "border_comparison_r64.json").write_text(json.dumps({
            "internal": {"bora": metrics},
        }))


def test_smoothing_gate_accepts_fewer_absolute_tile_remnants(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    old_vector, new_vector = tmp_path / "old_vector", tmp_path / "new_vector"
    _write_cycle(old, old_vector, new=False)
    _write_cycle(new, new_vector, new=True)
    script = Path(__file__).resolve().parents[1] / "scripts/evaluate_216_cycle.py"
    common = [sys.executable, str(script), "--candidate-root", str(new),
              "--candidate-vector-root", str(new_vector),
              "--incumbent-root", str(old),
              "--incumbent-vector-root", str(old_vector),
              "--border-radius", "64", "--internal-only"]
    smooth = tmp_path / "smooth.json"
    subprocess.run([*common, "--smoothing-gate", "--output", str(smooth)],
                   check=True, capture_output=True)
    result = json.loads(smooth.read_text())
    assert result["accepted"]
    assert result["per_tissue"][0]["diagnostic_boundary_metrics"][
        "internal_boundary_f1_8px"]["change"] < 0
    strict = tmp_path / "strict.json"
    subprocess.run([*common, "--contour-quality-gate", "--output", str(strict)],
                   check=True, capture_output=True)
    assert not json.loads(strict.read_text())["accepted"]
