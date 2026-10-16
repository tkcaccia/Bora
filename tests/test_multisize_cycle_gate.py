"""An internal-edge artifact cannot pass the multisize acceptance gate."""
import json

from scripts.evaluate_multisize_cycle import evaluate


def _write_case(root, evaluation, *, new, artifact):
    for size in (56, 90, 216):
        for index in range(4):
            sample = f"sample_{index}"
            output = root / str(size) / sample
            scores = evaluation / str(size) / sample
            output.mkdir(parents=True, exist_ok=True)
            scores.mkdir(parents=True, exist_ok=True)
            (output / "comparison.json").write_text(json.dumps({
                "refinement_seconds": 0.9 if new else 1.0}))
            (scores / "vector_comparison.json").write_text(json.dumps({
                "vector_vs_ground_truth": {
                    "per_class": [{"tp": 20, "support": 20}]}}))
            (scores / "border_comparison_r64.json").write_text(json.dumps({
                "internal": {"bora": {
                    "macro_iou": 0.81 if new else 0.80,
                    "accuracy": 0.91 if new else 0.90,
                    "false_tile_imprint_edge_pixels": 90 if new else 100,
                    "off_band_internal_edge_pixels": (11 if artifact else 9)
                    if new else 10,
                }}}))


def test_rejects_new_internal_edges_outside_border_band(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    old_eval, new_eval = tmp_path / "old_eval", tmp_path / "new_eval"
    _write_case(old, old_eval, new=False, artifact=False)
    _write_case(new, new_eval, new=True, artifact=True)
    rejected = evaluate(new, new_eval, old, old_eval)
    assert not rejected["accepted"]
    assert not rejected["rows"][0]["off_band_internal_edges_non_increasing"]

    _write_case(new, new_eval, new=True, artifact=False)
    assert evaluate(new, new_eval, old, old_eval)["accepted"]
