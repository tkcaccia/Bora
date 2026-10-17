"""Audit gates must reject tampering rather than certify stale evidence."""
import json
import numpy as np
from PIL import Image
import pytest
import tifffile

from experiments.bora_cycles.audit_cycle import audit, glass_changes
from experiments.bora_cycles.run import aggregate, digest, palette_bytes, environment_record


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


@pytest.fixture
def evidence(tmp_path):
    source, root = tmp_path/"inputs", tmp_path/"result"
    src, out = source/"cases/example", root/"cases/example"
    (src/"ensemble").mkdir(parents=True)
    (out/"C17").mkdir(parents=True)
    palette = [[210, 20, 30]]
    labels = np.ones((8, 9), np.uint8)
    rgb = np.full((8, 9, 3), palette[0], np.uint8)
    for name, data in (("coarse.tif", labels), ("tissue.native.tif", labels),
                       ("reference.native.tif", labels)):
        tifffile.imwrite(src/name, data)
    Image.fromarray(rgb).save(src/"ensemble/representation.png")
    inputs = {str(p.relative_to(src)): digest(p) for p in src.rglob("*") if p.is_file()}
    write(src/"INPUTS_FROZEN.json", dict(files_sha256=inputs))
    sample = dict(case="example", palette=palette, native_shape_yx=[8, 9],
                  reference_native_sha256=digest(src/"reference.native.tif"))
    write(source/"PLAN.json", dict(samples=[sample]))
    protocol = dict(cycle=3, pilot=True, methods=["C17"], cases=["example"],
                    roster=dict(C17={}), source_hashes={}, lineage=None,
                    input_plan_sha256=digest(source/"PLAN.json"))
    write(root/"PROTOCOL.json", protocol)
    record = dict(status="ok", details={}, seconds=[.1], median_seconds=.1)
    tifffile.imwrite(out/"C17/native.tif", labels)
    write(out/"C17/DETAILS.json", record)
    prediction = dict(methods=dict(C17=record), fine_reference_read=False,
        files_sha256={"C17/"+name: digest(out/"C17"/name) for name in ("native.tif", "DETAILS.json")},
        input_manifest_sha256=digest(src/"INPUTS_FROZEN.json"),
        rgb_sha256=digest(src/"ensemble/representation.png"))
    write(out/"PREDICTIONS.json", prediction)
    write(root/"ALL_PREDICTIONS_FROZEN.json", dict(protocol_sha256=digest(root/"PROTOCOL.json"),
        fine_reference_read=False, manifests=dict(example=digest(out/"PREDICTIONS.json"))))
    score = dict(macro_iou=1., macro_dice=1., regions=dict(small_le_400_um2=dict(
        recall_at_50pct=1., completely_missed=0)))
    row = dict(case="example", dataset="synthetic", sample=sample,
               methods=dict(C17=dict(record, score=score)))
    write(out/"SCORES.json", row)
    write(root/"RESULTS.json", dict(protocol=protocol, cases=[row], ranking=aggregate([row], ["C17"])))
    image = Image.fromarray(labels, "P")
    image.putpalette(palette_bytes(palette))
    image.save(out/"C17/mask.png")
    Image.fromarray(rgb).save(out/"contact.jpg")
    write(root/"VISUAL_MANIFEST.json", dict(contact_sha256=dict(example=digest(out/"contact.jpg"))))
    write(root/"VISUAL_REVIEW.json", dict(reviewed={}))
    return source, root


def test_passing_integrity_audit_does_not_claim_visual_review_or_full_scope(evidence):
    source, root = evidence
    a = audit(source, root)
    assert a["cases"] == 1 and a["pilot"]
    assert a["native_masks"] == a["exact_display_rasters"] == 1
    assert a["actual_review_entries_at_audit"] == 0
    assert a["visual_reviews_created"] is False and a["selection_created"] is False


def test_native_mask_hash_tamper_is_rejected(evidence):
    source, root = evidence
    path = root/"cases/example/C17/native.tif"
    value = tifffile.imread(path);value[0, 0] = 0
    tifffile.imwrite(path, value)
    with pytest.raises(RuntimeError, match="Frozen file changed"):
        audit(source, root)


def test_display_raster_tamper_is_rejected_even_if_tiff_is_unchanged(evidence):
    source, root = evidence
    path = root/"cases/example/C17/mask.png"
    with Image.open(path) as original:
        pixels, palette = np.array(original), original.getpalette()
    pixels[0, 0] = 0
    changed = Image.fromarray(pixels, "P");changed.putpalette(palette);changed.save(path)
    with pytest.raises(ValueError, match="PNG indices differ"):
        audit(source, root)


def test_aggregate_tamper_is_rejected(evidence):
    source, root = evidence
    path = root/"RESULTS.json"
    data = json.loads(path.read_text());data["ranking"][0]["mean_iou"] = .7
    write(path, data)
    with pytest.raises(ValueError, match="Saved aggregate differs"):
        audit(source, root)


def test_new_runtime_aggregate_is_verified_from_its_frozen_policy(evidence):
    from experiments.bora_cycles.runtime_policy import DEFAULT_POLICY, active_policy
    source, root = evidence
    protocol = json.loads((root/'PROTOCOL.json').read_text())
    protocol.update(cycle=5,study_protocol=dict(runtime_penalty=DEFAULT_POLICY))
    write(root/'PROTOCOL.json',protocol)
    freeze = json.loads((root/'ALL_PREDICTIONS_FROZEN.json').read_text())
    freeze['protocol_sha256'] = digest(root/'PROTOCOL.json')
    write(root/'ALL_PREDICTIONS_FROZEN.json',freeze)
    result = json.loads((root/'RESULTS.json').read_text())
    result['protocol'] = protocol
    result['ranking'] = aggregate(result['cases'],protocol['methods'],active_policy(protocol))
    write(root/'RESULTS.json',result)
    assert audit(source,root)['native_masks'] == 1
    result['ranking'][0]['runtime_adjusted_score'] -= .1
    write(root/'RESULTS.json',result)
    with pytest.raises(ValueError,match='Saved aggregate differs'):
        audit(source,root)


def test_glass_only_intervention_cannot_relabel_nonzero_tissue():
    parent = np.array([[1, 2, 0], [2, 1, 2]], np.uint8)
    child = parent.copy();child[0, 0] = child[1, 2] = 0
    assert glass_changes(parent, child) == 2
    child[1, 1] = 2
    with pytest.raises(ValueError, match="nonzero tissue"):
        glass_changes(parent, child)


def test_environment_fingerprint_includes_build_and_loaded_library_identity():
    record = environment_record()
    assert record["packages"]["numpy"] == np.__version__
    assert record["numpy_build"] is not None
    assert len(record["opencv_build_sha256"]) == 64
    assert isinstance(record["threadpools"], list)
    for path, sha in record["loaded_numerical_library_sha256"].items():
        assert sha == digest(path)
    json.dumps(record, allow_nan=False)
