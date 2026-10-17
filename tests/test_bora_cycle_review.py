import json
import pytest
from experiments.bora_cycles.run import digest, save
from experiments.bora_cycles.review import record_review, require_review, validate_children


def fixture(root, pilot):
    (root/"cases/example").mkdir(parents=True)
    (root/"cases/example/contact.jpg").write_bytes(b"actual frozen sheet")
    sha = digest(root/"cases/example/contact.jpg")
    save(root/"VISUAL_MANIFEST.json", dict(contact_sha256={"example": sha}))
    save(root/"VISUAL_REVIEW.json", dict(reviewed={}))
    save(root/"PROTOCOL.json", dict(pilot=pilot, cases=["example"], methods=["C01", "C02"],
                                    roster={"C01": {}, "C02": {}}))
    return sha


def test_review_requires_actual_viewed_sheet_and_all_methods(tmp_path):
    sha = fixture(tmp_path, True)
    note = "C01 preserves the central island; C02 expands the outer region."
    with pytest.raises(ValueError, match="frozen sheet"):
        record_review(tmp_path, "example", "wrong", note, ["C01", "C02"])
    with pytest.raises(ValueError, match="every method"):
        record_review(tmp_path, "example", sha, note, ["C01"])
    record_review(tmp_path, "example", sha, note, ["C01", "C02"])
    review = json.loads((tmp_path/"VISUAL_REVIEW.json").read_text())
    assert review["reviewed"]["example"]["sheet_sha256"] == sha


def test_pilot_is_not_completed_cycle_even_after_review(tmp_path):
    fixture(tmp_path, True)
    with pytest.raises(ValueError, match="pilot"):
        require_review(tmp_path)


def test_full_scope_cannot_be_replaced_with_fewer_cases(tmp_path):
    fixture(tmp_path, False)
    with pytest.raises(ValueError, match="149"):
        require_review(tmp_path)


def test_each_of_five_parents_needs_three_real_interventions():
    parents = [f"C{i:02d}" for i in range(1, 6)]
    children = [
        dict(id=f"D{i}_{j}", parent=parent, intervention=f"mechanism_{i}_{j}",
             equation_or_rule="A concrete mathematical rule stated before the next cycle is evaluated.",
             failure_hypothesis="This intervention may remove genuine small disconnected tissue regions.",
             difference_from_parent="The child changes the inference equation, not a numerical threshold.")
        for i, parent in enumerate(parents) for j in range(3)]
    assert validate_children(dict(survivors=parents), children)
    children[0]["parameter_sweep"] = True
    with pytest.raises(ValueError, match="parameter"):
        validate_children(dict(survivors=parents), children)
