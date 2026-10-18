import json
import pytest
from experiments.bora_cycles.run import digest, save
from experiments.bora_cycles.review import (record_review, require_review, validate_children,
                                          publish_selection_status)


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


def report_fixture(root):
    fixture(root, True)
    survivors = ["F01", "C17", "C01", "D08", "C15"]
    protocol = dict(cycle=3, pilot=False, cases=["example"], methods=survivors+["REJECTED"],
                    roster={key: dict(title=key+" title") for key in survivors+["REJECTED"]})
    save(root/"PROTOCOL.json", protocol)
    score = dict(macro_iou=.6, macro_dice=.7, annotated_unassigned_pixels=0,
                 regions=dict(small_le_400_um2=dict(recall_at_50pct=.2, completely_missed=3)))
    methods = {key: dict(status="ok", median_seconds=.1, score=score) for key in protocol["methods"]}
    ranking = [dict(method=key, complete=True, mean_iou=.6, mean_dice=.7,
                    mean_seconds=.1, small_region_recall=.2, completely_missed_small_regions=3)
               for key in protocol["methods"]]
    sample = dict(native_shape_yx=[10,20], native_mpp_xy=[.5,.5], source_region_type="ROI",
                  class_names={"1":"tumour"}, palette=[[220,45,65]])
    save(root/"RESULTS.json", dict(protocol=protocol, ranking=ranking,
        cases=[dict(case="example", dataset="test", sample=sample, methods=methods)]))
    selection = dict(cycle=3, survivors=survivors,
        results_sha256=digest(root/"RESULTS.json"), review_sha256=digest(root/"VISUAL_REVIEW.json"),
        protocol_sha256=digest(root/"PROTOCOL.json"))
    save(root/"SELECTION.json", selection)
    (root/"gallery.html").write_text('<title>Bora</title><h1>Results</h1>CYCLE AWAITING VISUAL REVIEW')


def test_completed_viewer_status_uses_signed_selection_and_is_idempotent(tmp_path):
    report_fixture(tmp_path)
    contact = digest(tmp_path/"cases/example/contact.jpg")
    first = publish_selection_status(tmp_path)
    second = publish_selection_status(tmp_path)
    assert first == second
    html = (tmp_path/"gallery.html").read_text()
    assert '3/20 CYCLES COMPLETE' in html
    assert 'Selected: F01, C17, C01, D08, C15' in html
    assert 'CYCLE AWAITING VISUAL REVIEW' not in html
    assert html.count('id="cycle-selection"') == 1
    assert digest(tmp_path/"cases/example/contact.jpg") == contact
    selected = (tmp_path/"selected.html").read_text()
    assert selected.count('<figure>') == 9
    assert 'REJECTED' not in selected
    assert 'contact.jpg' not in selected
    assert 'cases/example/F01/mask.png' in selected
    data = json.loads((tmp_path/"SELECTED_RESULTS.json").read_text())
    assert set(data['cases'][0]['methods']) == {'F01','C17','C01','D08','C15'}
    assert first['selected_gallery'] == 'selected.html'
    assert first['displayed_cases'] == 1


def test_completed_viewer_refuses_changed_scored_evidence(tmp_path):
    report_fixture(tmp_path)
    save(tmp_path/"RESULTS.json", dict(cases=["changed"]))
    with pytest.raises(ValueError, match="changed signed"):
        publish_selection_status(tmp_path)


def test_selected_calibrated_viewer_uses_new_cohort_labels_and_physical_windows(tmp_path,monkeypatch):
    from experiments.bora_cycles.selected_viewer import publish_selected
    from experiments.bora_cycles import selected_assets
    monkeypatch.setattr(selected_assets,'layers',lambda source,root,sample,methods:
        ({'reference':'selected-view/reference.png','coarse':'selected-view/coarse.png'},dict(case=sample['case'])))
    report_fixture(tmp_path)
    protocol=json.loads((tmp_path/'PROTOCOL.json').read_text())
    protocol.update(cycle=11,source='unused-in-this-html-test',benchmark_scope=dict(version='calibrated-tissues-v1'))
    save(tmp_path/'PROTOCOL.json',protocol)
    result=json.loads((tmp_path/'RESULTS.json').read_text());result['protocol']=protocol
    result['cases'][0]['sample'].update(native_shape_yx=[2560,2560],working_shape_yx=[1280,1280],
        native_mpp_xy=[.25,.25],working_mpp_xy=[.5,.5])
    for row in result['ranking']:
        row['strata']=dict(previous_cohort=dict(mean_iou=.6),new_tissue_datasets=dict(mean_iou=None))
    save(tmp_path/'RESULTS.json',result)
    selection=json.loads((tmp_path/'SELECTION.json').read_text());selection.update(cycle=11,
        results_sha256=digest(tmp_path/'RESULTS.json'),protocol_sha256=digest(tmp_path/'PROTOCOL.json'))
    save(tmp_path/'SELECTION.json',selection)
    publish_selected(tmp_path)
    html=(tmp_path/'selected.html').read_text()
    assert 'Previous cohort IoU' in html and 'New tissue datasets IoU' in html
    assert 'Original149 IoU' not in html
    assert '112.0 × 112.0 µm; 224.0 × 224.0 µm; 448.0 × 448.0 µm' in html
    assert html.count('class="scale-bar"')==9 and html.count('width:15.625%')==9
