import json

from solo.data.dataloader import save_weights
from solo.engine.proposals import rank_proposals_for_budget
from solo.engine.proposal_diagnostics import proposal_diagnostics, shape_multiplier_from_diagnostics
from solo.models.matcher import match_pt, prepare_weight_index
from solo.utils.bbox import _bbox_payload


def _weight(label, value, negative=False):
    vector = [value, value, value, value]
    return {
        "name": f"{label}_{value}",
        "annotation": {
            "label": "__negative__" if negative else label,
            "bbox": _bbox_payload((0, 0, 10, 10), 100, 100),
        },
        "negative": negative,
        "pt": [[value, value], [value, value]],
        "features": {
            "vector": vector,
            "mode": "multi",
            "channel_layout": {
                "gray": {"start": 0, "end": 4, "length": 4},
                "stats": {"start": 4, "end": 4, "length": 0},
            },
            "channel_stats": {"gray": {"min": value, "max": value, "mean": value, "range": 0.0, "variance": 0.0}},
        },
    }


def test_compact_weights_load_without_training_entries(tmp_path):
    path = tmp_path / "compact.json"
    save_weights(
        [_weight("target", 0.2), _weight("__negative__", 0.8, negative=True)],
        path,
        config={
            "feature_mode": "multi",
            "channels": ["gray"],
            "pt_size": 2,
            "prototype_count": 2,
            "structure_mode": "none",
        },
        compact=True,
        precision=4,
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["compact"] is True
    assert payload["weights"] == []
    assert payload["prototypes"]

    index = prepare_weight_index([path], accelerator="cpu")
    match = match_pt([0.2, 0.2, 0.2, 0.2], index, accelerator="cpu")

    assert index["entries"] == []
    assert match["positive_label"] == "target"


def test_compact_weights_can_keep_representative_exemplars(tmp_path):
    path = tmp_path / "compact_exemplars.json"
    save_weights(
        [_weight("target", 0.2), _weight("target", 0.35), _weight("__negative__", 0.8, negative=True)],
        path,
        config={
            "feature_mode": "multi",
            "channels": ["gray"],
            "pt_size": 2,
            "prototype_count": 1,
            "compact_exemplars": 1,
            "structure_mode": "none",
        },
        compact=True,
        precision=4,
        compact_exemplars=1,
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["weights"] == []
    assert any(item.get("exemplars") for item in payload["prototypes"])

    index = prepare_weight_index([path], accelerator="cpu")
    match = match_pt([0.2, 0.2, 0.2, 0.2], index, accelerator="cpu")

    assert index["entries"] == []
    assert match["positive_label"] == "target"


def test_compact_weights_can_keep_nearest_entries(tmp_path):
    path = tmp_path / "compact_entries.json"
    save_weights(
        [_weight("target", 0.2), _weight("target", 0.35), _weight("__negative__", 0.8, negative=True)],
        path,
        config={
            "feature_mode": "multi",
            "channels": ["gray"],
            "pt_size": 2,
            "prototype_count": 1,
            "compact_sample_limit": 2,
            "structure_mode": "none",
        },
        compact=True,
        precision=4,
        compact_exemplars=0,
        compact_sample_limit=2,
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["weights"] == []
    assert len(payload["compact_entries"]) == 2

    index = prepare_weight_index([path], accelerator="cpu")
    match = match_pt([0.2, 0.2, 0.2, 0.2], index, match_mode="nearest", accelerator="cpu")

    assert len(index["entries"]) == 2
    assert match["positive_label"] == "target"


def test_rank_proposals_prefers_objectness_and_prior():
    image_width = image_height = 100
    proposals = [
        {"bbox": _bbox_payload((0, 0, 80, 80), image_width, image_height), "proposal": "sliding", "objectness": 0.05},
        {"bbox": _bbox_payload((10, 10, 30, 30), image_width, image_height), "proposal": "objectness", "objectness": 0.80},
        {"bbox": _bbox_payload((40, 40, 58, 58), image_width, image_height), "proposal": "anchor", "objectness": 0.10},
    ]

    kept = rank_proposals_for_budget(proposals, [(20, 20)], max_proposals=1)

    assert kept[0]["bbox"]["width"] == 20
    assert kept[0]["proposal_rank_score"] > 0


def test_shape_diagnostics_penalize_poles_and_road_patches():
    pole = proposal_diagnostics(_bbox_payload((48, 0, 52, 90), 100, 100))
    road = proposal_diagnostics(_bbox_payload((10, 78, 90, 98), 100, 100))
    car_like = proposal_diagnostics(_bbox_payload((20, 55, 78, 78), 100, 100))

    assert pole["thin_vertical_score"] > 0.5
    assert shape_multiplier_from_diagnostics(pole) < 0.55
    assert road["road_score"] > 0.25
    assert car_like["vehicle_shape_score"] > 0.5
    assert shape_multiplier_from_diagnostics(car_like) > shape_multiplier_from_diagnostics(pole)


def test_rank_proposals_demotes_thin_vertical_candidates():
    image_width = image_height = 100
    pole = {"bbox": _bbox_payload((48, 0, 52, 90), image_width, image_height), "proposal": "edge_component", "objectness": 0.95}
    car_like = {"bbox": _bbox_payload((20, 55, 78, 78), image_width, image_height), "proposal": "anchor", "objectness": 0.30}
    pole["diagnostics"] = proposal_diagnostics(pole["bbox"], image_width=image_width, image_height=image_height)
    car_like["diagnostics"] = proposal_diagnostics(car_like["bbox"], image_width=image_width, image_height=image_height)

    kept = rank_proposals_for_budget([pole, car_like], [(58, 23)], max_proposals=1)

    assert kept[0]["bbox"]["width"] == 58
