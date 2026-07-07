from __future__ import annotations

import math
from typing import Any

from solo.config import *
from solo.utils.bbox import *
from solo.core.features import *
from solo.engine.proposal_diagnostics import proposal_diagnostics, proposal_rank_bonus
from solo.utils.cv_image import Image

def proposal_templates_from_prior(
    prior: dict[str, Any] | None,
    image_width: int,
    image_height: int,
) -> list[tuple[int, int]]:
    if not prior or not prior.get("enabled"):
        return []
    templates = []
    scale_pairs = (
        (0.60, 0.60),
        (0.75, 0.75),
        (0.90, 0.90),
        (1.00, 1.00),
        (1.15, 1.15),
        (1.35, 1.35),
        (1.60, 1.60),
        (1.20, 0.85),
        (0.85, 1.20),
        (1.45, 0.75),
        (0.75, 1.45),
    )

    def add_scaled(width: float, height: float, normalized: bool) -> None:
        if width <= 0 or height <= 0:
            return
        for scale_width, scale_height in scale_pairs:
            if normalized:
                scaled_width = round(image_width * width * scale_width)
                scaled_height = round(image_height * height * scale_height)
            else:
                scaled_width = round(width * scale_width)
                scaled_height = round(height * scale_height)
            scaled_width = max(4, min(image_width, scaled_width))
            scaled_height = max(4, min(image_height, scaled_height))
            templates.append((scaled_width, scaled_height))

    for item in prior.get("normalized_templates", []):
        add_scaled(float(item.get("width", 0)), float(item.get("height", 0)), normalized=True)
    for item in prior.get("templates", []):
        add_scaled(float(item.get("width", 0)), float(item.get("height", 0)), normalized=False)

    unique = list(dict.fromkeys(templates))
    if len(unique) <= 64:
        return unique

    bucket_size = max(4, round(min(image_width, image_height) / 80))
    buckets: dict[tuple[int, int], tuple[int, int]] = {}
    for template_width, template_height in sorted(unique, key=lambda item: item[0] * item[1]):
        buckets.setdefault(
            (round(template_width / bucket_size), round(template_height / bucket_size)),
            (template_width, template_height),
        )
    deduped = list(buckets.values())
    if len(deduped) <= 64:
        return deduped
    indexes = [round(index * (len(deduped) - 1) / 63) for index in range(64)]
    return [deduped[index] for index in sorted(set(indexes))]

def proposal_priority_rank(proposal_name: str) -> int:
    if proposal_name.endswith("_vehicle_expand"):
        return 1
    return {
        "anchor_refined": 0,
        "objectness": 1,
        "color": 1,
        "edge_component": 2,
        "horizontal_body": 2,
        "anchor": 3,
        "sliding": 4,
    }.get(proposal_name, 5)

def _proposal_objectness_score(proposal: dict[str, Any]) -> float:
    try:
        return float(proposal.get("objectness", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0

def proposal_priority(proposal: dict[str, Any]) -> tuple[int, float, int]:
    priority = proposal_priority_rank(str(proposal.get("proposal")))
    bbox = proposal.get("bbox", {})
    area = int(bbox.get("width", 0)) * int(bbox.get("height", 0))
    return priority, -_proposal_objectness_score(proposal), area

def proposal_bucket_key(proposal: dict[str, Any], grid: int = 4) -> tuple[int, int, int, int]:
    bbox = proposal.get("bbox", {})
    normalized = bbox.get("normalized") or {}
    center_x = (float(normalized.get("x1", 0.0)) + float(normalized.get("x2", 0.0))) / 2
    center_y = (float(normalized.get("y1", 0.0)) + float(normalized.get("y2", 0.0))) / 2
    cell_x = max(0, min(grid - 1, int(center_x * grid)))
    cell_y = max(0, min(grid - 1, int(center_y * grid)))
    area = max(1.0, float(bbox.get("width", 0)) * float(bbox.get("height", 0)))
    scale_bucket = max(0, min(12, int(math.log2(math.sqrt(area)))))
    priority = proposal_priority_rank(str(proposal.get("proposal")))
    return cell_y, cell_x, scale_bucket, priority

def diverse_proposal_cap(proposals: list[dict[str, Any]], max_proposals: int) -> list[dict[str, Any]]:
    if max_proposals <= 0 or len(proposals) <= max_proposals:
        return proposals
    buckets: dict[tuple[int, int, int, int], list[dict[str, Any]]] = {}
    for proposal in proposals:
        buckets.setdefault(proposal_bucket_key(proposal), []).append(proposal)
    keys = list(buckets)
    kept = []
    while len(kept) < max_proposals:
        progressed = False
        for key in keys:
            items = buckets[key]
            if not items:
                continue
            kept.append(items.pop(0))
            progressed = True
            if len(kept) >= max_proposals:
                break
        if not progressed:
            break
    return kept

def dedupe_proposals(
    proposals: list[dict[str, Any]],
    iou_threshold: float = DEFAULT_PROPOSAL_DEDUPE_IOU,
    max_proposals: int = DEFAULT_MAX_PROPOSALS,
) -> list[dict[str, Any]]:
    if iou_threshold <= 0:
        return diverse_proposal_cap(proposals, max_proposals)
    kept: list[dict[str, Any]] = []
    bucket_counts: dict[tuple[int, int, int, int], int] = {}
    bucket_limit = max(2, math.ceil(max(1, max_proposals) / 32)) if max_proposals else 0
    for proposal in sorted(proposals, key=proposal_priority):
        if max_proposals:
            bucket_key = proposal_bucket_key(proposal)
            if bucket_counts.get(bucket_key, 0) >= bucket_limit:
                continue
        if all(bbox_iou(proposal["bbox"], existing["bbox"]) <= iou_threshold for existing in kept):
            kept.append(proposal)
            if max_proposals:
                bucket_counts[bucket_key] = bucket_counts.get(bucket_key, 0) + 1
    return diverse_proposal_cap(kept, max_proposals)

def _template_prior_score(bbox: dict[str, Any], templates: list[tuple[int, int]]) -> float:
    if not templates:
        return 0.0
    width = max(1.0, float(bbox.get("width", 0) or 0))
    height = max(1.0, float(bbox.get("height", 0) or 0))
    best = 0.0
    for template_width, template_height in templates:
        if template_width <= 0 or template_height <= 0:
            continue
        log_w = abs(math.log(width / max(1.0, template_width)))
        log_h = abs(math.log(height / max(1.0, template_height)))
        best = max(best, math.exp(-(log_w + log_h)))
    return best

def _score_proposal_for_budget(
    proposal: dict[str, Any],
    templates: list[tuple[int, int]],
) -> float:
    source_rank = proposal_priority_rank(str(proposal.get("proposal")))
    source_score = max(0.0, 1.0 - source_rank * 0.11)
    objectness = _proposal_objectness_score(proposal)
    prior = _template_prior_score(proposal.get("bbox", {}), templates)
    refined_bonus = 0.10 if proposal.get("refined") else 0.0
    shape_bonus = proposal_rank_bonus(proposal.get("diagnostics"))
    return source_score * 0.42 + objectness * 0.20 + prior * 0.26 + refined_bonus + shape_bonus

def rank_proposals_for_budget(
    proposals: list[dict[str, Any]],
    templates: list[tuple[int, int]],
    max_proposals: int,
) -> list[dict[str, Any]]:
    if max_proposals <= 0 or len(proposals) <= max_proposals:
        return proposals
    ranked = []
    for proposal in proposals:
        item = dict(proposal)
        score = _score_proposal_for_budget(item, templates)
        item["proposal_rank_score"] = round(score, 6)
        ranked.append(item)
    return diverse_proposal_cap(
        sorted(ranked, key=lambda item: item.get("proposal_rank_score", 0.0), reverse=True),
        max_proposals,
    )

def attach_proposal_diagnostics(
    proposals: list[dict[str, Any]],
    maps: dict[str, Any] | None,
    image_width: int,
    image_height: int,
) -> list[dict[str, Any]]:
    if not proposals:
        return proposals
    diagnosed = []
    for proposal in proposals:
        item = dict(proposal)
        item["diagnostics"] = proposal_diagnostics(
            item.get("bbox", {}),
            maps=maps,
            image_width=image_width,
            image_height=image_height,
        )
        diagnosed.append(item)
    return diagnosed

def expanded_vehicle_proposals(
    proposals: list[dict[str, Any]],
    image_width: int,
    image_height: int,
    maps: dict[str, Any] | None,
    max_items: int = 120,
) -> list[dict[str, Any]]:
    if not proposals or max_items <= 0:
        return []
    candidates = sorted(
        proposals,
        key=lambda item: float((item.get("diagnostics") or {}).get("partial_vehicle_score", 0.0)),
        reverse=True,
    )
    expanded: list[dict[str, Any]] = []
    for proposal in candidates[:max_items]:
        diagnostics = proposal.get("diagnostics") or {}
        if float(diagnostics.get("partial_vehicle_score", 0.0) or 0.0) < 0.18:
            continue
        bbox = _bbox_tuple(proposal["bbox"])
        x1, y1, x2, y2 = bbox
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        if height > width * 1.15:
            continue
        for scale_w, scale_h in ((1.45, 1.08), (1.80, 1.14), (2.25, 1.20)):
            candidate = _clamp_bbox(
                (
                    (x1 + x2) / 2 - width * scale_w / 2,
                    (y1 + y2) / 2 - height * scale_h / 2,
                    (x1 + x2) / 2 + width * scale_w / 2,
                    (y1 + y2) / 2 + height * scale_h / 2,
                ),
                image_width,
                image_height,
            )
            if candidate is None or candidate == bbox:
                continue
            item = dict(proposal)
            item["bbox"] = _bbox_payload(candidate, image_width, image_height)
            item["proposal"] = f"{proposal.get('proposal', 'proposal')}_vehicle_expand"
            item["proposal_expanded_from"] = proposal.get("bbox")
            item["expanded"] = True
            item["diagnostics"] = proposal_diagnostics(
                item["bbox"],
                maps=maps,
                image_width=image_width,
                image_height=image_height,
            )
            expanded.append(item)
    return expanded

def prefilter_proposals_for_dedupe(
    proposals: list[dict[str, Any]],
    templates: list[tuple[int, int]],
    max_proposals: int,
) -> list[dict[str, Any]]:
    if max_proposals <= 0:
        return proposals
    prefilter_limit = max(max_proposals * 4, max_proposals + 80)
    if len(proposals) <= prefilter_limit:
        return proposals
    return rank_proposals_for_budget(proposals, templates, prefilter_limit)

def proposal_bbox_area(bbox: tuple[int, int, int, int]) -> int:
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])

def refined_anchor_proposals(
    image: Image.Image,
    proposals: list[dict[str, Any]],
    min_area: int,
    max_area_ratio: float,
    min_box_size: int,
    max_box_size: int,
    max_aspect_ratio: float,
    proposal_dedupe_iou: float,
    max_refined: int,
    maps: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if not proposals:
        return []
    width, height = image.size
    source = [
        proposal
        for proposal in dedupe_proposals(proposals, proposal_dedupe_iou, max_refined)
        if str(proposal.get("proposal")) == "anchor"
    ]
    refined: list[dict[str, Any]] = []
    for item in source:
        original_bbox = _bbox_tuple(item["bbox"])
        candidate = refine_proposal_bbox(image, original_bbox, min_box_size=min_box_size, maps=maps)
        if candidate is None or candidate == original_bbox:
            continue
        original_area = proposal_bbox_area(original_bbox)
        candidate_area = proposal_bbox_area(candidate)
        if original_area <= 0 or candidate_area <= 0:
            continue
        area_ratio = candidate_area / original_area
        if area_ratio < 0.30 or area_ratio > 2.20:
            continue
        original_center_x = (original_bbox[0] + original_bbox[2]) / 2
        original_center_y = (original_bbox[1] + original_bbox[3]) / 2
        candidate_center_x = (candidate[0] + candidate[2]) / 2
        candidate_center_y = (candidate[1] + candidate[3]) / 2
        original_width = max(1, original_bbox[2] - original_bbox[0])
        original_height = max(1, original_bbox[3] - original_bbox[1])
        if abs(candidate_center_x - original_center_x) > original_width * 0.40:
            continue
        if abs(candidate_center_y - original_center_y) > original_height * 0.40:
            continue
        relaxed_aspect = max(max_aspect_ratio, 10.0) if max_aspect_ratio else 0.0
        if not _proposal_passes_shape_filters(
            candidate,
            width,
            height,
            min_area,
            max_area_ratio,
            min_box_size,
            max_box_size,
            relaxed_aspect,
        ):
            continue
        source_payload = item["bbox"]
        candidate_payload = _bbox_payload(candidate, width, height)
        refined_item = dict(item)
        refined_item["bbox"] = candidate_payload
        refined_item["proposal"] = "anchor_refined"
        refined_item["proposal_refined_from"] = source_payload
        refined_item["refined"] = True
        refined_item["refine_iou_with_source"] = round(bbox_iou(candidate_payload, source_payload), 6)
        refined.append(refined_item)
    return refined

def generate_proposals(
    image: Image.Image,
    proposal: str = "color",
    min_area: int = 80,
    max_area_ratio: float = 0.15,
    expand: float = 1.1,
    min_box_size: int = 6,
    max_box_size: int = 0,
    max_aspect_ratio: float = 6.0,
    window_sizes: list[int] | None = None,
    window_ratios: list[float] | None = None,
    stride_ratio: float = 0.5,
    bbox_prior: dict[str, Any] | None = None,
    proposal_dedupe_iou: float = DEFAULT_PROPOSAL_DEDUPE_IOU,
    max_proposals: int = DEFAULT_MAX_PROPOSALS,
    edge_proposals: bool = DEFAULT_EDGE_PROPOSALS,
    body_proposals: bool = DEFAULT_BODY_PROPOSALS,
    proposal_refine: bool = DEFAULT_PROPOSAL_REFINE,
    proposal_maps: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if proposal not in {"color", "sliding", "both"}:
        raise ValueError("proposal must be 'color', 'sliding', or 'both'")
    width, height = image.size
    maps = proposal_maps if proposal_refine else None
    if maps is None and proposal_refine:
        maps = context_maps_for_image(image)
    prior_templates = proposal_templates_from_prior(bbox_prior, width, height)
    anchor_position_cap = 100 if max_proposals > 0 else 160
    anchors = (
        anchor_proposals(
            image,
            prior_templates,
            stride_ratio=stride_ratio,
            max_positions_per_template=anchor_position_cap,
        )
        if prior_templates
        else []
    )
    proposals: list[dict[str, Any]] = []
    if proposal == "color":
        proposals.extend(
            color_proposals(
                image,
                min_area=min_area,
                max_area_ratio=max_area_ratio,
                expand=expand,
                min_box_size=min_box_size,
                max_box_size=max_box_size,
                max_aspect_ratio=max_aspect_ratio,
            )
        )
    elif proposal == "sliding":
        proposals.extend(
            sliding_proposals(
                image,
                window_sizes or [32, 48, 64],
                window_ratios=window_ratios,
                stride_ratio=stride_ratio,
            )
        )
    elif proposal == "both":
        proposals.extend(
            color_proposals(
                image,
                min_area=min_area,
                max_area_ratio=max_area_ratio,
                expand=expand,
                min_box_size=min_box_size,
                max_box_size=max_box_size,
                max_aspect_ratio=max_aspect_ratio,
            )
        )
        proposals.extend(
            sliding_proposals(
                image,
                window_sizes or [32, 48, 64],
                window_ratios=window_ratios,
                stride_ratio=stride_ratio,
            )
        )

    if edge_proposals:
        edge_area_ratio = max(max_area_ratio, 0.25) if max_area_ratio > 0 else 0.0
        proposals.extend(
            edge_component_proposals(
                image,
                min_area=min_area,
                max_area_ratio=edge_area_ratio,
                expand=max(expand, 1.08),
                min_box_size=min_box_size,
                max_box_size=max_box_size,
                max_aspect_ratio=max(max_aspect_ratio, 8.0) if max_aspect_ratio else 0.0,
                maps=None,
            )
        )
    if body_proposals:
        body_area_ratio = max(max_area_ratio, 0.35) if max_area_ratio > 0 else 0.0
        proposals.extend(
            horizontal_body_proposals(
                image,
                min_area=min_area,
                max_area_ratio=body_area_ratio,
                expand=max(expand, 1.05),
                min_box_size=min_box_size,
                max_box_size=max_box_size,
                max_aspect_ratio=max(max_aspect_ratio, 10.0) if max_aspect_ratio else 0.0,
                maps=None,
            )
        )

    combined = attach_proposal_diagnostics(anchors + proposals, None, width, height)
    if proposal_refine and anchors:
        refine_limit = max(32, min(max_proposals // 3, 160)) if max_proposals > 0 else 160
        refined = refined_anchor_proposals(
            image,
            anchors,
            min_area=min_area,
            max_area_ratio=max(max_area_ratio, 0.35) if max_area_ratio > 0 else 0.0,
            min_box_size=min_box_size,
            max_box_size=max_box_size,
            max_aspect_ratio=max_aspect_ratio,
            proposal_dedupe_iou=proposal_dedupe_iou,
            max_refined=refine_limit,
            maps=maps,
        )
        combined = attach_proposal_diagnostics(refined, None, width, height) + combined

    budgeted = prefilter_proposals_for_dedupe(combined, prior_templates, max_proposals)
    deduped = dedupe_proposals(budgeted, proposal_dedupe_iou, 0)
    diagnostic_limit = max(max_proposals * 2, max_proposals + 80) if max_proposals > 0 else 0
    diagnostic_source = rank_proposals_for_budget(deduped, prior_templates, diagnostic_limit)
    diagnostics_maps = maps or context_maps_for_image(image)
    diagnosed = attach_proposal_diagnostics(diagnostic_source, diagnostics_maps, width, height)
    expanded = expanded_vehicle_proposals(
        diagnosed,
        width,
        height,
        diagnostics_maps,
        max_items=max(12, min(48, max_proposals // 4)) if max_proposals > 0 else 48,
    )
    final_candidates = dedupe_proposals(expanded + diagnosed, proposal_dedupe_iou, 0)
    return rank_proposals_for_budget(final_candidates, prior_templates, max_proposals)

__all__ = [
    'proposal_templates_from_prior',
    'proposal_priority_rank',
    'proposal_priority',
    'proposal_bucket_key',
    'diverse_proposal_cap',
    'dedupe_proposals',
    '_template_prior_score',
    '_score_proposal_for_budget',
    'rank_proposals_for_budget',
    'prefilter_proposals_for_dedupe',
    'proposal_bbox_area',
    'refined_anchor_proposals',
    'generate_proposals',
]
