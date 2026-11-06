"""Compute validation AP in original coordinates with the official COCO bbox evaluator."""

import torch
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm.auto import tqdm


# ============================================================================
# COCO Bounding Box Evaluation
# ============================================================================
@torch.inference_mode()
def evaluate(
    model,
    loader,
    annotation_data,
    category_ids,
    conf_thres=0.01,
    use_amp=False,
):
    """Restore predictions to original coordinates and return overall and per-class AP."""
    model.eval()
    device = next(model.parameters()).device
    predictions = []

    # Map model labels 0-19 to original category IDs before calling COCOeval.
    for images, _, metadata in tqdm(loader, desc="COCO evaluation", leave=False):
        with torch.autocast(device.type, enabled=use_amp and device.type == "cuda"):
            detections = model.inference(images.to(device), conf_thres=conf_thres)
        for detection, record in zip(detections, metadata):
            if detection is None:
                continue
            width, height = record["original_size"]
            for cx, cy, w, h, objectness, probability, label in detection.cpu().tolist():
                x1, y1 = max(0, (cx - w / 2) * width), max(0, (cy - h / 2) * height)
                x2, y2 = min(width, (cx + w / 2) * width), min(height, (cy + h / 2) * height)
                if x2 <= x1 or y2 <= y1:
                    continue
                predictions.append(
                    {
                        "image_id": record["image_id"],
                        "category_id": category_ids[int(label)],
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": objectness * probability,
                    }
                )

    # Preserve original crowd flags and areas for the official evaluator.
    ground_truth = COCO()
    ground_truth.dataset = {
        **annotation_data,
        "annotations": [dict(annotation) for annotation in annotation_data["annotations"]],
    }
    ground_truth.createIndex()
    if predictions:
        detected = ground_truth.loadRes(predictions)
    else:
        detected = COCO()
        detected.dataset = {
            "images": ground_truth.dataset["images"],
            "annotations": [],
            "categories": ground_truth.dataset["categories"],
        }
        detected.createIndex()

    evaluator = COCOeval(ground_truth, detected, "bbox")
    evaluator.params.catIds = list(category_ids)
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    # The first six stats contain AP, AP50, AP75, and AP for three object sizes.
    metrics = dict(
        zip(
            ("AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large"),
            map(float, evaluator.stats[:6]),
        )
    )
    metrics["images"] = len(ground_truth.imgs)
    metrics["detections"] = len(predictions)

    # Read per-class results in the order of COCOeval catIds.
    names = {category["id"]: category["name"] for category in ground_truth.dataset["categories"]}
    per_class = {}
    for index, category_id in enumerate(evaluator.params.catIds):
        precision = evaluator.eval["precision"][:, :, index, 0, -1]
        valid = precision[precision >= 0]
        per_class[names[category_id]] = float(np.mean(valid)) if len(valid) else None
    metrics["per_class_ap"] = per_class
    return metrics
