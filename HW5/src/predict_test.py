"""Generate result.csv from trained YOLOv3 weights and the test image list."""

import argparse
import csv
import json
from pathlib import Path

from PIL import Image
import torch
from torchvision.transforms import functional as TF
from tqdm.auto import tqdm

from .config import CLASS_NAMES, IMG_MEAN, IMG_STD, MODEL_NAME, YOLO_IMG_DIM
from .yolo import getODmodel


# ============================================================================
# Image Preprocessing & Box Conversion
# ============================================================================
def preprocess_image(image_path, image_size=YOLO_IMG_DIM):
    """Use the same RGB conversion, resizing, and normalization as validation."""
    with Image.open(image_path) as source:
        image = source.convert("RGB")
        original_size = image.size

    image = TF.resize(image, [image_size, image_size])
    image = TF.normalize(TF.to_tensor(image), IMG_MEAN, IMG_STD)
    return image, original_size


def convert_detections(detections, original_size):
    """Convert normalized cxcywh to original-image xyxy pixels, keeping float coordinates."""
    if detections is None:
        return []

    image_width, image_height = original_size
    boxes = []
    for cx, cy, width, height, objectness, probability, label in detections[:100].cpu().tolist():
        x1 = max(0.0, (cx - width / 2) * image_width)
        y1 = max(0.0, (cy - height / 2) * image_height)
        x2 = min(float(image_width), (cx + width / 2) * image_width)
        y2 = min(float(image_height), (cy + height / 2) * image_height)
        if x2 <= x1 or y2 <= y1:
            continue

        class_name = CLASS_NAMES[int(label)]
        confidence = objectness * probability
        boxes.append([class_name, confidence, x1, y1, x2, y2])

    return boxes


# ============================================================================
# Test Inference & CSV Output
# ============================================================================
@torch.inference_mode()
def predict_test(
    weights_path=Path("checkpoints/best_detector.pth"),
    data_root=Path("dataset"),
    output_path=Path("result.csv"),
    batch_size=16,
    conf_thres=0.01,
    nms_thres=0.4,
    device="auto",
    model_name=MODEL_NAME,
    test_list=None,
    image_size=YOLO_IMG_DIM,
):
    """Read test.txt, load the best weights, and write Kaggle predictions in batches."""
    if batch_size < 1 or not 0 <= conf_thres <= 1 or not 0 <= nms_thres <= 1:
        raise ValueError("batch_size must be positive and thresholds must be within [0, 1].")

    if image_size < 32 or image_size % 32 != 0:
        raise ValueError("image_size must be a positive multiple of 32.")

    data_root = Path(data_root)
    output_path = Path(output_path)
    test_list = data_root / "test.txt" if test_list is None else Path(test_list)
    image_names = [line.strip() for line in test_list.read_text().splitlines() if line.strip()]
    image_ids = [Path(name).name for name in image_names]
    if not image_names or len(set(image_ids)) != len(image_ids):
        raise ValueError("The test list must contain non-empty, unique image IDs.")

    if str(device) == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    # Load the trained checkpoint without downloading additional backbone weights.
    model = getODmodel(pretrained=False, model_name=model_name)
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device).eval()

    # Write batches to a temporary file and replace result.csv after completion.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "prediction_list"])

        for start in tqdm(range(0, len(image_names), batch_size), desc="Predict test"):
            names = image_names[start:start + batch_size]
            samples = [preprocess_image(data_root / name, image_size) for name in names]
            images = torch.stack([image for image, _ in samples]).to(device)
            detections = model.inference(images, conf_thres=conf_thres, nms_thres=nms_thres)

            for name, detection, (_, original_size) in zip(names, detections, samples):
                boxes = convert_detections(detection, original_size)
                # Write predictions as a JSON array; use [] when no objects are detected.
                writer.writerow([Path(name).name, json.dumps(boxes, allow_nan=False)])

    temporary.replace(output_path)
    print(f"Wrote {len(image_names):,} images to {output_path}")
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights",
        dest="weights_path",
        type=Path,
        default=Path("checkpoints/best_detector.pth"),
    )
    parser.add_argument("--data-root", type=Path, default=Path("dataset"))
    parser.add_argument("--output", dest="output_path", type=Path, default=Path("result.csv"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--conf-thres", type=float, default=0.01)
    parser.add_argument("--nms-thres", type=float, default=0.4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--test-list", type=Path)
    parser.add_argument("--image-size", type=int, default=YOLO_IMG_DIM)
    return parser.parse_args()


if __name__ == "__main__":
    predict_test(**vars(parse_args()))
