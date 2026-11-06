"""Read TXT annotations and build YOLOv3 targets and COCO evaluation data."""

import math
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from .config import ANCHORS, CLASS_NAMES, COCO_CATEGORY_IDS, IMG_MEAN, IMG_STD, YOLO_IMG_DIM


# ============================================================================
# COCO Dataset
# ============================================================================
class CocoDetectorDataset(Dataset):
    """Return an image, targets, original dimensions, and normalized boxes."""

    def __init__(
        self,
        root,
        list_file,
        train=False,
        encode_target=True,
        image_size=YOLO_IMG_DIM,
    ):
        if image_size < 32 or image_size % 32 != 0:
            raise ValueError("image_size must be a positive multiple of 32.")
        self.root = Path(root)
        self.list_file = Path(list_file)
        self.train = train
        self.encode_target = encode_target
        self.image_size = image_size
        self.grid_sizes = tuple(image_size // stride for stride in (32, 16, 8))
        self.class_names = CLASS_NAMES
        self.category_ids = COCO_CATEGORY_IDS
        self.images, annotations = [], []
        self.boxes, self.labels, self.crowd_boxes = [], [], []
        image_ids = set()

        # Each line: image path followed by groups of x1 y1 x2 y2 class_id area iscrowd.
        for line_number, line in enumerate(self.list_file.read_text().splitlines(), 1):
            fields = line.split()
            if not fields:
                continue
            if (len(fields) - 1) % 7 != 0:
                raise ValueError(
                    f"{self.list_file.name}:{line_number}: each object needs 7 fields."
                )
            filename = fields[0]
            image_id = int(Path(filename).stem)
            if image_id in image_ids:
                raise ValueError(f"Duplicate image ID: {image_id}")
            image_ids.add(image_id)

            # Read image headers for original dimensions without decoding every image.
            with Image.open(self.root / filename) as image:
                width, height = image.size
            self.images.append({
                "id": image_id,
                "file_name": filename,
                "width": width,
                "height": height,
            })
            boxes, labels, crowd_boxes = [], [], []
            for start in range(1, len(fields), 7):
                x1, y1, x2, y2 = map(float, fields[start:start + 4])
                label = int(fields[start + 4])
                area = float(fields[start + 5])
                crowd = int(fields[start + 6])
                if (
                    not all(math.isfinite(value) for value in (x1, y1, x2, y2, area))
                    or x2 <= x1 or y2 <= y1 or area < 0
                    or not 0 <= label < len(CLASS_NAMES) or crowd not in (0, 1)
                ):
                    raise ValueError(f"{self.list_file.name}:{line_number}: invalid annotation.")

                # Preserve original boxes, annotation areas, and crowd flags for COCO evaluation.
                annotations.append({
                    "id": len(annotations) + 1,
                    "image_id": image_id,
                    "category_id": self.category_ids[label],
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": area,
                    "iscrowd": crowd,
                })
                box = [
                    max(0, x1) / width,
                    max(0, y1) / height,
                    min(width, x2) / width,
                    min(height, y2) / height,
                ]
                if box[2] <= box[0] or box[3] <= box[1]:
                    raise ValueError(f"{self.list_file.name}:{line_number}: box outside the image.")
                if crowd:
                    crowd_boxes.append(box)
                else:
                    boxes.append(box)
                    labels.append(label)

            self.boxes.append(np.asarray(boxes, dtype=np.float32).reshape(-1, 4))
            self.labels.append(np.asarray(labels, dtype=np.int64))
            self.crowd_boxes.append(np.asarray(crowd_boxes, dtype=np.float32).reshape(-1, 4))

        if not self.images or (encode_target and not annotations):
            raise ValueError("A non-empty annotated image list is required for encoded targets.")
        self.coco = {
            "info": {},
            "images": self.images,
            "annotations": annotations,
            "categories": [
                {"id": category_id, "name": name}
                for category_id, name in zip(self.category_ids, self.class_names)
            ],
        }
        self.flat_anchors = np.asarray(ANCHORS, dtype=np.float32).reshape(-1, 2)

    def __len__(self):
        return len(self.images)

    def encode(self, boxes, labels, crowd_boxes):
        """Build (S, S, 3, 25) targets at strides 32/16/8.

        Each target stores [tx, ty, w, h, obj, one-hot classes].
        tx/ty are cell offsets; w/h are image fractions; obj=-1 marks ignored slots.
        """
        targets = [
            np.zeros((grid, grid, 3, 5 + len(CLASS_NAMES)), dtype=np.float32)
            for grid in self.grid_sizes
        ]

        # Ignore grid centers inside crowd regions instead of assigning instance targets.
        for grid, target in zip(self.grid_sizes, targets):
            centers = (np.arange(grid) + 0.5) / grid
            for x1, y1, x2, y2 in crowd_boxes:
                inside = (
                    (centers[:, None] >= y1)
                    & (centers[:, None] <= y2)
                    & (centers[None, :] >= x1)
                    & (centers[None, :] <= x2)
                )
                target[..., 4][inside] = -1

        # Rank anchors by size IoU and assign each object to an unoccupied slot.
        for box, label in zip(boxes, labels):
            center = ((box[:2] + box[2:]) / 2).clip(0, 1 - 1e-7)
            size = box[2:] - box[:2]
            intersection = np.minimum(size, self.flat_anchors).prod(axis=1)
            ious = intersection / (size.prod() + self.flat_anchors.prod(axis=1) - intersection)

            for anchor_id in np.argsort(-ious, kind="stable"):
                scale, anchor = divmod(int(anchor_id), 3)
                position = center * self.grid_sizes[scale]
                grid_x, grid_y = position.astype(int)
                target = targets[scale][grid_y, grid_x, anchor]
                if target[4] == 1:
                    continue
                target[:2] = position - [grid_x, grid_y]
                target[2:4] = size
                target[4] = 1
                target[5 + int(label)] = 1
                break
            else:
                raise ValueError("物件中心的九個 anchor slots 已滿，需調整指派規則。")

        assert sum(int((target[..., 4] == 1).sum()) for target in targets) == len(boxes)
        return [torch.from_numpy(target) for target in targets]

    def __getitem__(self, index):
        record = self.images[index]
        with Image.open(self.root / record["file_name"]) as source:
            image = source.convert("RGB")

        boxes = self.boxes[index].copy()
        crowd_boxes = self.crowd_boxes[index].copy()
        labels = self.labels[index]

        # Apply horizontal flips during training, updating the image and all boxes together.
        if self.train and torch.rand(()) < 0.5:
            image = TF.hflip(image)
            for regions in (boxes, crowd_boxes):
                regions[:, [0, 2]] = 1 - regions[:, [2, 0]]

        # Resize the input and normalize RGB channels using ImageNet statistics.
        image = TF.resize(image, [self.image_size, self.image_size])
        image = TF.normalize(TF.to_tensor(image), IMG_MEAN, IMG_STD)

        encoded = self.encode(boxes, labels, crowd_boxes) if self.encode_target else None
        metadata = {
            "image_id": record["id"],
            "original_size": (record["width"], record["height"]),
            "boxes": boxes,
            "labels": labels,
            "crowd_boxes": crowd_boxes,
        }
        return image, encoded, metadata


# ============================================================================
# Batch Collation
# ============================================================================
def collate_fn(batch):
    """Stack images and targets into batches; keep instance metadata per image."""
    images, targets, metadata = zip(*batch)
    encoded = None
    if targets[0] is not None:
        encoded = [torch.stack([target[scale] for target in targets]) for scale in range(3)]

    return torch.stack(images), encoded, list(metadata)
