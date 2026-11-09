"""Model and image preprocessing settings."""

# ============================================================================
# Classes
# ============================================================================
CLASS_NAMES = (
    "chair",
    "couch",
    "dining table",
    "tv",
    "laptop",
    "mouse",
    "keyboard",
    "remote",
    "cell phone",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "book",
    "clock",
    "vase",
    "potted plant",
)
NUM_CLASSES = len(CLASS_NAMES)

# Map TXT class IDs 0-19 to the original category IDs for COCO evaluation.
COCO_CATEGORY_IDS = (
    62, 63, 67, 72, 73, 74, 76, 75, 77, 44,
    46, 47, 48, 49, 50, 51, 84, 85, 86, 64,
)


# ============================================================================
# Backbone & Image Preprocessing
# ============================================================================
MODEL_NAME = "convnext_large.fb_in1k"
YOLO_IMG_DIM = 416
GRID_SIZES = tuple(YOLO_IMG_DIM // stride for stride in (32, 16, 8))
IMG_MEAN = (0.485, 0.456, 0.406)
IMG_STD = (0.229, 0.224, 0.225)


# ============================================================================
# Anchors
# ============================================================================
# Three scales, from coarse to fine, with three anchors per scale.
# Normalize anchor widths/heights using the 416-pixel reference image size.
ANCHOR_REFERENCE_DIM = 416
ANCHOR_PIXELS = (
    ((116, 90), (156, 198), (373, 326)),
    ((30, 61), (62, 45), (59, 119)),
    ((10, 13), (16, 30), (33, 23)),
)
ANCHORS = tuple(
    tuple((width / ANCHOR_REFERENCE_DIM, height / ANCHOR_REFERENCE_DIM) for width, height in scale)
    for scale in ANCHOR_PIXELS
)
