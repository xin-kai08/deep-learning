"""Three-scale YOLOv3 with an ImageNet backbone, feature fusion, and box decoding."""

import torch
import torch.nn as nn
import timm
from torchvision.ops import batched_nms

from .config import ANCHORS, MODEL_NAME, NUM_CLASSES


# ============================================================================
# Backbone
# ============================================================================
class Backbone(nn.Module):
    """Return timm backbone feature maps at strides 32, 16, and 8, in that order."""

    def __init__(self, model_name=MODEL_NAME, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
        )
        self.channels = tuple(reversed(self.backbone.feature_info.channels()[-3:]))
        reductions = tuple(reversed(self.backbone.feature_info.reduction()[-3:]))
        if reductions != (32, 16, 8):
            raise ValueError(f"Backbone must provide stride 32/16/8, got {reductions}")

    def forward(self, x):
        features = self.backbone(x)
        return features[-1], features[-2], features[-3]


# ============================================================================
# Neck & Prediction Head
# ============================================================================
class ConvBlock(nn.Module):
    """Convolution -> BatchNorm -> LeakyReLU."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))


class YOLOv3Head(nn.Module):
    """Produce three prediction scales through upsampling and feature concatenation.

    Each output has shape (N, S, S, 3 * (5 + C)); 20 classes require 75 channels.
    """

    def __init__(
        self,
        num_classes=NUM_CLASSES,
        num_anchors=3,
        in_channels=(1024, 512, 256),
    ):
        super().__init__()
        self.output_channels = num_anchors * (5 + num_classes)
        coarse_channels, medium_channels, fine_channels = in_channels

        ########################################################################
        # TODO 1: Complete the YOLOv3 head with three scales of convolutional layers.
        ########################################################################
        # Scale 1: stride 32 (13x13 at 416), for larger objects.
        self.scale1_conv = nn.Sequential(
            ...
        )
        self.scale1_detect_conv = nn.Sequential(
            ...
        )
        self.scale_13_upsample = nn.Sequential(
            ...
        )

        # Scale 2: concatenate upsampled features with stride-16 backbone features.
        self.scale2_conv = nn.Sequential(
            ...
        )
        self.scale2_detect_conv = nn.Sequential(
            ...
        )
        self.scale_26_upsample = nn.Sequential(
            ...
        )

        # Scale 3: stride 8 (52x52 at 416), preserving spatial detail for small objects.
        self.scale3_conv = nn.Sequential(
            ...
        )
        self.scale3_detect_conv = nn.Sequential(
            ...
        )
        ########################################################################
        # End of your code
        ########################################################################

    def forward(self, features):
        feat_13, feat_26, feat_52 = features

        # Scale 1: 13x13。
        x1 = self.scale1_conv(feat_13)
        pred_13 = self.scale1_detect_conv(x1)

        # Scale 2: 26x26。
        x1_up = self.scale_13_upsample(x1)
        x2 = self.scale2_conv(torch.cat([x1_up, feat_26], dim=1))
        pred_26 = self.scale2_detect_conv(x2)

        # Scale 3: 52x52。
        x2_up = self.scale_26_upsample(x2)
        x3 = self.scale3_conv(torch.cat([x2_up, feat_52], dim=1))
        pred_52 = self.scale3_detect_conv(x3)

        # Output raw logits; apply sigmoid/softmax during loss computation or inference.
        return (
            pred_13.permute(0, 2, 3, 1).contiguous(),
            pred_26.permute(0, 2, 3, 1).contiguous(),
            pred_52.permute(0, 2, 3, 1).contiguous(),
        )


# ============================================================================
# Non-Maximum Suppression & IoU
# ============================================================================
def non_max_suppression(
    prediction,
    conf_thres=0.3,
    nms_thres=0.4,
    max_detections=100,
):
    """
    Apply class-wise NMS to center-format predictions of shape (N, boxes, 5 + C).

    Return one (K, 7) tensor or None per image, with each row containing:
    [cx, cy, w, h, object_confidence, class_confidence, class_id].
    """
    output = [None for _ in range(len(prediction))]
    for image_index, image_pred in enumerate(prediction):
        class_conf, class_pred = image_pred[:, 5:].max(dim=1)
        combined_conf = image_pred[:, 4] * class_conf
        valid = combined_conf >= conf_thres
        image_pred = image_pred[valid]
        class_conf = class_conf[valid]
        class_pred = class_pred[valid]
        if len(image_pred) == 0:
            continue

        # Clip boxes to image bounds so NMS and evaluation use the same geometry.
        boxes = torch.cat(
            [
                image_pred[:, :2] - image_pred[:, 2:4] / 2,
                image_pred[:, :2] + image_pred[:, 2:4] / 2,
            ],
            dim=1,
        ).clamp(0, 1)
        scores = image_pred[:, 4] * class_conf
        keep = batched_nms(boxes, scores, class_pred, nms_thres)[:max_detections]
        output[image_index] = torch.cat(
            [
                image_pred[keep, :5],
                class_conf[keep, None],
                class_pred[keep, None].float(),
            ],
            dim=1,
        )

    return output


def bbox_iou(box1, box2):
    """Pairwise IoU of normalized cxcywh boxes, returning an (N, M) tensor."""
    left1 = box1[:, :2] - box1[:, 2:] / 2
    right1 = box1[:, :2] + box1[:, 2:] / 2
    left2 = box2[:, :2] - box2[:, 2:] / 2
    right2 = box2[:, :2] + box2[:, 2:] / 2
    intersection_size = (
        torch.minimum(right1[:, None], right2) - torch.maximum(left1[:, None], left2)
    ).clamp(min=0)
    intersection = intersection_size.prod(dim=-1)
    area1 = (right1 - left1).clamp(min=0).prod(dim=-1)
    area2 = (right2 - left2).clamp(min=0).prod(dim=-1)
    union = area1[:, None] + area2 - intersection
    return intersection / union.clamp(min=1e-7)


# ============================================================================
# Object Detection Model
# ============================================================================
class ODModel(nn.Module):
    """Combine an ImageNet backbone, a three-scale YOLOv3 head, and prediction decoding."""

    def __init__(
        self,
        num_classes=NUM_CLASSES,
        num_anchors=3,
        pretrained=True,
        nms_thres=0.4,
        conf_thres=0.3,
        model_name=MODEL_NAME,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.backbone = Backbone(model_name=model_name, pretrained=pretrained)
        self.head = YOLOv3Head(num_classes, num_anchors, self.backbone.channels)

        self.anchors = ANCHORS
        self.nms_thres = nms_thres
        self.conf_thres = conf_thres

    def forward(self, x):
        return self.head(self.backbone(x))

    def _transform_predictions(self, pred, anchors):
        """Decode raw logits into image-normalized center-format boxes and scores."""
        # Use float32 for exp and coordinates to prevent overflow under AMP.
        pred = pred.float()
        batch_size, grid_size = pred.shape[:2]
        pred = pred.reshape(
            batch_size,
            grid_size,
            grid_size,
            self.num_anchors,
            5 + self.num_classes,
        )

        # Build grid-cell offsets and the corresponding anchor sizes.
        grid_y, grid_x = torch.meshgrid(
            torch.arange(grid_size, device=pred.device, dtype=pred.dtype),
            torch.arange(grid_size, device=pred.device, dtype=pred.dtype),
            indexing="ij",
        )
        grid_xy = torch.stack([grid_x, grid_y], dim=-1)[None, :, :, None, :]
        anchor_wh = pred.new_tensor(anchors).view(1, 1, 1, self.num_anchors, 2)

        # Decode x/y from cell offsets and w/h relative to the anchors.
        center = (pred[..., :2].sigmoid() + grid_xy) / grid_size
        size = pred[..., 2:4].clamp(-10, 10).exp() * anchor_wh
        objectness = pred[..., 4:5].sigmoid()

        # Each box has one class, so classification training uses CrossEntropy.
        class_probability = pred[..., 5:].softmax(dim=-1)
        return torch.cat([center, size, objectness, class_probability], dim=-1)

    @torch.inference_mode()
    def inference(self, x, conf_thres=None, nms_thres=None):
        if conf_thres is None:
            conf_thres = self.conf_thres
        if nms_thres is None:
            nms_thres = self.nms_thres

        self.eval()
        predictions = self(x)
        decoded = []
        for prediction, anchors in zip(predictions, self.anchors):
            scale_prediction = self._transform_predictions(prediction, anchors)
            decoded.append(scale_prediction.reshape(x.shape[0], -1, 5 + self.num_classes))

        all_predictions = torch.cat(decoded, dim=1)
        return non_max_suppression(all_predictions, conf_thres, nms_thres)


def getODmodel(pretrained=True, num_classes=NUM_CLASSES, model_name=MODEL_NAME):
    """Create the object detection model."""
    return ODModel(num_classes=num_classes, pretrained=pretrained, model_name=model_name)
