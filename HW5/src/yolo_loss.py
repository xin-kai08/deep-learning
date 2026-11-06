"""YOLOv3 losses for GIoU localization, focal objectness, and classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ANCHORS


# ============================================================================
# Focal Loss
# ============================================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="none"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """Compute binary focal loss from raw logits and 0/1 targets."""
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        probabilities = inputs.sigmoid()
        p_t = targets * probabilities + (1 - targets) * (1 - probabilities)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_loss = alpha_t * (1 - p_t).pow(self.gamma) * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        if self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


# ============================================================================
# Bounding Box Loss
# ============================================================================
class BoxLoss(nn.Module):
    def __init__(self, loss_type="giou"):
        super().__init__()
        self.type = loss_type

    def forward(self, pred_boxes, target_boxes, anchors):
        """Compare raw predictions and encoded targets of shape (N, S, S, A, 4).

        Targets store [tx, ty, w, h]: cell offsets and image-relative sizes.
        Anchors use image-relative sizes; return losses of shape (N, S, S, A).
        """
        pred_boxes = pred_boxes.float()
        target_boxes = target_boxes.float()
        _, grid_size, _, num_anchors, _ = pred_boxes.shape
        anchor_wh = pred_boxes.new_tensor(anchors).view(1, 1, 1, num_anchors, 2)

        # Build grid offsets to decode centers in image-normalized coordinates.
        grid_y, grid_x = torch.meshgrid(
            torch.arange(grid_size, device=pred_boxes.device, dtype=pred_boxes.dtype),
            torch.arange(grid_size, device=pred_boxes.device, dtype=pred_boxes.dtype),
            indexing="ij",
        )
        grid_xy = torch.stack([grid_x, grid_y], dim=-1)[None, :, :, None, :]

        if self.type == "giou":
            ########################################################################
            # TODO 3: Implement GIoU box loss.
            # grid_xy and anchor_wh are provided above.
            # 1. Decode predictions and targets to image-normalized centers/sizes.
            # 2. Convert both sets of boxes to xyxy corner coordinates.
            # 3. Calculate intersection, union, and the smallest enclosing box.
            # 4. Return GIoU, with shape (N, S, S, A).
            ########################################################################
            ...
            ########################################################################
            # End of your code
            ########################################################################

        elif self.type == "mse":
            ########################################################################
            # TODO 2: Implement MSE box loss if you choose this loss type.
            # Compare x/y cell offsets after applying sigmoid to predictions.
            # Compare w/h in log space relative to anchor_wh.
            # Return a loss tensor of shape (N, S, S, A).
            ########################################################################
            ...
            ########################################################################
            # End of your code
            ########################################################################

        else:
            raise NotImplementedError(f"Box loss type '{self.type}' not implemented")


# ============================================================================
# Multi-Scale YOLOv3 Loss
# ============================================================================
class YOLOv3Loss(nn.Module):
    def __init__(
        self,
        lambda_coord=5.0,
        lambda_obj=1.0,
        lambda_noobj=0.5,
        lambda_class=1.0,
        anchors=ANCHORS,
        box_loss_type="giou",
    ):
        super().__init__()
        self.lambda_coord = lambda_coord
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj
        self.lambda_class = lambda_class

        self.focal_loss = FocalLoss(reduction="none")
        self.box_loss = BoxLoss(loss_type=box_loss_type)
        self.anchors = anchors

    def forward(self, predictions, targets):
        """Sum losses across scales and normalize positives and negatives separately.

        predictions: three (N, S, S, A * (5 + C)) tensors of raw logits.
        targets: three (N, S, S, A, 5 + C) encoded tensors.
        Return total, box, obj, noobj, and cls losses.
        """
        if len(predictions) != len(targets) or len(predictions) != len(self.anchors):
            raise ValueError("Predictions, targets and anchors must have the same scale count")

        # Keep the graph connected for empty targets so background objectness can still train.
        zero = sum(prediction.float().sum() * 0 for prediction in predictions)
        total_box_loss = zero
        total_obj_loss_pos = zero
        total_obj_loss_neg = zero
        total_cls_loss = zero

        total_num_pos = 0
        total_num_neg = 0

        for prediction, target, anchors in zip(predictions, targets, self.anchors):
            batch_size, grid_size, _, num_anchors, target_channels = target.shape
            prediction = prediction.float().reshape(
                batch_size,
                grid_size,
                grid_size,
                num_anchors,
                -1,
            )
            target = target.float()
            if prediction.shape != target.shape or target_channels < 6:
                raise ValueError(
                    f"Prediction/target shape mismatch: {prediction.shape}, {target.shape}"
                )

            ########################################################################
            # TODO 4: Accumulate the losses for this prediction scale.
            ########################################################################
            ...
            ########################################################################
            # End of your code
            ########################################################################

        # Normalize positives and negatives separately so background slots do not dominate.
        pos_denom = max(total_num_pos, 1)
        neg_denom = max(total_num_neg, 1)
        box_loss = total_box_loss / pos_denom
        obj_loss = total_obj_loss_pos / pos_denom
        noobj_loss = total_obj_loss_neg / neg_denom
        cls_loss = total_cls_loss / pos_denom

        total_loss = (
            self.lambda_coord * box_loss
            + self.lambda_obj * obj_loss
            + self.lambda_noobj * noobj_loss
            + self.lambda_class * cls_loss
        )
        return {
            "total": total_loss,
            "box": box_loss,
            "obj": obj_loss,
            "noobj": noobj_loss,
            "cls": cls_loss,
        }
