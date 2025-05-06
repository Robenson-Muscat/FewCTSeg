import numpy as np
import cv2
import pandas as pd


NUM_CLASSES = 54

def dice_image(prediction, ground_truth):
    intersection = np.sum(prediction * ground_truth)
    if np.sum(prediction) == 0 and np.sum(ground_truth) == 0:
        return np.nan
    return 2 * intersection / (np.sum(prediction) + np.sum(ground_truth))

def dice_multiclass(prediction, ground_truth):
    dices = []
    for i in range(1, NUM_CLASSES + 1):
        dices.append(dice_image(prediction == i, ground_truth == i))
    return np.array(dices)

def dice_pandas(y_true_df: pd.DataFrame, y_pred_df: pd.DataFrame) -> float:
    y_pred_df = y_pred_df.T
    y_true_df = y_true_df.T
    individual_dice = []
    for row_index in range(y_true_df.values.shape[0]):
        dices = dice_multiclass(y_true_df.values[row_index].ravel(), y_pred_df.values[row_index].ravel())
        individual_dice.append(dices)

    final = np.stack(individual_dice)
    cls_dices = np.nanmean(final, axis=0)
    return float(np.nanmean(cls_dices))

# Compute metric
#dice_pandas(labels_val, labels_val_predicted_baseline)
# ---------------------------
# Dice metric function
# ---------------------------
def compute_dice_score(preds, masks, num_classes=55):
    # preds, masks: torch.LongTensor (B,H,W)
    dices = []
    for c in range(num_classes):
        pred_c = (preds == c).float()
        mask_c = (masks == c).float()
        intersection = (pred_c * mask_c).sum()
        union = pred_c.sum() + mask_c.sum()
        if union.item() == 0:
            continue
        dices.append((2.0 * intersection / union).item())
    return np.mean(dices) if dices else 0.0


# ---------------------------
# Dice metric
# ---------------------------
def compute_dice_score(preds, masks, C=55):
    dices = []
    for c in range(C):
        pc = (preds==c).float()
        mc = (masks==c).float()
        inter = (pc*mc).sum()
        denom = pc.sum()+mc.sum()
        if denom>0:
            dices.append((2*inter/denom).item())
    return np.mean(dices) if dices else 0.0


def dice_score(preds: torch.Tensor, targets: torch.Tensor, num_classes: int = 55, smooth: float = 1e-6) -> torch.Tensor:
    """
    Calculate the average Dice score over all classes for a batch.
    preds: logits output from the model (B, C, H, W)
    targets: ground-truth masks (B, H, W)
    """
    # Convert logits to predicted class labels
    preds = torch.argmax(preds, dim=1)
    dice = 0.0
    for cls in range(num_classes):
        pred_cls = (preds == cls).float()
        true_cls = (targets == cls).float()
        intersection = (pred_cls * true_cls).sum(dim=(1, 2))
        union = pred_cls.sum(dim=(1, 2)) + true_cls.sum(dim=(1, 2))
        dice += ((2 * intersection + smooth) / (union + smooth)).mean()
    return dice / num_classes



def weighted_dice_score(preds: torch.Tensor,
                        targets: torch.Tensor,
                        weights: torch.Tensor,
                        num_classes: int = 55,
                        smooth: float = 1e-6) -> torch.Tensor:
    """
    Calculate the weighted Dice score over all classes for a batch.
    weights: 1D tensor of length num_classes
    """
    preds = torch.argmax(preds, dim=1)
    dice_per_class = torch.zeros(num_classes, device=preds.device)
    for cls in range(num_classes):
        pred_cls = (preds == cls).float()
        true_cls = (targets == cls).float()
        intersection = (pred_cls * true_cls).sum(dim=(1, 2))
        union = pred_cls.sum(dim=(1, 2)) + true_cls.sum(dim=(1, 2))
        dice_per_class[cls] = ((2 * intersection + smooth) / (union + smooth)).mean()
    # Weighted sum
    return (dice_per_class * weights).sum() / weights.sum()
