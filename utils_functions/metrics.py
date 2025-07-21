import numpy as np
import cv2
import pandas as pd


NUM_CLASSES = 54

def dice_image(prediction, ground_truth):
    """
    Compute the dice score for an image

    Args:
        ground_truth : Ground truth mask 
        prediction : Prediction mask 
    Returns :
        Dice score for an image
    """

    intersection = np.sum(prediction * ground_truth)
    if np.sum(prediction) == 0 and np.sum(ground_truth) == 0:
        return np.nan
    return 2 * intersection / (np.sum(prediction) + np.sum(ground_truth))

def dice_multiclass(prediction, ground_truth):
    """
    Compute the dice score per class for an image

    Args:
        ground_truth : Ground truth mask 
        prediction : Prediction mask 
    Returns :
        Array of dice score per class for an image
    """
    dices = []
    for i in range(1, NUM_CLASSES + 1):
        dices.append(dice_image(prediction == i, ground_truth == i))
    return np.array(dices)

def dice_pandas(y_true_df: pd.DataFrame, y_pred_df: pd.DataFrame) -> float:
    """
    Args:
        y_true_df : Ground truth masks (pd.DataFrame)
        y_pred_df : Prediction masks (pd.DataFrame)
    Returns :
        Dice score
    """
    y_pred_df = y_pred_df.T
    y_true_df = y_true_df.T
    individual_dice = []
    for row_index in range(y_true_df.values.shape[0]):
        dices = dice_multiclass(y_true_df.values[row_index].ravel(), y_pred_df.values[row_index].ravel())
        individual_dice.append(dices)

    final = np.stack(individual_dice)
    cls_dices = np.nanmean(final, axis=0)
    return float(np.nanmean(cls_dices))

