import numpy as np
import cv2
import pandas as pd


# ---------------------------
# Dice metric function
# ---------------------------
def compute_dice_score(preds, masks, num_classes):
    """
    Compute mean (macro) Dice score over classes for a single batch.

    Args:
        preds: LongTensor shape (B, C, H, W) predicted class indices.
        masks: LongTensor shape (B, C, H, W) ground-truth class indices.
        num_classes: total number of classes.

    Returns:
        mean_dice: float, average Dice across classes that appear in GT or predictions.
                   If no class has any pixels in both pred and gt (empty), returns 0.0.
    """
    
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




def compute_per_class_dice(model, val_loader, num_classes,
                                     device=None, ignore_index=None, eps=1e-6,
                                     show_progress=True, plot=True, figsize=(16,6),
                                     save_path=None):
    """
    Compute per-class Dice over a validation DataLoader and optionally plot the results.

    Args:
        model: nn.Module.
        val_loader: DataLoader (validation set).
        num_classes: int.
        device: torch.device ou None .
        eps: float, 
        show_progress: bool, 
        plot: bool, display the barplot if True.
        figsize: tuple, taille de la figure matplotlib.
        save_path: str ou None, path to save the figure (png). 

    Returns:
        dice_array: numpy array of shape (num_classes,) with per-class Dice (np.nan where denom==0)
        df: pandas DataFrame with columns ['class','dice','support_pixels','pred_pixels']
        macro_dice: float, mean Dice over finite classes
        weighted_dice: float, dice weighted by support pixels (ignores classes with zero support)
        fig_or_none: matplotlib.Figure if return_figure else None
    Example:
        model.eval()
        dice_array, dice_df, macro, weighted = compute_per_class_dice_with_plot(
            model, val_loader, NUM_CLASSES, device=DEVICE, ignore_index=None,
            show_progress=True, plot=True, figsize=(20,5), save_path=None
        )
        print(dice_df)
    """
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device('cpu')

    model.eval()
    inter = torch.zeros(num_classes, dtype=torch.double, device=device)
    pred_sum = torch.zeros(num_classes, dtype=torch.double, device=device)
    target_sum = torch.zeros(num_classes, dtype=torch.double, device=device)

    iterator = tqdm(val_loader) if show_progress else val_loader

    with torch.no_grad():
        for batch in iterator:
            imgs, masks = batch[0], batch[1]

            imgs = imgs.to(device)
            # convert masks to tensor if needed
            if not torch.is_tensor(masks):
                masks = torch.tensor(masks)
            masks = masks.to(device)
            if masks.dim() == 4 and masks.shape[1] == 1:
                masks = masks.squeeze(1)
            masks = masks.long()

            logits = model(imgs)
            preds = logits.argmax(dim=1).long()  # (B,H,W)

            preds_flat = preds.view(-1)
            masks_flat = masks.view(-1)

            if ignore_index is not None:
                valid = masks_flat != ignore_index
                if valid.sum().item() == 0:
                    continue
                preds_flat = preds_flat[valid]
                masks_flat = masks_flat[valid]

            # accumulate per class
            # vectorized using one-hot like counting with bincount
            # mais torch.bincount with minlength fonctionne only on 1D CPU tensors -> move to CPU
            preds_cpu = preds_flat.cpu()
            masks_cpu = masks_flat.cpu()

            # pred counts
            pred_counts = torch.bincount(preds_cpu, minlength=num_classes).double().to(device)
            target_counts = torch.bincount(masks_cpu, minlength=num_classes).double().to(device)

            # intersection: for each class c, count where preds==c and masks==c.
            # compute mask of equal elements and do bincount on preds (or masks)
            eq_mask = (preds_cpu == masks_cpu)
            if eq_mask.any():
                preds_eq = preds_cpu[eq_mask]
                inter_counts = torch.bincount(preds_eq, minlength=num_classes).double().to(device)
            else:
                inter_counts = torch.zeros(num_classes, dtype=torch.double, device=device)

            pred_sum += pred_counts
            target_sum += target_counts
            inter += inter_counts

    denom = pred_sum + target_sum
    dice = torch.full((num_classes,), float('nan'), dtype=torch.double, device=device)
    nonzero = denom > 0
    dice[nonzero] = (2.0 * inter[nonzero]) / (denom[nonzero] + eps)

    dice_array = dice.cpu().numpy()
    supports = target_sum.cpu().numpy().astype(np.int64)
    preds_pixels = pred_sum.cpu().numpy().astype(np.int64)

    df = pd.DataFrame({
        'class': np.arange(num_classes),
        'dice': dice_array,
        'support_pixels': supports,
        'pred_pixels': preds_pixels
    })

    # macro: mean over finite values
    finite_mask = np.isfinite(dice_array)
    if finite_mask.any():
        macro_dice = float(np.nanmean(dice_array))
    else:
        macro_dice = float('nan')

    # weighted dice: weights by support pixels (ignore classes with zero support)
    support_pos = supports > 0
    if support_pos.any():
        weighted_dice = float(np.sum(dice_array[support_pos] * supports[support_pos]) / np.sum(supports[support_pos]))
    else:
        weighted_dice = float('nan')

    # Plot barplot
    if plot:
        x = np.arange(num_classes)
        heights = np.nan_to_num(dice_array, nan=0.0)  # matplotlib can't plot nan as bar height
        plt.figure(figsize=figsize)
        plt.bar(x, heights)
        plt.xlabel('Classe')
        plt.ylabel('Dice')
        plt.title(f'Dice par classe — macro: {macro_dice:.4f} — weighted: {weighted_dice:.4f}')
        plt.ylim(0.0, 1.0)
        # annotate bars for finite values
        for xi, h, is_finite in zip(x, heights, finite_mask):
            if is_finite:
                plt.text(xi, h + 0.01, f'{h:.3f}', ha='center', va='bottom', fontsize=6, rotation=90)
            else:
                # marque les classes absentes
                plt.text(xi, 0.01, 'abs', ha='center', va='bottom', fontsize=6, rotation=90)

        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight', dpi=200)
        plt.show()

    return dice_array, df, macro_dice, weighted_dice

