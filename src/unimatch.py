import torch
import numpy as np


# ------------------ CutMix : a customized augmentation ------------------
def cutmix(images, masks):
    """
    Apply cutmix on a batch of images
    Args : 
        images, masks: tensors shape (B,C,H,W),(B,H,W)
    """
    B, C, H, W = images.shape
    lam = np.random.beta(1.0, 1.0)
    rx = np.random.randint(W)
    ry = np.random.randint(H)
    rw = int(W * np.sqrt(1 - lam))
    rh = int(H * np.sqrt(1 - lam))
    x1 = np.clip(rx - rw // 2, 0, W)
    y1 = np.clip(ry - rh // 2, 0, H)
    x2 = np.clip(rx + rw // 2, 0, W)
    y2 = np.clip(ry + rh // 2, 0, H)
    perm = torch.randperm(B)
    mixed_images = images.clone()
    mixed_masks = masks.clone()
    mixed_images[:, :, y1:y2, x1:x2] = images[perm, :, y1:y2, x1:x2]
    mixed_masks[:, y1:y2, x1:x2] = masks[perm, y1:y2, x1:x2]
    return mixed_images, mixed_masks



# ------------------ Pseudo-labeling ------------------
def pseudo_targets_from_logits(logits_w, tau=0.95, ignore_index=255):
    """
     Convert weak logits into hard pseudo-labels, masking low-confidence pixels.

    Args:
        logits_w: Tensor of shape (B, Cl, H, W), raw model outputs (no softmax applied).
        tau: Confidence threshold; pixels below this confidence are ignored.
        ignore_index: Label assigned to ignored pixels.

    Returns:
        target: LongTensor (B, H, W) containing predicted classes per pixel.
                Low-confidence pixels are set to `ignore_index`.
        mask: BoolTensor (B, H, W), True for kept pixels (confidence >= tau).
    """
    probs = torch.softmax(logits_w, dim=1)
    confs, preds = probs.max(dim=1)   # confs (B,H,W), preds (B,H,W)
    mask = (confs >= tau)
    target = preds.clone().long()
    target[~mask] = ignore_index
    return target, mask


# ------------------ Infinite dataloader iterator ------------------
def inf_loop(dl):
    """
    Creates an infinite generator from a PyTorch DataLoader.

    Args:
        dl: A PyTorch DataLoader object.

    Yields:
        Batches indefinitely, cycling through the DataLoader.
        Typically used for semi-supervised setups (labeled & unlabeled iterators of different lengths

    Example:
        >>> lab_iter = inf_loop(train_loader)
        >>> imgs, masks = next(lab_iter)  # never raises StopIteration

    """
    while True:
        for x in dl:
            yield x


# ------------------ Image-level and feature-level perturbations ------------------

def strong_perturbation(batch_imgs_tensor, strong_transform, mean,std, device):
    """
    Applies two independent strong (non-deterministic) image-level perturbations
    to a normalized image batch.

    Args:
        batch_imgs_tensor: Float tensor of shape (B, C, H, W),
                           normalized with given mean/std.
        strong_transform: Albumentations Compose object defining the strong perturbation.
        device: Output device.
        mean, std: Normalization parameters used during training.

    Returns:
        (batch_s1, batch_s2): Two augmented tensors (B, C, H, W), normalized.

    """
    batch_imgs_np = batch_imgs_tensor.permute(0,2,3,1).cpu().numpy()
    batch_imgs_np = (batch_imgs_np * std + mean) * 255.0  
    batch_s1 = torch.stack([strong_transform(image=img.astype(np.uint8))['image']
                            for img in batch_imgs_np]).to(device)
    batch_s2 = torch.stack([strong_transform(image=img.astype(np.uint8))['image']
                            for img in batch_imgs_np]).to(device)
    return batch_s1, batch_s2
    

def weak_perturbation(batch_imgs_tensor, weak_transform,device):
    """
    Applies a weak geometric perturbation to a batch of raw images
    Args:
        weak_transform: Albumentations Compose object defining the weak perturbation.
        device: Output device.

    Returns:
        batch_w: Normalized tensor (B, C, H, W) ready for model input.
    """
    batch_imgs_np = batch_imgs_tensor.cpu().numpy()
    batch_w = torch.stack([weak_transform(image=im.astype(np.uint8))['image'] for im in batch_imgs_np]).to(device)  
    return batch_w

def feature_perturbation(model, feat_w,dropout):
    """
    Applies a feature-level perturbation to the last encoder feature map
    (dropout-style regularization) and computes the resulting logits.

    Args:
        model: segmentation model.
        feat_w : list of features maps.
        drop2d: nn.Dropout2d layer used for feature perturbation.

    Returns:
        logits_fp: Tensor of perturbed logits (B, Cl, H, W).

    """
    # feature perturbation on last feature map
    feat_fp = [f.clone() for f in feat_w]
    feat_fp[-1] = dropout(feat_fp[-1])
    logits_fp = model.segmentation_head(model.decoder(feat_fp))
    return logits_fp


