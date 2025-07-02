import numpy as np
import torch

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

