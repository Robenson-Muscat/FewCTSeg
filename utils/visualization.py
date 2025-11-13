import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder
from mpl_toolkits.axes_grid1 import ImageGrid

from torch.utils.data import DataLoader, Dataset
from torchvision import models, datasets, transforms
import os
import glob
from PIL import Image
from typing import Tuple, List, Optional, Union
import cv2
import albumentations as A
from tqdm import tqdm


def denormalize(tensor,mean,std):
    """
    Denormalize a tensor and return HWC numpy image in [0,1].

    Args:
        tensor: torch.Tensor of shape (C, H, W) with normalization applied.
        mean, std: arrays or lists of length 3 used for normalization.

    Returns:
        img_np: H x W x 3 float32 in [0,1].
    """
    
    tensor = tensor.permute(1, 2, 0).cpu().numpy()
    tensor = tensor * std + mean  # Unnormalize
    tensor = np.clip(tensor, 0, 1)  # Clip values to valid range
    return tensor




def visualize_test_prediction(index, model, dataset, device):
    """
    Display one test image and the model's predicted mask side-by-side.

    Args:
        index (int): index of an image of the class dataset
        model (torch.nn.Module): trained model
        dataset (Dataset): instance of TestCTScanDataset
        device (torch.device): CPU or GPU


    Example : 
    #
    test_ds = CTTestDataset(image_dir=os.path.join(PATH, "test-images"), transform=base_transform)
    model.load_state_dict(torch.load(..., map_location=DEVICE,weights_only=True))
    visualize_test_prediction(index=113, model=model, dataset=test_ds)

    """
    model.eval()
    pass

    
    img_tensor, _ = dataset[index]
    name = os.path.basename(dataset.image_paths[index])
    img = img_tensor.unsqueeze(0).to(device)

    #  Prediction
    with torch.no_grad():
        logits = model(img)
        pred_mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    img_np = denormalize(img_tensor)

    # Visualisation
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_np)
    plt.title(f"Image : {name}")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(pred_mask, cmap="nipy_spectral") 
    plt.title("Predicted mask")
    plt.axis("off")

    plt.show()





def plot_slice_seg(dataset, index):
    """ 
    Plot a labeled slice: original image and corresponding mask.

    Args:
        dataset: instance of a labeled dataset that returns (image_tensor, mask).
                 image_tensor may be normalized tensor (C,H,W) or numpy HWC.
        index: index to plot.

    Returns:
        (img_np, mask_np)
    """
   
    fig, axes = plt.subplots(1, 2)

    image_tensor,mask = dataset[index]
    image = denormalize(image_tensor)

    
    axes[0].imshow(image, cmap="gray")
    axes[1].imshow(image, cmap="gray")
    seg_masked = np.ma.masked_where(mask.reshape((256, 256)) == 0, (mask.reshape((256, 256))))
    axes[1].imshow(seg_masked, cmap="tab20")
    plt.axis("off")



def visualize_grid_masks(index_img,dataset):
    
    """ 
    Display a grid of images and their masks (overlaid).
      
    Args:
            index_img : List of indexes .Ex :visu_label(range(780,800))
            dataset (Dataset)
    Returns:
            None (shows matplotlib figure) 

    """
    nrows = len(index_img) //10 +1
    fig =plt.figure(figsize=(15,9))
    grid = ImageGrid(fig,111, (nrows,10))
    for ax,im in zip(grid, index_img):
        aug = dataset.transform(image=cv2.cvtColor(cv2.imread(dataset.image_paths[im]), cv2.COLOR_BGR2RGB), mask=dataset.masks[im])
        img, mask = aug['image'], aug['mask']

        img=denormalize(img)
        ax.imshow(img, cmap="gray")
        seg_masked = np.ma.masked_where(mask.reshape((256, 256)) == 0,(mask.reshape((256, 256))))
        #seg_masked = np.ma.masked_where(dataset.masks[im].reshape((256, 256)) == 0,(dataset.masks[im].reshape((256, 256))))
        ax.imshow(seg_masked)


def visualize_randomcrop(full_lab, idx, crop_size=(256,256), scale=(0.2,1.0), ratio=(0.75, 1.3333)):
    """
    Show original image+mask and their RandomResizedCrop result (image+mask).

    Args:
        full_lab_dataset: LabeledCTScanDataset instance.
        idx: index of the sample to crop
        crop_size: (height, width)
        scale, ratio: parameters forwarded to albumentations.RandomResizedCrop

    Returns:
        img_orig, mask_orig, img_crop, mask_crop (numpy arrays)
    """
   
    if not hasattr(full_lab, 'image_paths') or not hasattr(full_lab, 'masks'):
        raise ValueError("full_lab must have `image_paths` and `masks` (LabeledCTScanDataset).")
    if idx < 0 or idx >= len(full_lab.image_paths):
        raise IndexError(f'Index {idx} out of range for dataset of size {len(full_lab.image_paths)}')

    #
    img_path = full_lab.image_paths[idx]
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise RuntimeError(f"Impossible to process: {img_path}")
    img_orig = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mask_orig = full_lab.masks[idx].astype(np.uint8)

    h, w = crop_size
    try:
        rr = A.Compose([ A.RandomResizedCrop(size=(h, w), scale=scale, ratio=ratio, p=1.0) ])
    except Exception:
        # fallback pour anciennes versions d'albumentations
        rr = A.Compose([ A.RandomResizedCrop(height=h, width=w, scale=scale, ratio=ratio, p=1.0) ])

    # 
    augmented = rr(image=img_orig, mask=mask_orig)
    img_crop = augmented['image']
    mask_crop = augmented['mask'].astype(np.uint8)

    # affichage 2x2
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    ax = axes.ravel()

    ax[0].imshow(img_orig)
    ax[0].set_title(f"Original image")
    ax[0].axis('off')

    im0 = ax[1].imshow(mask_orig, interpolation='nearest')
    ax[1].set_title(f"Original mask")
    ax[1].axis('off')
    plt.colorbar(im0, ax=ax[1], fraction=0.046, pad=0.01)

    ax[2].imshow(img_crop)
    ax[2].set_title(f"Cropped image")
    ax[2].axis('off')

    im1 = ax[3].imshow(mask_crop, interpolation='nearest')
    ax[3].set_title(f"Cropped mask")
    ax[3].axis('off')
    plt.colorbar(im1, ax=ax[3], fraction=0.046, pad=0.01)

    plt.tight_layout()
    plt.show()



##

def get_indices_by_label(label: int, mask_df) -> list:

    """
    Return list of indices of images that contain `label` in their mask.

    Args:
        label: integer label to search for.
        masks_array: array of shape (N, H, W) or (N, H*W).

    Returns:
        list of integer indices.
    Ex :
        get_indices_by_label(26, full_lab.masks)
    """
    
    #masks = mask_df.values
    masks=mask_df.reshape(-1,256*256)
    # Boolean mask of images with the label
    mask_contains = (masks == label).any(axis=1)
    # Return indices of True
    return mask_contains.nonzero()[0].tolist()



def visualize_specific_label(label,dataset):
    """
     Visualize all images in `dataset` that contain a specific label.

    Args:
        label: integer label to display
        dataset: Labeled dataset

    Example :
        visualize_specific_label(7, full_lab)
    """
    index_img = get_indices_by_label(label, dataset.masks)
    nrows = len(index_img) //3 +1
    fig =plt.figure(figsize=(6,9))
    grid = ImageGrid(fig,111, (nrows,3))
    
    for ax,im in zip(grid, index_img):
        if dataset.transform is not None:
          aug = dataset.transform(image=cv2.cvtColor(cv2.imread(dataset.image_paths[im]), cv2.COLOR_BGR2RGB), mask=dataset.masks[im])
          img, mask = aug['image'], aug['mask']

          img=denormalize(img)
        else:
          img_tensor,mask=dataset[im]
          
          #mask = dataset.masks[im]
          img = img_tensor.permute(1, 2, 0).cpu().numpy()
    
          img = np.clip(img, 0, 1)  # Clip values to valid range

        ax.imshow(img, cmap="gray")
        seg_masked = np.ma.masked_where(mask.reshape((256, 256)) !=label,(mask.reshape((256, 256))))
        #seg_masked = np.ma.masked_where(dataset.masks[im].reshape((256, 256)) == 0,(dataset.masks[im].reshape((256, 256))))
        ax.imshow(seg_masked,cmap="hsv")




