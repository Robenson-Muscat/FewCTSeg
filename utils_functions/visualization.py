import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder
from mpl_toolkits.axes_grid1 import ImageGrid

def denormalize(tensor, mean, std):
    """Reverses the normalization applied during preprocessing and get image from tensor"""
    if mean is None:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3)
    if std is None:
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3)
    tensor = tensor.permute(1, 2, 0).numpy()
    tensor = tensor * std.numpy() + mean.numpy()  # Unnormalize
    tensor = np.clip(tensor, 0, 1)  # Clip values to valid range
    return tensor



def plot_slice_seg(slice_image, seg):
    """ Plot a slice with his corresponding mask"""
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(slice_image, cmap="gray")
    axes[1].imshow(slice_image, cmap="gray")
    seg_masked = np.ma.masked_where(seg.reshape((256, 256)) == 0, (seg.reshape((256, 256))))
    axes[1].imshow(seg_masked, cmap="tab20")
    plt.axis("off")


def visu_index(index_img, index):
    """Visualize index mask on index_img images"""

  nrows = len(index_img) //3 +1
  fig =plt.figure(figsize=(6,9))
  grid = ImageGrid(fig,111, (nrows,3))
  for ax,im in zip(grid, index_img):
    ax.imshow(dataset.imgs[im], cmap="gray")
    seg_masked = np.ma.masked_where(labels_train.iloc[im].values.reshape((256, 256))!= index, (labels_train.iloc[im].values.reshape((256, 256))))
    ax.imshow(seg_masked)

def visu_label(index_img):
  """Visualize all masks on the index_img images"""
  nrows = len(index_img) //3 +1
  fig =plt.figure(figsize=(6,9))
  grid = ImageGrid(fig,111, (nrows,3))
  for ax,im in zip(grid, index_img):
    ax.imshow(dataset.imgs[im], cmap="gray")
    seg_masked = np.ma.masked_where(labels_train.iloc[im].values.reshape((256, 256)) == 0, (labels_train.iloc[im].values.reshape((256, 256))))
    ax.imshow(seg_masked)


 
def visualize_segmentation(dataset, idx=0, samples=3):
    """Visualization of the image and its mask"""
    if isinstance(dataset.transform, A.Compose):
        vis_transform_list = [
            t for t in dataset.transform
            if not isinstance(t, (A.Normalize, A.ToTensorV2))
        ]
        vis_transform = A.Compose(vis_transform_list)
    else:
        print("Warning: Could not automatically strip Normalize/ToTensor for visualization.")
        vis_transform = dataset.transform
 
    figure, ax = plt.subplots(samples + 1, 2, figsize=(8, 4 * (samples + 1)))
 
    # --- Get the original image and mask --- #
    original_transform = dataset.transform
    dataset.transform = None # Temporarily disable for raw data access
    image, mask = dataset[idx]
    dataset.transform = original_transform # Restore
 
    # Display original
    ax[0, 0].imshow(image)
    ax[0, 0].set_title("Original Image")
    ax[0, 0].axis("off")
    ax[0, 1].imshow(mask, cmap='gray') # Show mask directly
    ax[0, 1].set_title("Original Mask")
    ax[0, 1].axis("off")
    # ax[0, 1].imshow(overlay_mask(image, mask)) # Or show overlay
    # ax[0, 1].set_title("Original Overlay")
 
    # --- Apply and display augmented versions --- #
    for i in range(samples):
        # Apply the visualization transform
        if vis_transform:
            augmented = vis_transform(image=image, mask=mask)
            aug_image = augmented['image']
            aug_mask = augmented['mask']
        else:
            aug_image, aug_mask = image, mask # Should not happen normally
 
        # Display augmented image and mask
        ax[i + 1, 0].imshow(aug_image)
        ax[i + 1, 0].set_title(f"Augmented Image {i+1}")
        ax[i + 1, 0].axis("off")
 
        ax[i + 1, 1].imshow(aug_mask, cmap='gray') # Show mask directly
        ax[i + 1, 1].set_title(f"Augmented Mask {i+1}")
        ax[i + 1, 1].axis("off")
        # ax[i+1, 1].imshow(overlay_mask(aug_image, aug_mask)) # Or show overlay
        # ax[i+1, 1].set_title(f"Augmented Overlay {i+1}")
 
 
    plt.tight_layout()
    plt.show()
 
# Assuming train_dataset is created with train_transform:
# visualize_segmentation(train_dataset, samples=3)
 

 

def overlay_mask(image, mask, alpha=0.5, color=(0, 1, 0)): # Green overlay
    """ Simple function to overlay mask on image for visualization (green overlay)"""
    # Convert mask to 3 channels if needed, ensure boolean type
    mask_overlay = np.zeros_like(image, dtype=np.uint8)
    # Create a color overlay where mask is > 0
    mask_overlay[mask > 0] = (np.array(color) * 255).astype(np.uint8)
 
    # Blend image and overlay
    overlayed_image = cv2.addWeighted(image, 1, mask_overlay, alpha, 0)
    return overlayed_image


#Visualize augmentations
def visualize_augmentations(dataset, idx=0, samples=5):
    dataset = copy.deepcopy(dataset)
    dataset.transform = A.Compose([t for t in dataset.transform if not isinstance(t, (A.Normalize, ToTensorV2))])
    _, ax = plt.subplots(nrows=samples, ncols=2, figsize=(10, 24))
    for i in range(samples):
        image, mask = dataset[idx]
        ax[i, 0].imshow(image)
        ax[i, 1].imshow(mask, interpolation="nearest")
        ax[i, 0].set_title("Augmented image")
        ax[i, 1].set_title("Augmented mask")
        ax[i, 0].set_axis_off()
        ax[i, 1].set_axis_off()
    plt.tight_layout()
    plt.show()

#How to use augmentations
#aug = A.ElasticTransform(always_apply=False, p=1.0, alpha=1.0, sigma=50.0, alpha_affine=50.0, interpolation=0, border_mode=0, value=(0, 0, 0), mask_value=None, approximate=False)
#elt = aug(image=originalImage)[‘image’]
#cv2_imshow(elt)

