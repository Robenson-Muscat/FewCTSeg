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


