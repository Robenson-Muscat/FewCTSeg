import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder
from mpl_toolkits.axes_grid1 import ImageGrid


def denormalize(tensor, mean=None, std=None):
    """Reverses the normalization applied during preprocessing and get image from tensor"""
    if mean is None:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3)
    if std is None:
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3)
    tensor = tensor.permute(1, 2, 0).numpy()
    tensor = tensor * std.numpy() + mean.numpy()  # Unnormalize
    tensor = np.clip(tensor, 0, 1)  # Clip values to valid range
    return tensor



def plot_slice_seg(image, mask):
    """ Plot a slice image with his corresponding mask

    Args:
        image : (ndarray) an image, shape (H, W, C),
        mask  : (ndarray) corresponding mask,   shape (H,  W)
    """
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(image, cmap="gray")
    axes[1].imshow(image, cmap="gray")
    seg_masked = np.ma.masked_where(mask.reshape((256, 256)) == 0, (mask.reshape((256, 256))))
    axes[1].imshow(seg_masked, cmap="tab20")
    plt.axis("off")
    
#paths = sorted(glob.glob(os.path.join(IMG_DIR, '*.png')), key=alphanumeric_sort)
#masks = pd.read_csv(MASK_CSV, index_col=0).T.values.reshape(-1,256,256).astype(np.uint8)
#valid = [m.sum()!=0 for m in masks]
#image_paths = [p for p,v in zip(paths,valid) if v]
#image =  cv2.cvtColor(cv2.imread(image_paths[0]), cv2.COLOR_BGR2RGB)
#mask = masks[0]

def visu_index(index_img, index,labels_train):
    """Visualize index mask on index_img images"""

  nrows = len(index_img) //3 +1
  fig =plt.figure(figsize=(6,9))
  grid = ImageGrid(fig,111, (nrows,3))
  for ax,im in zip(grid, index_img):
    ax.imshow(dataset.imgs[im], cmap="gray")
    seg_masked = np.ma.masked_where(labels_train.iloc[im].values.reshape((256, 256))!= index, (labels_train.iloc[im].values.reshape((256, 256))))
    ax.imshow(seg_masked)

def visu_label(index_img):
  """ 
  Visualize all masks on the index_img images
  
  Args:
    index_img : List of indexes .Ex :visu_label(range(780,800))

  Returns:
    ImageGrid of overlays image-masks 
        
    """
  nrows = len(index_img) //3 +1
  fig =plt.figure(figsize=(6,9))
  grid = ImageGrid(fig,111, (nrows,3))
  for ax,im in zip(grid, index_img):
    ax.imshow(dataset.imgs[im], cmap="gray")
    seg_masked = np.ma.masked_where(labels_train.iloc[im].values.reshape((256, 256)) == 0, (labels_train.iloc[im].values.reshape((256, 256))))
    ax.imshow(seg_masked)

def visu_label(index_img):
  """ 
  Visualize all masks on the index_img images
  
    Args:
        index_img : List of indexes .Ex :visu_label(range(780,800))

    Returns:
        ImageGrid of overlays image-masks 
        
    """
  nrows = len(index_img) //10 +1
  fig =plt.figure(figsize=(15,9))
  grid = ImageGrid(fig,111, (nrows,10))
  for ax,im in zip(grid, index_img):
    ax.imshow(data_train[im], cmap="gray")
    #ax.imshow(dataset.imgs[im], cmap="gray")
      
    seg_masked = np.ma.masked_where(labels_train.iloc[im].values.reshape((256, 256)) == 0, (labels_train.iloc[im].values.reshape((256, 256))))
    ax.imshow(seg_masked)


def visualize_transformations(dataset, idx=0):
    """Visualize transformations on an image (idx) of the dataset class"""
    dataset.transform = A.Compose([t for t in dataset.transform if not isinstance(t, (A.Normalize, ToTensorV2))])
    _, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 24))
    for i in range(2):
        image, mask = dataset[idx]
        ax[i, 0].imshow(image)
        ax[i, 1].imshow(mask, interpolation="nearest")
        ax[i, 0].set_title("Augmented image")
        ax[i, 1].set_title("Augmented mask")
        ax[i, 0].set_axis_off()
        ax[i, 1].set_axis_off()
    plt.tight_layout()
    plt.show()




def get_indices_by_label(label: int, mask_df: pd.DataFrame) -> list:
    """
    Returns the indices of the images whose mask contains the given label.

    Args:
        label: integer corresponding to the desired label.
        mask_df: DataFrame loaded by pd.read_csv(mask_csv, index_col=0), "
        " where each column is a pixel and each line a pixel index.

    Returns:
        List[int]: indices of images containing at least one pixel of the label.
    """
    
    masks = mask_df.T.values
    # Boolean mask of images with the label
    mask_contains = (masks == label).any(axis=1)
    # Return indices of True
    return mask_contains.nonzero()[0].tolist()

#masks = pd.read_csv("/content/drive/MyDrive/Raidium_challenge/y_train.csv")
#get_indices_by_label(7,masks)


def visualize_cutmix(images, masks, mean=None, std=None, alpha=1.0):
    """
    Display the result of CutMix on a batch
    
    Args:
        images (Tensor): batch of images, shape (B, C, H, W),
        masks  (Tensor): batch of masks,   shape (B, H, W)
        mean   (list ou tuple, optional), std    (list ou tuple, optional): 
        alpha  (float): paramater for CutMix
    """
    B, C, H, W = images.shape

    #
    mixed_imgs, mixed_masks = cutmix(images, masks)

    if mean is not None and std is not None:
        def denorm_batch_image(t):
            """Apply a reverse normalization to a batch of images """
            t = t.clone().cpu()
            for c, (m, s) in enumerate(zip(mean, std)):
                t[:, c, :, :].mul_(s).add_(m)
            return t
        mixed_imgs = denorm_batch_image(mixed_imgs)

    fig = plt.figure(figsize=(6, 3*B))
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(B, 2),
                     axes_pad=0.4,
                     share_all=False,
                     cbar_mode=None)
    
    # Display each image and mask
    for i in range(B):
        img_np = mixed_imgs[i].permute(1, 2, 0).cpu().numpy().clip(0,1)
        mask_np = mixed_masks[i].cpu().numpy()

        ax_img = grid[2*i]
        ax_mask = grid[2*i + 1]

        
        ax_img.imshow((img_np * 255).astype(np.uint8))
        ax_img.axis('off')

        ax_mask.imshow(mask_np, vmin=0, vmax=mixed_masks.max().item())
        ax_mask.axis('off')

    plt.tight_layout()
    plt.show()


#imgs, masks = next(iter(train_loader))  # imgs: (B,3,256,256), masks: (B,256,256)
#visualize_cutmix(imgs, masks,mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])



###Créer une fonction overlay image - masque prédit + Visualisation du cutmix 
### J'ai la fonction overlay image -masque prédit dans mon code pour le SAM + chercher dans mes codes commennt gérer les iter(train_loader)











def visualize_mask(image, mask, original_image=None, original_mask=None):
    """Visualize mask of the corresponding image side by side (equivalent of plot_slice_seg)"""
    
    fontsize = 18

    if original_image is None and original_mask is None:
        f, ax = plt.subplots(2, 1, figsize=(8, 8))

        ax[0].imshow(image)
        ax[1].imshow(mask)
    else:
        f, ax = plt.subplots(2, 2, figsize=(8, 8))

        ax[0, 0].imshow(original_image)
        ax[0, 0].set_title('Original image', fontsize=fontsize)

        ax[1, 0].imshow(original_mask)
        ax[1, 0].set_title('Original mask', fontsize=fontsize)

        ax[0, 1].imshow(image)
        ax[0, 1].set_title('Transformed image', fontsize=fontsize)

        ax[1, 1].imshow(mask)
        ax[1, 1].set_title('Transformed mask', fontsize=fontsize)



#image = cv2.imread('0fea4b5049_image.png')
#mask = cv2.imread('0fea4b5049.png', cv2.IMREAD_GRAYSCALE)
#print(image.shape, mask.shape)
#original_height, original_width = image.shape[:2]
#visualize(image, mask)



def visu_augmentation(dataset, index=0):
    """
    VERSION PESTIFEREE
    
    Display an image + mask w/o augmentation, without normalize
    """
    transformed_img, transformed_mask = dataset[index]

    # Denormalize image for display
    img_np = denormalize(transformed_img)
    mask_np = transformed_mask.cpu().numpy()

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow((img_np * 255).astype(np.uint8))
    axs[0].set_title('Augmentated image (without normalize)')
    axs[0].axis('off')

    axs[1].imshow(mask_np, cmap='gray')
    axs[1].set_title('Mask')
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()

