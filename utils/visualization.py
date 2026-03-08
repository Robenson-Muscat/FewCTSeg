import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
import matplotlib.pyplot as plt
import torch.nn.functional as F

from collections import Counter
from torch.utils.data import Dataset
from torchvision import transforms

from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.colors import ListedColormap

MEAN = np.array([0.485, 0.456, 0.406])
STD  = np.array([0.229, 0.224, 0.225])

#Legend colors
base_cmap = plt.cm.get_cmap("tab20", NUM_CLASSES)
colors = base_cmap(np.arange(NUM_CLASSES))
colors[0] = [0, 0, 0, 1] # background in black
seg_cmap = ListedColormap(colors)





def denormalize(tensor,mean=MEAN,std=STD):
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




def visualize_test_prediction(index, model, dataset, device, display=False):
    """
    Display one test image and the model's predicted mask side-by-side.

    Args:
        index (int): index of an image of the class dataset
        model (torch.nn.Module): trained model
        dataset (Dataset): instance of TestCTScanDataset
        device (torch.device): CPU or GPU
        display : True if you want to print the predicted labels

    Example :
    test_ds = TestCTScanDataset(image_dir=os.path.join(PATH, "test-images"), transform=base_transform)
    model.load_state_dict(torch.load(..., map_location=DEVICE,weights_only=True))
    visualize_test_prediction(index=113, model=model, dataset=test_ds,device = DEVICE)

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

    img = denormalize(img_tensor)
    seg_masked = np.ma.masked_where(pred_mask.reshape((256, 256)) == 0, (pred_mask.reshape((256, 256))))
    if display is True:
        print(np.unique(pred_mask))

    # Visualization

    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title(f"Image : {name}")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(img)
    plt.imshow(
        seg_masked,
        cmap=seg_cmap,
        vmin=0,
        vmax=NUM_CLASSES - 1,
        alpha=0.5,   
        interpolation="nearest"
        )
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
    """
   
    fig, axes = plt.subplots(1, 2)

    image_tensor,mask = dataset[index]
    image = denormalize(image_tensor)

    
    axes[0].imshow(image)
    axes[0].axis("off")
    seg_masked = np.ma.masked_where(mask.reshape((256, 256)) == 0, (mask.reshape((256, 256))))
    axes[1].imshow(image)
    axes[1].imshow(
        seg_masked,
        cmap=seg_cmap,
        vmin=0,
        vmax=NUM_CLASSES - 1,
        alpha=0.5,  
        interpolation="nearest"
        )
    axes[1].axis("off")

  

def visualize_grid_masks(index_img, dataset):
    """
    Display a grid of images and their masks (overlaid).

    Args:
        index_img : List of indexes
        dataset (Dataset)

    Returns:
        None (shows matplotlib figure)
    Example :
        visualize_grid_masks([389,390], full_lab)
    """ 

    nrows = len(index_img) // 10 + 1
    fig = plt.figure(figsize=(24,8*nrows))
    grid = ImageGrid(fig, 111, (nrows, 10))

    for ax, im in zip(grid, index_img):
        if dataset.transform is not None:
            aug = dataset.transform(image=cv2.cvtColor(cv2.imread(dataset.image_paths[im]), cv2.COLOR_BGR2RGB), mask=dataset.masks[im])
            img, mask = aug['image'], aug['mask']
            img = denormalize(img)
        else:
            img_tensor, mask = dataset[im]
            img = img_tensor.permute(1, 2, 0).cpu().numpy()
            img = np.clip(img, 0, 1)  # Clip values to valid range
        ax.axis("off")
        ax.imshow(img)

        
        ax.imshow(
            mask.reshape(256,256),
            cmap=seg_cmap,
            vmin=0,
            vmax=NUM_CLASSES - 1,
            alpha=0.5,   
            interpolation="nearest"
        )

    plt.show()


def visualize_specific_label(labels, dataset, max_images=30, mask_only=True):
    """
    Visualize images containing specific label(s) and highlight only those labels.

    Args:
        labels (int or list[int]): label(s) to visualize
        dataset (LabeledCTScanDataset)
        max_images (int): maximum number of images to display
        mask_only (bool): if True, only show selected labels in the mask

    Example:
        visualize_specific_label(7, full_lab)
        visualize_specific_label([37, 38], full_lab)
    """
    if isinstance(labels, int):
        labels = [labels]

    index_img = get_indices_by_labels_fast(labels, dataset)
    index_img = index_img[:max_images]

    if len(index_img) == 0:
        print(f"No image with labels {labels}")
        return

    ncols = 3
    nrows = len(index_img) // ncols + 1

    fig = plt.figure(figsize=(12, 4 * nrows))
    grid = ImageGrid(fig, 111, (nrows, ncols))

    for ax, im in zip(grid, index_img):

        if dataset.transform is not None:
            aug = dataset.transform(
                image=cv2.cvtColor(
                    cv2.imread(dataset.image_paths[im]), cv2.COLOR_BGR2RGB
                ),
                mask=dataset.masks[im]
            )
            img, mask = aug["image"], aug["mask"]
            img = denormalize(img)
        else:
            img_tensor, mask = dataset[im]
            img = img_tensor.permute(1, 2, 0).cpu().numpy()
            img = np.clip(img, 0, 1)

        ax.axis("off")
        ax.imshow(img)

        if mask_only:
            # Mask only on the requested labels
            mask_sel = np.zeros_like(mask)
            for lbl in labels:
                mask_sel[mask == lbl] = lbl
        else:
            mask_sel = mask

        ax.imshow(
            mask_sel.reshape(256, 256),
            cmap=seg_cmap,
            vmin=0,
            vmax=NUM_CLASSES - 1,
            alpha=0.5,
            interpolation="nearest"
        )

    plt.show()


def get_indices_by_labels_fast(labels, dataset):
    masks = dataset.masks.reshape(len(dataset.masks), -1)
    idx = np.ones(len(masks), dtype=bool)

    for label in labels:
        idx &= (masks == label).any(axis=1)

    return np.where(idx)[0].tolist()


def get_labels_by_index(index: int, dataset) -> list:
    """
    Return list of unique labels present in the mask at a given dataset index.

    Args:
        index: index of the sample in the dataset.
        dataset: Labeled Dataset object .

    Returns:
        list of labels present in the mask.
    """

    # Retrieve the mask depending on dataset structure
    if hasattr(dataset, "masks"):
        mask = dataset.masks[index]
    else:
        _, mask = dataset[index]

    # Flatten and return unique labels
    labels = np.unique(mask)
    return labels.tolist()




def top_cooccurring_labels(label, dataset, top_k=3, exclude_background=True):
    """
    Return the top-k labels that most frequently co-occur with a given label
    in the same image, along with their occurrence counts.

    Args:
        label (int): target label
        dataset (LabeledCTScanDataset)
        top_k (int): number of top co-occurring labels to return
        exclude_background (bool): exclude label 0 from results

    Returns:
        list of tuples: [(label_i, count_i), ...]

    Example:
        top3 = top_cooccurring_labels(51, full_lab)
    """

    cooccurrence_counter = Counter()

    masks = dataset.masks  # (N, H, W)

    for mask in masks:
        labels_present = set(np.unique(mask))

        # If the target label is not present → skip
        if label not in labels_present:
            continue

        # Remove the target label itself
        labels_present.discard(label)

        if exclude_background:
            labels_present.discard(0)

        # Increment counters
        for lbl in labels_present:
            cooccurrence_counter[lbl] += 1

    return cooccurrence_counter.most_common(top_k)




def compute_mean_area_per_label(dataset, num_classes=55, ignore_index=0):
    """
    Compute mean area per label
    Example :
        df_stats_more = compute_mean_area_per_label(full_lab)
        df_stats_more.head(10)
    """
    H, W = dataset.masks[0].shape
    total_pixels = H * W

    stats = {}

    for label in range(num_classes):
        if label == ignore_index:
            continue

        areas = []
        for mask in dataset.masks:
            pix = np.sum(mask == label)
            if pix > 0:
                areas.append(pix / total_pixels)

        if len(areas) > 0:
            stats[label] = {
                "mean": np.mean(areas)*100,
                "median": np.median(areas)*100,
                "count": len(areas)
            }

    return pd.DataFrame(stats).T.sort_values("mean", ascending=False) 


def plot_multilabel_mean_area_bar(
    dataset,
    num_classes=55,
    ignore_index=0,
    normalize=True,
    min_presence=5
):
    """
    Bar plot of the mean area per label (computed only on images where the label is present).

    Args:
        dataset : LabeledCTScanDataset
        num_classes :
        ignore_index :
        normalize : area as ratio (True) or in pixels (False)
        min_presence : minimum number of images required to display the label

    Example:
        plot_multilabel_mean_area_bar(dataset=full_lab, num_classes=NUM_CLASSES, min_presence=10)
    """

    H, W = dataset.masks[0].shape
    total_pixels = H * W

    labels = []
    mean_areas = []
    counts = []

    for label in range(num_classes):
        if label == ignore_index:
            continue

        areas = []
        for mask in dataset.masks:
            pix = np.sum(mask == label)
            if pix > 0:
                areas.append(pix / total_pixels if normalize else pix)

        if len(areas) >= min_presence:
            labels.append(label)
            mean_areas.append(np.mean(areas))
            counts.append(len(areas))

    # sort by decreasing mean area
    order = np.argsort(mean_areas)[::-1]
    labels = np.array(labels)[order]
    mean_areas = np.array(mean_areas)[order]
    counts = np.array(counts)[order]

    plt.figure(figsize=(14, 6))
    plt.bar(labels, mean_areas, color="red")
    plt.xlabel("Label")
    plt.ylabel("Mean Area" + (" (ratio)" if normalize else " (pixels)"))
    plt.title("Mean Area Covered per Label (only when the label is present)")
    plt.xticks(labels, rotation=90)
    plt.grid(axis="y", alpha=0.3)
    plt.show()


def plot_class_distribution(dataset):
    """
    Plot the pixel distribution of each class in the dataset as percentages.

    Args:
        dataset (Dataset): dataset returning (image, mask) pairs where mask
                           contains class indices per pixel.

    Example:
        plot_class_distribution(full_lab)
    """


    class_counts = {}

    for _, mask in dataset:
        unique_classes = torch.unique(mask)
        unique_classes = unique_classes[unique_classes != IGNORE_INDEX]

        for class_id in unique_classes:
            class_id = class_id.item()

            if class_id not in class_counts:
                class_counts[class_id] = 0

            class_counts[class_id] += (mask == class_id).sum().item()

    # Convert to percentage
    total_pixels = sum(class_counts.values())
    class_percentages = {
        k: (v / total_pixels) * 100 for k, v in class_counts.items()
    }

    class_percentages = dict(sorted(class_percentages.items()))

    # Display the histogram
    plt.figure(figsize=(10, 6))
    plt.bar(class_percentages.keys(), class_percentages.values(), color='skyblue')

    plt.title("Class Distribution in Percentage", fontsize=14)
    plt.xlabel("Label")
    plt.ylabel("Percentage")
    plt.xticks(range(NUM_CLASSES), rotation=90)
    plt.tight_layout()
    plt.show()


