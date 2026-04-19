import os
import math
import torch
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
import umap

import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataset import TestCTScanDataset

# ------------------ Dataset ------------------
class FullCTScanDataset(Dataset):
    def __init__(self, img_dir, mask_csv, transform=None, indices=None):
        paths = sorted(glob.glob(os.path.join(img_dir, '*.png')), key=alphanumeric_sort)
        masks = pd.read_csv(mask_csv, index_col=0).T.values.reshape(-1,256,256).astype(np.uint8)
        #valid = [m.sum()>0 for m in masks]
        #paths = [p for p,v in zip(paths, valid) if v]
        #masks = masks[np.array(valid)]
        if indices is not None:
            self.image_paths = [paths[i] for i in indices]
            self.masks = masks[np.array(indices)]
        else:
            self.image_paths = paths
            self.masks = masks
        self.transform = transform

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(self.image_paths[idx]), cv2.COLOR_BGR2RGB)
        mask = self.masks[idx]
        if self.transform is not None:
            aug = self.transform(image=img, mask=mask)
            return aug['image'], aug['mask']
        else:
            return torch.from_numpy(img).permute(2,0,1).float()/255.0 , mask




# ------------------ Config ------------------
SEED = 26
PATH = "../data/"
IMG_DIR = os.path.join(PATH, 'train-images/')
MASK_CSV = os.path.join(PATH, 'y_train.csv')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------ Embedding Extraction ------------------
# Fonction pour extraire les embeddings
def extract_embeddings(model, dataloader):
    embeddings = []
    labels = []
    with torch.no_grad():
        for images, label in tqdm(dataloader):
            images = images.to(device)
            # Extraire les embeddings
            embedding = model(images)
            embeddings.append(embedding[0].cpu().numpy())
            labels.append(label)
    return np.vstack(embeddings), np.hstack(labels)
# ------------------ UMAP Reduction ------------------
def umap_reduce(embeddings, n_components=64, random_state=SEED):
    """
    Reduce embeddings using UMAP.

    Args:
        embeddings: (N, D)
        n_components: target dimension

    Returns:
        reduced embeddings and fitted reducer
    """
    reducer = umap.UMAP(n_components=n_components, random_state=random_state)
    reduced = reducer.fit_transform(embeddings)
    return reduced, reducer


# ------------------ Nearest Neighbor Pairs ------------------
def find_top_k_pairs(reduced_emb, k=5):
    """
    Find top-k closest unique pairs based on Euclidean distance.
    """
    nbrs = NearestNeighbors(n_neighbors=10, metric="euclidean").fit(reduced_emb)
    distances, indices = nbrs.kneighbors(reduced_emb)

    pairs = {}
    N = reduced_emb.shape[0]

    for i in range(N):
        for j_idx in range(1, indices.shape[1]):  # skip self
            j = indices[i, j_idx]
            if i == j:
                continue

            a, b = min(i, j), max(i, j)
            dist = float(distances[i, j_idx])

            if (a, b) not in pairs or dist < pairs[(a, b)]:
                pairs[(a, b)] = dist

    sorted_pairs = sorted(
        [(a, b, d) for (a, b), d in pairs.items()],
        key=lambda x: x[2]
    )

    return sorted_pairs[:k]


# ------------------ Dataset Filtering ------------------
def get_indices_by_labels(dataset, target_labels):
    """
    Return indices of samples containing at least one label in target_labels.
    """
    indices = []

    for i in range(len(dataset)):
        _, mask = dataset[i]

        mask_np = mask.numpy() if torch.is_tensor(mask) else mask
        unique_labels = np.unique(mask_np)

        if any(label in unique_labels for label in target_labels):
            indices.append(i)

    return indices


# ------------------ Main Pipeline ------------------
def main():
    # Load DINOv2 model
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(DEVICE)
    model.eval()

    # Image normalization (must match ImageNet stats)
    MEAN = np.array([0.485, 0.456, 0.406])
    STD = np.array([0.229, 0.224, 0.225])

    transform = A.Compose([
        A.CenterCrop(224, 224),
        A.Normalize(mean=tuple(MEAN.tolist()), std=tuple(STD.tolist())),
        ToTensorV2()
    ])

    # ---- Load datasets (you must define these) ----
    dataset = FullCTScanDataset(IMG_DIR, MASK_CSV, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1)

    test_dataset = TestCTScanDataset(
        img_dir=os.path.join(PATH, "test-images"),
        transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=1)

    # ---- Extract embeddings ----
    train_embeddings, _ = extract_embeddings(model, dataloader, DEVICE)
    test_embeddings, _ = extract_embeddings(model, test_loader, DEVICE)

    # Merge embeddings (important fix)
    mixed_embeddings = np.vstack([train_embeddings, test_embeddings])

    # ---- Neighborhood selection ----
    NEIGHBOR_PERCENTILE = 40
    target_label = [6]

    core_idx = get_indices_by_labels(dataset, target_label)

    centroid = mixed_embeddings[core_idx].mean(axis=0, keepdims=True)
    distances = pairwise_distances(mixed_embeddings, centroid).reshape(-1)

    core_distances = distances[core_idx]
    threshold = np.percentile(core_distances, NEIGHBOR_PERCENTILE)

    neighbor_indices = np.where(distances <= threshold)[0].tolist()

    print(f"Found {len(neighbor_indices)} neighbors (threshold={threshold:.4f})")

    # ---- Save result ----
    print(neighbor_indices)
    
    test_head_index = [90, 91, 92, 93, 94, 248, 250, 254, 311, 313, 448, 449, 450, 451]
    print(f"We choose to save the following indices :{test_head_index}")

    os.makedirs(PATH, exist_ok=True)
    np.save(os.path.join(PATH, "test_head_index.npy"), test_head_index)

    print("Saved test_head_index.npy")


# ------------------ Entry Point ------------------
if __name__ == "__main__":
    main()