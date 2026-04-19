import glob
import os
import sys

import numpy as np
import torch
import cv2
from torch.utils.data import Dataset
import pandas as pd

sys.path.append('../utils/')
cv2.setNumThreads(0)  #- To avoid slower computation



# ------------------ Datasets ------------------
class LabeledCTScanDataset(Dataset):
    def __init__(self, img_dir, mask_csv, transform=None, indices=None):
        all_paths = sorted(glob.glob(os.path.join(img_dir, '*.png')), key=alphanumeric_sort)
        masks_all = pd.read_csv(mask_csv, index_col=0).T.values.reshape(-1,256,256).astype(np.uint8)
        valid = [m.sum()>0 for m in masks_all]
        filtered_global_idxs = [i for i, v in enumerate(valid) if v]  

        # If caller provided `indices`, we expect them to be indices relative to the filtered set
        if indices is not None:
            self.global_indices = [filtered_global_idxs[i] for i in indices]
        else:
            self.global_indices = filtered_global_idxs

        self.image_paths = [all_paths[i] for i in self.global_indices]
        self.masks = masks_all[np.array(self.global_indices)]
        self.transform = transform

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(self.image_paths[idx]), cv2.COLOR_BGR2RGB)
        mask = self.masks[idx]
        if self.transform is not None:
            aug = self.transform(image=img, mask=mask)
            img_t, mask_t = aug['image'], aug['mask']
        else:
            img_t = torch.from_numpy(img).permute(2,0,1).float()/255.0
            mask_t = torch.from_numpy(mask).long()
        global_idx = self.global_indices[idx]
        return img_t, mask_t, global_idx



class UnlabeledPathsDataset(Dataset):
    """Return raw numpy image (H,W,3) for on-the-fly augmentation in UniMatch"""
    def __init__(self, img_paths):
        self.image_paths = img_paths
    def __len__(self): return len(self.image_paths)
    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(self.image_paths[idx]), cv2.COLOR_BGR2RGB)
        name = os.path.basename(self.image_paths[idx])
        global_idx = int(os.path.splitext(name)[0])
        return img, self.image_paths[idx], global_idx

class TestCTScanDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.image_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")), key=alphanumeric_sort)
        self.transform = transform
    def __len__(self): return len(self.image_paths)
    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(self.image_paths[idx]), cv2.COLOR_BGR2RGB)
        name = os.path.basename(self.image_paths[idx])
        global_idx = int(os.path.splitext(name)[0])

        if self.transform:
            img = self.transform(image=img)['image']
        else:
            img = torch.from_numpy(img).permute(2,0,1).float()/255.0
            
        return img, name, global_idx


