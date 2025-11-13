import sys
sys.path.append("../")
from utils.sort_files import alphanumeric_sort #Function which sort alphanumerically files
import glob
import os
import numpy as np
from PIL import Image
import torch
import cv2
import albumentations as A
from torch.utils.data import Dataset
import pandas as pd
#cv2.setNumThreads(0)  - To avoid slower computation


# ------------------ Datasets ------------------
class LabeledCTScanDataset(Dataset):
    def __init__(self, img_dir, mask_csv, transform=None, indices=None):
        paths = sorted(glob.glob(os.path.join(img_dir, '*.png')), key=alphanumeric_sort)
        masks = pd.read_csv(mask_csv, index_col=0).T.values.reshape(-1,256,256).astype(np.uint8)
        valid = [m.sum()>0 for m in masks]
        paths = [p for p,v in zip(paths, valid) if v]
        masks = masks[np.array(valid)]
        if indices is not None:
            self.image_paths = [paths[i] for i in indices]
            self.masks = masks[np.array(indices)]
        else:
            self.image_paths = paths
            self.masks = masks
        self.transform = transform

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError(f'Index {idx} out of range for dataset of size {len(self)}')
        img = cv2.cvtColor(cv2.imread(self.image_paths[idx]), cv2.COLOR_BGR2RGB)
        mask = self.masks[idx]
        if self.transform is not None:
            aug = self.transform(image=img, mask=mask)
            return aug['image'], aug['mask']
        else:
            return torch.from_numpy(img).permute(2,0,1).float()/255.0 , torch.from_numpy(mask).long()



class UnlabeledPathsDataset(Dataset):
    """Return raw numpy image (H,W,3) for on-the-fly augmentation in UniMatch"""
    def __init__(self, img_paths):
        self.image_paths = img_paths
    def __len__(self): return len(self.image_paths)
    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError(f'Index {idx} out of range for dataset of size {len(self)}')
        img = cv2.cvtColor(cv2.imread(self.image_paths[idx]), cv2.COLOR_BGR2RGB)
        return img, self.image_paths[idx]


class TestCTScanDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.image_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")), key=alphanumeric_sort)
        self.transform = transform
    def __len__(self): return len(self.image_paths)
    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError(f'Index {idx} out of range for dataset of size {len(self)}')
        img = cv2.cvtColor(cv2.imread(self.image_paths[idx]), cv2.COLOR_BGR2RGB)
        name = os.path.basename(self.image_paths[idx])
        if self.transform:
            img = self.transform(image=img)['image']
        return img, name
