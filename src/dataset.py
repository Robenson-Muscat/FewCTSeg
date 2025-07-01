import sys
sys.path.append("utils_functions/")
from sort_files import alphanumeric_sort
#from utils_functions.sort_files import alphanumeric_sort #Function which sort alphanumerically files
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


######Pipeline pour UNIMATCH à 0.4622######

# ------------------ Datasets ------------------
class LabeledCTScanDataset(Dataset):
    def __init__(self, img_dir, mask_csv, transform):
        paths = sorted(glob.glob(os.path.join(img_dir, '*.png')), key=alphanumeric_sort)
        masks = pd.read_csv(mask_csv, index_col=0).T.values.reshape(-1,256,256).astype(np.uint8)
        valid = [m.sum()>0 for m in masks]
        self.image_paths = [p for p,v in zip(paths,valid) if v]
        self.masks = masks[np.array(valid)]
        self.transform = transform
    def __len__(self): return len(self.image_paths)
    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(self.image_paths[idx]), cv2.COLOR_BGR2RGB)
        #Read mask
        mask = None
        if self.masks is not None:
            mask = self.masks[idx]  #(H,W)
            
        if self.transform is not None:
            aug = self.transform(image=img, mask=mask)
            return aug['image'], aug['mask']
        else:
          img = img.astype(np.float32) / 255.0  
          img_tensor = torch.from_numpy(img).permute(2,0,1).contiguous()  # (C,H,W)
          if mask is not None:
              mask_tensor = torch.from_numpy(mask).long()
          else:
              mask_tensor = torch.zeros((img.shape[0], img.shape[1]), dtype=torch.long)

            return img_tensor, mask_tensor

class UnlabeledCTScanDataset(Dataset):
    def __init__(self, img_dir, mask_csv, weak_transform):
        #Enlever ou pas le weak_transform en paramètres
        paths = sorted(glob.glob(os.path.join(img_dir, '*.png')), key=alphanumeric_sort)
        masks = pd.read_csv(mask_csv, index_col=0).T.values.reshape(-1,256,256).astype(np.uint8)
        valid = [m.sum()==0 for m in masks]
        self.image_paths = [p for p,v in zip(paths,valid) if v]
        self.transform = weak_transform
    def __len__(self): return len(self.image_paths)
    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(self.image_paths[idx]), cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            aug = self.transform(image=img)
            return aug['image'], self.image_paths[idx]
        else:
            return img, self.image_paths[idx]

class PseudoCTScanDataset(Dataset):
    #Enlever ou pas le base_transform en paramètres
    def __init__(self, image_paths, masks#base_transform or aug_transform):
        self.image_paths, self.masks = image_paths, masks
    def __len__(self): return len(self.masks)
    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(self.image_paths[idx]), cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            aug = base_transform(image=img, mask=self.masks[idx])
            return aug['image'], aug['mask']

