import sys
sys.path.append("utils_functions/")
#from sort_files import alphanumeric_sort
from utils_functions.sort_files import alphanumeric_sort #Function which sort alphanumerically files
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


######Pipeline pour UNIMATCH à 0.4622 et 0.4713######

# ------------------ Datasets ------------------
class LabeledCTScanDataset(Dataset):
    def __init__(self, img_dir, mask_csv, transform=None):
        """Basic augmentations"""
        paths = sorted(glob.glob(os.path.join(img_dir, '*.png')), key=alphanumeric_sort)
        masks = pd.read_csv(mask_csv, index_col=0).T.values.reshape(-1,256,256).astype(np.uint8)
        valid = [m.sum()>0 for m in masks]
        self.imgs = [p for p,v in zip(paths,valid) if v]
        self.masks = masks[np.array(valid)]
        self.transform = transform
    def __len__(self): return len(self.imgs)
    def __getitem__(self, idx):

        #if idx >= len(self):
            #raise IndexError(f'Index {idx} out of range for dataset of size {len(self)}')
        
        img = cv2.cvtColor(cv2.imread(self.imgs[idx]), cv2.COLOR_BGR2RGB)
        #Read mask
        mask = None
        if self.masks is not None:
            mask = self.masks[idx]  #(H,W)
            
        #Assert that is correct shape
        
        #assert img.shape == (256, 256, 3), f"Image shape mismatch: {img.shape}"
        #assert mask.shape == (256, 256), f"Mask shape mismatch: {mask.shape}"
            
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
    def __init__(self, img_dir, mask_csv, transform = None):
        """Unlabeled images(every pixel is labeled as 0) ; Weak augmentations to apply"""
        paths = sorted(glob.glob(os.path.join(img_dir, '*.png')), key=alphanumeric_sort)
        masks = pd.read_csv(mask_csv, index_col=0).T.values.reshape(-1,256,256).astype(np.uint8)
        valid = [m.sum()==0 for m in masks]
        self.imgs = [p for p,v in zip(paths,valid) if v]
        self.transform = transform
    def __len__(self): return len(self.imgs)
    def __getitem__(self, idx):
        
        #if idx >= len(self):
            #raise IndexError(f'Index {idx} out of range for dataset of size {len(self)}')
        img = cv2.cvtColor(cv2.imread(self.imgs[idx]), cv2.COLOR_BGR2RGB)
        img_path = None

        #Assert that is correct shape
        
        #assert img.shape == (256, 256, 3), f"Image shape mismatch: {img.shape}"
        
        if self.imgs is not None:
            img_path = self.imgs[idx]
        if self.transform is not None:
            aug = self.transform(image=img)
            return aug['image'], img_path
        else:
          img = img.astype(np.float32) / 255.0  
          img = torch.from_numpy(img).permute(2,0,1).contiguous()  # (C,H,W)
          return img_tensor, img_path

class PseudoCTScanDataset(Dataset):
    def __init__(self, imgs, masks, transform):
        self.imgs, self.masks, self.transform = imgs, masks,transform
    def __len__(self): return len(self.masks)
    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(self.imgs[idx]), cv2.COLOR_BGR2RGB)

        #Assert that is correct shape
        #assert img.shape == (256, 256, 3), f"Image shape mismatch: {img.shape}"
        #assert mask.shape == (256, 256), f"Mask shape mismatch: {mask.shape}"
        
        
        aug = self.transform(image=img, mask=self.masks[idx])
        return aug['image'], aug['mask']

# Test Dataset
class CTTestDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")), key=alphanumeric_sort)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        name = os.path.basename(self.image_paths[idx])
        if self.transform:
            img = self.transform(image=img)['image']
        return img, name

# Test Dataset
class TestCTScanDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")), key=alphanumeric_sort)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        name = os.path.basename(self.image_paths[idx])
        if self.transform:
            img = self.transform(image=img)['image']
        return img, name
