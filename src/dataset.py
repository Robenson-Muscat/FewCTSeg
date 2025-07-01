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

class CTVisuDataset(Dataset):
  def __init__(self,path,transform=None):
    self.filenames = sorted(glob.glob(os.path.join(path, '**/*.png'),recursive = True),key=alphanumeric_sort)
    self.transform = transform
    dataset_list = []
    for image_file in self.filenames:
        dataset_list.append(Image.open(image_file).convert("RGB"))
    self.imgs = np.stack(dataset_list, axis=0)


  def __getitem__(self, index):
    img_path=self.filenames[index]
    #label = self.filenames[index].split("/")[-2]
    img = Image.open(img_path).convert("RGB")
    if self.transform is not None:
      img=self.transform(img)
    return img, img_path


  def __len__(self):
    return len(self.imgs)


class CTScanDataset(Dataset):
    def __init__(self, img_dir, masks, transform=None):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, '**/*.png'),recursive = True),key=alphanumeric_sort)
        # 
        self.masks = masks  # Masks
        self.transform = transform
        print(f"Nombre d'images : {len(self.img_paths)}")
        print(f"Nombre de masques : {self.masks.shape[0]}")


    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_paths[idx])#, cv2.IMREAD_GRAYSCALE) 
        img = np.expand_dims(img, axis=0) 

        mask = self.masks[idx].astype(np.float32)  # Get the mask
        mask = np.expand_dims(mask, axis=0)  # Add a canal

        if self.transform:
            img = self.transform(img)
            mask = torch.tensor(mask)  # Convert to a Tensor

        return img, mask



# ---------------------------
# Chemins des données
# ---------------------------
PATH = "./data/"
train_path = os.path.join(PATH, "train-images/")

# ---------------------------
# 1. Dataset Definition
# ---------------------------
class GrayCTScanDataset(Dataset):
    def __init__(self, image_dir, mask_csv=None, transform=None):
        self.image_dir = image_dir
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")),key = alphanumeric_sort)
        self.transform = transform
        
        if mask_csv is not None:
            masks = pd.read_csv(mask_csv, index_col=0, header=0).T.values
            self.masks = masks.reshape(-1, 256, 256).astype(np.int64)
        else:
            self.masks = None

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("L")
        img = np.array(img, dtype=np.float32) /255.0
        img_tensor = torch.from_numpy(img).unsqueeze(0)  # (1,H,W) 
    

        mask_tensor = None
        if self.masks is not None:
            mask = self.masks[idx]
            mask_tensor = torch.from_numpy(mask)  # (H,W)
            
            # Maybe it is more optimized for img -- np.uint8
            # torch.Size([1, 256, 256]), dtype:  torch.int64
            # Maybe it has to be mask of shape [256,256], dtype: uint8
            #mask_tensor = mask_tensor.unsqueeze(0) #(1,H,W)


        if self.transform:
            img_tensor = self.transform(img_tensor)

        return img_tensor, mask_tensor

# ---------------------------
# 1. Dataset Definition
# ---------------------------

class RGBCTScanDataset(Dataset):
    def __init__(self, image_dir, mask_csv=None, transform=None):
        self.image_dir = image_dir
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")),key = alphanumeric_sort)
        self.transform = transform
        
        if mask_csv is not None:
            masks = pd.read_csv(mask_csv, index_col=0, header=0).T.values
            self.masks = masks.reshape(-1, 256, 256).astype(np.uint8)
        else:
            self.masks = None

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        
        # Read image and convert to RGB
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # (H,W,3)
        
        #img = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        img_tensor = torch.from_numpy(img).unsqueeze(0)  # (1,H,W,3) 
        img_tensor = img_tensor.permute(0,3,1,2) # (1,3,H,W)

        
        #Read mask
        mask_tensor = None
        if self.masks is not None:
            mask = self.masks[idx]
            mask_tensor = torch.from_numpy(mask)  # (H,W)
            
            #torch.Size([256, 256]), dtype: torch.uint8
            #mask_tensor = mask_tensor.unsqueeze(0) #(1,H,W)
            #torch.Size([1, 256, 256]), dtype: torch.uint8

        if self.transform:
            img_tensor = self.transform(img_tensor)
            

        return img_tensor, mask_tensor


class CTScanDataset(Dataset):
    ###THE RIGHT#####
    def __init__(self, image_dir, mask_csv=None, transform=None):
        self.image_dir = image_dir
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")),key = alphanumeric_sort)[:800]
        self.transform = transform

        if mask_csv is not None:
            masks = pd.read_csv(mask_csv, index_col=0, header=0).T.values[:800]
            self.masks = masks.reshape(-1, 256, 256).astype(np.uint8)
        else:
            self.masks = None

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        # Read image and convert to RGB
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # (H,W,3)

        # IF ONE CHANNEL
        #img = cv2.imread(self.image_paths[idx],cv2.IMREAD_UNCHANGED)
        #img = np.expand_dims(img, axis=2)

        #Read mask
        mask = None
        if self.masks is not None:
            mask = self.masks[idx]  #(H,W)

        # Apply Albumentations transforms
        if self.transform:
            # Pass both image and mask
            augmented = self.transform(image=img, mask=mask)
            img_tensor = augmented['image']
            mask_tensor = augmented['mask']
        else:
          img = img.astype(np.float32) / 255.0  
          img_tensor = torch.from_numpy(img).permute(2,0,1).contiguous()  # (C,H,W)
          if mask is not None:
              mask_tensor = torch.from_numpy(mask).long()
          else:
              mask_tensor = torch.zeros((img.shape[0], img.shape[1]), dtype=torch.long)

        return img_tensor, mask_tensor

# Dataset test (sans masques)
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


class UnlabCTScanDataset(Dataset):
    """Class Dataset to utilise semi-supervised methods on unlabeled data"""
    def __init__(self, image_dir, mask_csv=None, transform=None):
        self.image_dir = image_dir
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")),key = alphanumeric_sort)[800:]
        self.transform = transform

        if mask_csv is not None:
            masks = pd.read_csv(mask_csv, index_col=0, header=0).T.values[800:]
            self.masks = masks.reshape(-1, 256, 256).astype(np.uint8)
        else:
            self.masks = None

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        # Read image and convert to RGB
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # (H,W,3)

        #Read mask
        mask = None
        if self.masks is not None:
            mask = self.masks[idx]  #(H,W)

        # Apply Albumentations transforms
        if self.transform:
            # Pass both image and mask
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        
            

        return img, mask

class LabCTScanDataset(Dataset):
    """Class Dataset on labeled data"""
    def __init__(self, image_dir, mask_csv=None, transform=None):
        self.image_dir = image_dir
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")),key = alphanumeric_sort)[:800]
        
        self.transform = transform

        if mask_csv is not None:
            masks = pd.read_csv(mask_csv, index_col=0, header=0).T.values[:800]
            self.masks = masks.reshape(-1, 256, 256).astype(np.uint8)
        else:
            self.masks = None

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        # Read image and convert to RGB
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # (H,W,3)

        #Read mask
        mask = None
        if self.masks is not None:
            mask = self.masks[idx]  #(H,W)

        # Apply Albumentations transforms
        if self.transform:
            # Pass both image and mask
            augmented = self.transform(image=img, mask=mask)
            img_tensor = augmented['image']
            mask_tensor = augmented['mask']
            

        return img_tensor, mask_tensor
"""
weak_augment= A.Compose([
    A.Transpose(p=1), # Ou verticalFlip
    A.CLAHE(clip_limit=(4,4)),#ou/et AdvancedBlur
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    A.ToTensorV2(),
    ],
)
strong_augment= A.Compose([
        A.Transpose(p=1),
        A.ElasticTransform(alpha=400, sigma=10, alpha_affine=None, p=0.5),
        A.GridDistortion(distort_limit=0.8, num_steps = 5, p=1),#-0.8,0.8
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.ToTensorV2(),]
)

"""

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
        mask = self.masks[idx]
        aug = self.transform(image=img, mask=mask)
        return aug['image'], aug['mask']

class UnlabeledCTScanDataset(Dataset):
    def __init__(self, img_dir, mask_csv, weak_transform):
        paths = sorted(glob.glob(os.path.join(img_dir, '*.png')), key=alphanumeric_sort)
        masks = pd.read_csv(mask_csv, index_col=0).T.values.reshape(-1,256,256).astype(np.uint8)
        valid = [m.sum()==0 for m in masks]
        self.image_paths = [p for p,v in zip(paths,valid) if v]
        self.transform = weak_transform
    def __len__(self): return len(self.image_paths)
    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(self.image_paths[idx]), cv2.COLOR_BGR2RGB)
        aug = self.transform(image=img)
        return aug['image'], self.image_paths[idx]

class PseudoCTScanDataset(Dataset):
    def __init__(self, image_paths, masks):
        self.image_paths, self.masks = image_paths, masks
    def __len__(self): return len(self.masks)
    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(self.image_paths[idx]), cv2.COLOR_BGR2RGB)
        aug = base_transform(image=img, mask=self.masks[idx])
        #aug = strong_transform(image=img, mask=self.masks[idx])
        return aug['image'], aug['mask']

