import sys
sys.path.append("../")
from utils_functions.sort_files import alphanumeric_sort #Function which sort alphanumerically files
import glob
import os
import numpy as np
from PIL import Image
import torch
import cv2
import albumentations as A
#cv2.setNumThreads(0)  - To avoid slower computation

# ---------------------------
# ---------------------------
PATH = "./data/"
train_path = os.path.join(PATH, "train-images/")


# ---------------------------
# 1. Dataset Definition
# ---------------------------


class CTScanDataset(Dataset):
    def __init__(self, image_dir, mask_csv=None, transform=A.Compose([A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    # Convert image to PyTorch tensor
    A.ToTensorV2(),
])):
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

        
        #Read mask
        mask_tensor = None
        if self.masks is not None:
            mask = self.masks[idx]  #(H,W)
            

        # Apply Albumentations transforms
        if self.transform:
            # Pass both image and mask
            augmented = self.transform(image=img, mask=mask)
            img_tensor = augmented['image']
            mask_tensor = augmented['mask']
            

        return img_tensor, mask_tensor

dataset = CTScanDataset(image_dir=train_path, mask_csv=PATH+"y_train.csv")
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)


