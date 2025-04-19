import sys
sys.path.append("../")
from utils_functions.sort_files import alphanumeric_sort
import glob
import os
import numpy as np
from PIL import Image
import torch

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
        img = np.array(img, dtype=np.float32)  
        img_tensor = torch.from_numpy(img).unsqueeze(0)  # (1,H,W) 
    

        mask_tensor = None
        if self.masks is not None:
            mask = self.masks[idx]
            mask_tensor = torch.from_numpy(mask)  # (H,W)
            
            # torch.Size([1, 256, 256]), dtype:  torch.int64
            # Maybe it has to be mask of shape [256,256], dtype: uint8
            #mask_tensor = mask_tensor.unsqueeze(0) #(1,H,W)


        if self.transform:
            img_tensor = self.transform(img_tensor)

        return img_tensor, mask_tensor

