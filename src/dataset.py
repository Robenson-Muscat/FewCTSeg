import sys
sys.path.append("../")
from utils_functions.sort_files import alphanumeric_sort
import glob
import os
import numpy as np
from PIL import Image
import torch

class CTDataset(Dataset):
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
        self.masks = masks  # Masques déjà chargés
        self.transform = transform
        print(f"Nombre d'images : {len(self.img_paths)}")
        print(f"Nombre de masques : {self.masks.shape[0]}")


    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_paths[idx], cv2.IMREAD_GRAYSCALE)  # Charger image en niveaux de gris
        img = np.expand_dims(img, axis=0)  # Ajouter un canal

        mask = self.masks[idx].astype(np.float32)  # Récupérer le masque
        mask = np.expand_dims(mask, axis=0)  # Ajouter un canal

        if self.transform:
            img = self.transform(img)
            mask = torch.tensor(mask)  # Convertir en Tensor directement

        return img, mask

