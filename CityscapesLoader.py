import os
import glob
import torch
import cv2
from torch.utils.data import Dataset
import imageio.v2 as imageio
import numpy as np
from mapping_utils import mapping_8

class CityscapesDataset(Dataset):
    def __init__(self, image_dir, label_dir):
        self.image_dir=image_dir
        self.label_dir=label_dir
        self.imagepaths=sorted(glob.glob(self.image_dir))
        labelpaths=sorted(glob.glob(self.label_dir))
        self.label_paths=[]
        for img in labelpaths:
            if 'labelIds' in os.path.basename(img):
                self.label_paths.append(img)
        
    def __len__(self):
        return len(self.imagepaths)

    def __getitem__(self, idx):
                    
        image = imageio.imread(self.imagepaths[idx])
        mask = imageio.imread(self.label_paths[idx])    
                
        image = cv2.resize(image, (512,256))
        mask = cv2.resize(mask, (512,256))

        for i, j in np.ndindex(mask.shape):
            mask[i][j] = mapping_8[mask[i][j]]
        
        img = torch.tensor(image, dtype=torch.float32)
        img = torch.tensor(img.tolist())
        img = img.permute(2, 0, 1)
        mask = torch.tensor(mask, dtype=torch.uint8)
        mask = torch.tensor(mask.tolist())
        
        return image, mask