import os
import glob
import numpy as np
from PIL import Image
from PIL import ImageFile
import skimage.io as skio

import torch
from torchvision import transforms
from torch.utils.data import Dataset

def denorm(x, device):
    std = torch.Tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1).to(device)
    mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1).to(device)
    out = torch.clamp(x * std + mean, 0, 1)
    return out

class LoadDataset(Dataset):
    def __init__(self,content_dir,style_dir,transform=None,shuffle=False):
        content_imgs = sorted(glob.glob(os.path.join(content_dir,"*")))
        style_imgs = sorted(glob.glob(os.path.join(style_dir,"*")))
        if shuffle:
            np.random.shuffle(content_imgs)
            np.random.shuffle(style_imgs)
        self.content_imgs = content_imgs
        self.style_imgs = style_imgs
        self.transform = transform
    
    def __len__(self):
        return len(self.style_imgs)
    
    def __getitem__(self, index):
        Image.MAX_IMAGE_PIXELS = None
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        content_img = self.content_imgs[index]
        style_img = self.style_imgs[index]
        content_img = Image.open(content_img)
        style_img = Image.open(style_img)

        if self.transform:
            content_img = self.transform(content_img)
            style_img = self.transform(style_img)
        return content_img, style_img