import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

class WeatherTransform():
    def __init__(self, size=(224, 224)):
        self.size = size
    
    def __call__(self, img):
        img = img.resize(self.size)
        img = np.array(img)
        img = torch.from_numpy(img)
        img = img.permute(2, 0, 1).float()/255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = (img - mean) / std
        return img
    
class WeatherDataset(Dataset):
    def __init__(self, img_paths, labels, transform=WeatherTransform()):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = Image.open(img_path)
        label = self.labels[index]

        if self.transform:
            img = self.transform(img)
        
        return img, label

