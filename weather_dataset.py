import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image

class WeatherDataset(Dataset):
    def __init__(self, img_paths, labels, transform=None):
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

class WeatherTransform():
    def __init__(self, size=(224, 224)):
        self.size = size
    
    def __call__(self, img):
        img = img.resize(self.size)
        img = torch.tensor(img).permute(2, 0, 1).float()
        return img / 255.0