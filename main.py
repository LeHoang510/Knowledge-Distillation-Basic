import os
import os.path as osp
from pathlib import Path
import random

import torch
import numpy as np
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from visualize import visualize_img
from weather_dataset import WeatherDataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    print("\n=== AI SYSTEM STATUS ===")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print("\n=== GPU DETAILS ===")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"Device Count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Compute Capability: {torch.cuda.get_device_capability(i)}")
            print(f"  Total Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
            
        current_device = torch.cuda.current_device()
        print(f"\nCurrent Device: GPU {current_device}")
        
        # Memory usage
        print("\n=== MEMORY USAGE ===")
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"Cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    
def prepare_dataset(folder_path):
    folders = [f for f in Path(folder_path).iterdir() if f.is_dir()]
    img_paths = []
    labels = []
    for folder in folders:
        for img_path in folder.iterdir():
            img_paths.append(img_path)
            labels.append(folder.name)
    
    nb_class = len(folders)
    
    return img_paths, labels, nb_class

def dataset_split(img_paths, labels):
    val_size = 0.2
    test_size = 0.125
    is_shuffle = True
    X_train, X_val, y_train, y_val = train_test_split(img_paths, labels, 
                                                      test_size=val_size, 
                                                      stratify=labels, 
                                                      shuffle=is_shuffle)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, 
                                                        test_size=test_size, 
                                                        stratify=y_train, 
                                                        shuffle=is_shuffle)

    return X_train, X_val, X_test, y_train, y_val, y_test

def train(path):
    train_batch_size = 256
    test_batch_size = 128

    img_paths, labels, nb_class = prepare_dataset("dataset/weather-dataset/dataset")
    X_train, X_val, X_test, y_train, y_val, y_test = dataset_split(img_paths, labels)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = WeatherDataset(X_train, y_train, transform=transform)
    val_dataset = WeatherDataset(X_val, y_val, transform=transform)
    test_dataset = WeatherDataset(X_test, y_test, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


    

if __name__=='__main__':
    set_seed(42)
    # get_device()
    train("dataset/weather-dataset/dataset")
