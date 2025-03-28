import os
import os.path as osp
from pathlib import Path
import random

import torch
from sklearn.model_selection import train_test_split

def set_seed(seed):
    random.seed(seed)
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
    print(folders)
    img_paths = []
    labels = []
    for folder in folders:
        for img_path in folder.iterdir():
            img_paths.append(img_path)
            labels.append(folder.name)
    
    return img_paths, labels

def dataset_split(img_paths, labels):
    val_size = 0.2
    test_size = 0.125
    is_shuffle = True
    X_train, X_val, y_train, y_val = train_test_split(img_paths, 
                                                      labels, 
                                                      test_size=val_size, 
                                                      stratify=labels, 
                                                      shuffle=is_shuffle)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, 
                                                        test_size=test_size, 
                                                        stratify=y_train, 
                                                        shuffle=is_shuffle)

    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__=='__main__':
    # set_seed(42)
    # get_device()
    prepare_dataset("dataset/weather-dataset/dataset")
