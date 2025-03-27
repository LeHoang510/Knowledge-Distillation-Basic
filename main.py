import os
import os.path as osp
from pathlib import Path
import random

import torch

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

if __name__=='__main__':
    set_seed(42)
    get_device()
