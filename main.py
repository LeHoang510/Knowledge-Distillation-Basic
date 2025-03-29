import os
import os.path as osp
from pathlib import Path
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import timm
from tqdm import tqdm

from visualize import visualize_img
from weather_dataset import WeatherDataset
from utils import Logger

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
    
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
def prepare_dataset(folder_path):
    folders = [f for f in Path(folder_path).iterdir() if f.is_dir()]
    img_paths = []
    labels = []
    labels_dict = {}

    for id, folder in enumerate(folders):
        labels_dict[id] = folder.name

    for folder in folders:
        for img_path in folder.iterdir():
            img_paths.append(img_path)
            folder_id = [id for id, name in labels_dict.items() if name == folder.name][0]
            labels.append(folder_id)

    nb_class = len(folders)
    
    return img_paths, labels, labels_dict, nb_class

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

def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total = 0
    correct = 0
    val_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(data_loader, desc="Evaluating")):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            val_loss += loss.item()

    accuracy = correct/total
    val_loss = val_loss/len(data_loader)
    model.train()
    return val_loss, accuracy

def fit_model(
        model, 
        optimizer, 
        criterion, 
        train_loader, 
        val_loader, 
        logger,
        patience,
        device, 
        epochs
    ):
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    model.train()
    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        batch_loss = 0.0
        for i, (images, labels) in enumerate(tqdm(train_loader, desc="Training")):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_loss += loss.item()
        
        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device)
        train_losses.append(batch_loss / len(train_loader))
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        logger.write(epoch, epochs, train_losses[-1], val_loss, val_accuracy)
    
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    return train_losses, val_losses, val_accuracies

def train(path, device):
    train_batch_size = 64
    test_batch_size = 32

    img_paths, labels, labels_dict, nb_class = prepare_dataset(path)
    X_train, X_val, X_test, y_train, y_val, y_test = dataset_split(img_paths, labels)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = WeatherDataset(X_train, y_train, transform=transform)
    val_dataset = WeatherDataset(X_val, y_val, transform=transform)
    test_dataset = WeatherDataset(X_test, y_test, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=test_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    student_model = timm.create_model(
        "resnet18",
        pretrained=True,
        num_classes=nb_class
    ).to(device)
    teacher_model = timm.create_model(
        "densenet169",
        pretrained=True,
        num_classes=nb_class
    ).to(device)

    lr = 1e-2
    epochs = 15
    patience = 3
    criterion = nn.CrossEntropyLoss()

    student_logger = Logger(log_dir=Path("output/logs/student"))
    teacher_logger = Logger(log_dir=Path("output/logs/teacher"))
    
    Path("output/model").mkdir(parents=True, exist_ok=True)

    teacher_path = Path("output/model/teacher_model.pth")
    student_path = Path("output/model/student_model.pth")

    print("\n=== TRAIN STUDENT ===")
    student_optimizer = optim.Adam(student_model.parameters(), lr=lr)
    train_losses, val_losses, val_accuracies = fit_model(
        student_model, 
        student_optimizer, 
        criterion, 
        train_loader, 
        val_loader, 
        student_logger,
        patience,
        device, 
        epochs
    )
    print("\n=== TRAINING COMPLETE ===")

    print(f"Best Validation Loss: {min(val_losses)}")
    print(f"Best Validation Accuracy: {max(val_accuracies)}")
    print(f"Final Training Loss: {train_losses[-1]}")
    print(f"Final Validation Loss: {val_losses[-1]}")
    print(f"Final Validation Accuracy: {val_accuracies[-1]}")

    print("\n=== TEST STUDENT ===")
    test_loss, test_accuracy = evaluate_model(student_model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_accuracy}")

    student_logger.close()
    print("\n=== TESTING COMPLETE ===")

    torch.save(student_model.state_dict(), student_path)
    print("\n=== STUDENT MODEL SAVED ===")

    print("\n=== TRAIN TEACHER ===")
    teacher_optimizer = optim.Adam(teacher_model.parameters(), lr=lr)
    train_losses, val_losses, val_accuracies = fit_model(
        teacher_model, 
        teacher_optimizer, 
        criterion, 
        train_loader, 
        val_loader, 
        teacher_logger,
        patience,
        device, 
        epochs
    )
    print("\n=== TRAINING COMPLETE ===")
    print(f"Best Validation Loss: {min(val_losses)}")
    print(f"Best Validation Accuracy: {max(val_accuracies)}")
    print(f"Final Training Loss: {train_losses[-1]}")
    print(f"Final Validation Loss: {val_losses[-1]}")
    print(f"Final Validation Accuracy: {val_accuracies[-1]}")

    print("\n=== TEST TEACHER ===")
    test_loss, test_accuracy = evaluate_model(teacher_model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_accuracy}")

    teacher_logger.close()
    print("\n=== TESTING COMPLETE ===")

    torch.save(teacher_model.state_dict(), teacher_path)
    print("\n=== TEACHER MODEL SAVED ===")

if __name__=='__main__':
    set_seed(42)
    device = get_device()
    train("dataset/weather-dataset/dataset", device)
