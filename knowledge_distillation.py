from pathlib import Path

import timm
import torch
from torchvision import transforms
from tqdm import tqdm

from weather_dataset import WeatherDataset
from train import set_seed, get_device, prepare_dataset, dataset_split, evaluate_model
from utils import Logger

def fit(
        teacher_model,
        student_model,
        train_loader,
        val_loader,
        weight,
        temperature,
        criterion,
        optimizer,
        logger,
        patience,
        device,
        epochs
):
    teacher_model.eval()
    student_model.train()

    train_losses = []
    val_losses = []
    val_accuracies = []

    patience_counter = 0
    best_val_loss = float('inf')

    for epoch in range(epochs):
        batch_losses = []
        for i, (images, labels) in enumerate(tqdm(train_loader)):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.no_grad():
                teacher_outputs = teacher_model(images)
            student_outputs = student_model(images)

            ce_loss = criterion(student_outputs, labels)

            soft_targets = torch.softmax(teacher_outputs/temperature, dim=-1)
            soft_prob = torch.log_softmax(student_outputs/temperature, dim=-1)
            soft_targets_loss  = torch.sum(soft_targets * (soft_targets.log()-soft_prob))/soft_prob.size()[0]

            loss = (1-weight)*ce_loss + weight*(temperature**2)*soft_targets_loss 
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())
        
        val_loss, val_accuracy = evaluate_model(student_model, val_loader, criterion, device)
        train_losses.append(sum(batch_losses) / len(batch_losses))
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        logger.write(epoch, epochs, train_losses[-1], val_loss, val_accuracy)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping")
            break
    
    return train_losses, val_losses, val_accuracies


def teach(path, device):
    img_paths, labels, labels_dict, nb_class =  prepare_dataset(path)
    X_train, X_val, X_test, y_train, y_val, y_test = dataset_split(img_paths, labels)

    teacher_model = timm.create_model(
        "resnet18",
        pretrained=True,
        num_classes=nb_class
    ).to(device)
    teacher_model.load_state_dict(torch.load(Path("output/student_model.pth"), map_location=device))

    student_model = timm.create_model(
        "resnet50",
        pretrained=True,
        num_classes=nb_class
    ).to(device)

    train_batch_size = 64
    test_batch_size = 32
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = WeatherDataset(X_train, y_train, transform=transform)
    val_dataset = WeatherDataset(X_val, y_val, transform=transform)
    test_dataset = WeatherDataset(X_test, y_test, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=test_batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    lr = 1e-2
    epochs = 15
    patience = 3
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(student_model.parameters(), lr=lr)

    student_logger = Logger(log_dir=Path("output/logs/kd_student"))
    Path("output/model").mkdir(parents=True, exist_ok=True)
    student_path = Path("output/model/kd_student.pth")

    print("\n=== TRAIN STUDENT ===")
    train_losses, val_losses, val_accuracies = fit(
        teacher_model,
        student_model,
        train_loader,
        val_loader,
        weight=0.75,
        temperature=2.0,
        criterion=criterion,
        optimizer=optimizer,
        logger=student_logger,
        patience=patience,
        device=device,
        epochs=epochs
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

if __name__=="__main__":
    device = get_device()
    set_seed(42)
    teach("dataset/weather-dataset/dataset", device)