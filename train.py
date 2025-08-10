import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import time
import os
import argparse

def train_model(data_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_data_path = os.path.join(data_dir, 'train')
    try:
        dataset = datasets.ImageFolder(root=train_data_path, transform=transform)
        print(f"Found {len(dataset.classes)} classes in {train_data_path}")
    except FileNotFoundError as e:
        print(f"Error loading dataset: {e}")
        print(f"Please ensure your training data is in a 'train' subdirectory within '{data_dir}' and that each class has its own subdirectory.")
        return

    if device.type == "cpu":
        num_workers = 0  
        batch_size = 16  
    else:
        num_workers = 4
        batch_size = 32

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(dataset.classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    num_epochs = 10
    for epoch in range(num_epochs):
        start_time = time.time()
        running_loss = 0.0
        correct = 0
        total = 0

        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(dataset)
        epoch_acc = correct / total

        end_time = time.time()
        epoch_time = end_time - start_time

        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.4f} - Time: {epoch_time:.2f} seconds")


    print("Training finished. Saving model...")

    model_save_dir = "/content/drive/MyDrive/extracted_assessment/assessment/models"
    model_save_path = os.path.join(model_save_dir, "resnet50_trained.pth") # You can change the filename

    os.makedirs(model_save_dir, exist_ok=True)

    torch.save(model.state_dict(), model_save_path)

    print(f"Model saved to {model_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a ResNet model.')
    parser.add_argument('--data', type=str, required=True, help='Path to the root of the data directory (e.g., /path/to/your/data_folder)')
    args = parser.parse_args()

    train_model(args.data)