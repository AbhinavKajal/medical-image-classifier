"""test_model.py
Evaluate saved model on data/test and write confusion matrix to results/cm.png
"""
import os
import argparse
import torch
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

def load_model(model_path, device):
    # Changed from resnet18 to resnet50 to match training script
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT) # Use weights parameter
    num_ftrs = model.fc.in_features
    # Assuming the trained model has an output layer for 2 classes
    model.fc = torch.nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval() # Set model to evaluation mode
    return model

def evaluate(model, data_dir, device, batch_size=16):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    # Assuming test data is in a 'test' subdirectory within data_dir
    test_data_path = os.path.join(data_dir, 'test')
    try:
        test_ds = datasets.ImageFolder(test_data_path, transform=transform)
        print(f"Found {len(test_ds)} test images in {test_data_path}")
        print(f"Classes: {test_ds.classes}")
    except FileNotFoundError as e:
        print(f"Error loading test dataset: {e}")
        print(f"Please ensure your test data is in a 'test' subdirectory within '{data_dir}' and that each class has its own subdirectory.")
        return None, None, None, None, None, None # Return None if dataset loading fails


    loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    y_true = []
    y_pred = []
    with torch.no_grad(): # Disable gradient calculation during evaluation
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

    if not y_true: # Check if any data was loaded
        print("No test data loaded. Cannot perform evaluation.")
        return None, None, None, None, None, None


    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    # Save confusion matrix
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    cm_path = os.path.join(results_dir, 'cm.png')

    plt.figure(figsize=(5,4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues) # Added cmap for better visualization
    plt.title('Confusion matrix')
    plt.colorbar()
    ticks = np.arange(len(test_ds.classes))
    plt.xticks(ticks, test_ds.classes, rotation=45) # Added rotation for readability
    plt.yticks(ticks, test_ds.classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout() # Adjust layout to prevent labels overlapping
    plt.savefig(cm_path)
    print(f"Confusion matrix saved to {cm_path}")

    return acc, prec, rec, f1, cm, test_ds.classes # Return classes as well

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate a trained model.')
    parser.add_argument('--model', type=str, required=True, help='Path to the saved model file (.pth)')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the root of the data directory (containing test subdirectory)')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        model = load_model(args.model, device)
        print(f"Model loaded successfully from {args.model}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {args.model}")
        exit() # Exit if model file not found
    except Exception as e:
        print(f"Error loading model: {e}")
        exit() # Exit on other model loading errors


    acc, prec, rec, f1, cm, classes = evaluate(model, args.data_dir, device)

    if acc is not None: # Check if evaluation was successful
        print("\nEvaluation Results:")
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        print(f"F1 Score: {f1:.4f}")
        # Print confusion matrix with labels
        print("\nConfusion Matrix:")
        # Create a pandas DataFrame for better display
        import pandas as pd
        if classes:
            cm_df = pd.DataFrame(cm, index=classes, columns=classes)
            print(cm_df)
        else:
             print(cm) # Print raw matrix if classes not available

