import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, accuracy_score
from tqdm import tqdm  # Import tqdm for progress bar
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from networks.resnet import ResNet18

def get_loader(train_dir, test_dir) :
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    # Load train and test datasets
    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)

    # Create DataLoader for train and test datasets
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print("DataLoader created for train and test datasets.")
    return train_loader, test_loader

def train_model(model, train_dir, test_dir, num_epochs=10) :

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Initialize lists to store metrics for each epoch
    all_preds = []
    all_labels = []
    total_loss = 0.0

    # Training loop with tqdm progress bar
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

        train_loader, _ = get_loader(train_dir, test_dir)

        with tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}") as t:
            for images, labels in t:
                images = images.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                # Calculate running loss and accuracy
                running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct_preds += torch.sum(preds == labels).item()
                total_preds += labels.size(0)

                # Update the progress bar
                t.set_postfix(loss=running_loss / (t.n + 1), accuracy=correct_preds / total_preds)

        # Calculate epoch metrics
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct_preds / total_preds
        print(f"Epoch {epoch + 1} Loss: {epoch_loss:.4f} Accuracy: {epoch_accuracy:.4f}")

    # After training, evaluate metrics on the entire dataset for FNR and confusion matrix
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate confusion matrix and other metrics
    conf_matrix = confusion_matrix(all_labels, all_preds)
    accuracy = accuracy_score(all_labels, all_preds)
    false_negatives = conf_matrix.sum(axis=1) - np.diag(conf_matrix)
    false_negative_rate = false_negatives / conf_matrix.sum(axis=1)

    # Print metrics
    print(f'Final Accuracy: {accuracy:.4f}')
    print(f'False Negative Rate (FNR) per class: {false_negative_rate}')
    return model

def train_and_save_model(model_name):
    model = ResNet18(num_classes=10)
    model.cuda()

    trained_model = train_model(model, train_dir, test_dir, num_epochs)
    torch.save(trained_model.state_dict(), f'{model_path}/{model_name}.pth')

if __name__ == '__main__' :
    model_name = 'resnet18'
    base_path = '/home/project/Documents/RoTTA-Enhancement/Sem8'
    train_dir = f'{base_path}/Dataset/CIFAR-10/train'
    test_dir = f'{base_path}/Dataset/CIFAR-10/test'
    num_epochs = 20
    model_path = f'{base_path}/Models/saved_models'

    train_and_save_model(model_name)