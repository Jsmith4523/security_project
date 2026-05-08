"""
train.py - Train a CNN on MNIST
CTEC 450 - AI Security Project
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

# ── Model ──────────────────────────────────────────────────────────────────────

class CNN(nn.Module):
    """Simple convolutional network for MNIST digit classification."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 28x28 -> 28x28
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 28x28 -> 14x14
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 14x14 -> 14x14
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 14x14 -> 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ── Data ───────────────────────────────────────────────────────────────────────

def get_loaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean/std
    ])
    train_set = datasets.MNIST("data", train=True,  download=True, transform=transform)
    test_set  = datasets.MNIST("data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


# ── Training / Evaluation ──────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            correct += (model(images).argmax(1) == labels).sum().item()
    return correct / len(loader.dataset)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = get_loaders()
    model = CNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    epochs = 5
    train_accs, test_accs = [], []

    for epoch in range(1, epochs + 1):
        loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        test_acc = evaluate(model, test_loader, device)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        print(f"Epoch {epoch}/{epochs} | Loss: {loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

    # Save model
    os.makedirs("results", exist_ok=True)
    torch.save(model.state_dict(), "results/model.pth")
    print("\nModel saved to results/model.pth")

    # Plot accuracy
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, epochs + 1), train_accs, label="Train")
    plt.plot(range(1, epochs + 1), test_accs,  label="Test")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Baseline Model Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/baseline_accuracy.png", dpi=150)
    print("Plot saved to results/baseline_accuracy.png")


if __name__ == "__main__":
    main()
