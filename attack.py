"""
attack.py - Fast Gradient Sign Method (FGSM) adversarial attack
CTEC 450 - AI Security Project
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from train import CNN, get_loaders


# ── FGSM ───────────────────────────────────────────────────────────────────────

def fgsm_attack(images, labels, model, criterion, epsilon):
    """
    Fast Gradient Sign Method.

    Perturbs each image by epsilon in the direction that maximizes loss,
    causing the model to misclassify. The perturbation is imperceptible
    to humans at small epsilon values.

    Args:
        images:   batch of input images (requires_grad must be enabled)
        labels:   true labels
        model:    the target model
        criterion: loss function
        epsilon:  perturbation magnitude (0 = no attack, 1 = max attack)

    Returns:
        perturbed images (clamped to valid pixel range)
    """
    images = images.clone().detach().requires_grad_(True)
    outputs = model(images)
    loss = criterion(outputs, labels)
    model.zero_grad()
    loss.backward()

    # Sign of gradient tells us which direction increases the loss
    perturbation = epsilon * images.grad.sign()
    adversarial_images = images + perturbation

    # Clamp to keep pixels in normalized MNIST range (~[-2.8, 2.8])
    adversarial_images = torch.clamp(adversarial_images, -2.8, 2.8)
    return adversarial_images.detach()


# ── Evaluation under attack ────────────────────────────────────────────────────

def evaluate_under_attack(model, loader, criterion, epsilon, device):
    model.eval()
    correct = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        adv_images = fgsm_attack(images, labels, model, criterion, epsilon)
        preds = model(adv_images).argmax(1)
        correct += (preds == labels).sum().item()
    return correct / len(loader.dataset)


# ── Visualization ──────────────────────────────────────────────────────────────

def visualize_attack(model, loader, criterion, epsilon, device, n=8):
    """Show original vs adversarial images side by side."""
    model.eval()
    images, labels = next(iter(loader))
    images, labels = images[:n].to(device), labels[:n].to(device)
    adv_images = fgsm_attack(images, labels, model, criterion, epsilon)

    orig_preds = model(images).argmax(1).cpu()
    adv_preds  = model(adv_images).argmax(1).cpu()

    fig, axes = plt.subplots(2, n, figsize=(14, 4))
    for i in range(n):
        # Original
        axes[0, i].imshow(images[i, 0].cpu(), cmap="gray")
        axes[0, i].set_title(f"True: {labels[i].item()}\nPred: {orig_preds[i].item()}", fontsize=8)
        axes[0, i].axis("off")
        # Adversarial
        axes[1, i].imshow(adv_images[i, 0].cpu(), cmap="gray")
        color = "red" if adv_preds[i].item() != labels[i].item() else "green"
        axes[1, i].set_title(f"Adv pred: {adv_preds[i].item()}", fontsize=8, color=color)
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Original", fontsize=9)
    axes[1, 0].set_ylabel("Adversarial", fontsize=9)
    fig.suptitle(f"FGSM Attack — ε = {epsilon}", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"results/attack_examples_eps{epsilon}.png", dpi=150)
    print(f"Saved attack visualization: results/attack_examples_eps{epsilon}.png")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test_loader = get_loaders()
    criterion = nn.CrossEntropyLoss()

    # Load trained model
    model = CNN().to(device)
    model.load_state_dict(torch.load("results/model.pth", map_location=device))
    model.eval()

    # Test across a range of epsilon values
    epsilons = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    accuracies = []

    print("Evaluating model under FGSM attack...")
    print(f"{'Epsilon':>8} | {'Accuracy':>10}")
    print("-" * 22)
    for eps in epsilons:
        acc = evaluate_under_attack(model, test_loader, criterion, eps, device)
        accuracies.append(acc)
        print(f"{eps:>8.2f} | {acc:>10.4f}")

    # Plot accuracy vs epsilon
    plt.figure(figsize=(8, 4))
    plt.plot(epsilons, accuracies, marker="o", color="red")
    plt.xlabel("Epsilon (attack strength)")
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy vs FGSM Epsilon")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/attack_accuracy.png", dpi=150)
    print("\nPlot saved to results/attack_accuracy.png")

    # Visualize examples at epsilon=0.2
    visualize_attack(model, test_loader, criterion, epsilon=0.2, device=device)


if __name__ == "__main__":
    main()
