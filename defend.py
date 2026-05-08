"""
defend.py - Adversarial Training Defense
CTEC 450 - AI Security Project

Defense strategy: Adversarial Training
  During each training batch, we generate FGSM adversarial examples
  on-the-fly and mix them with clean examples. This teaches the model
  to correctly classify both clean and perturbed inputs.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os

from train import CNN, get_loaders, evaluate
from attack import fgsm_attack, evaluate_under_attack


# ── Adversarial Training ────────────────────────────────────────────────────────

def adversarial_train_epoch(model, loader, optimizer, criterion, device, epsilon=0.2):
    """
    Train one epoch using a 50/50 mix of clean and adversarial examples.
    Mixing both types prevents the model from losing accuracy on clean data.
    """
    model.train()
    total_loss, correct = 0.0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        # Generate adversarial examples for this batch
        adv_images = fgsm_attack(images, labels, model, criterion, epsilon)

        # Combine clean + adversarial (batch size doubles)
        combined_images = torch.cat([images, adv_images], dim=0)
        combined_labels = torch.cat([labels, labels], dim=0)

        # Standard training step on the combined batch
        model.train()
        optimizer.zero_grad()
        outputs = model(combined_images)
        loss = criterion(outputs, combined_labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * combined_images.size(0)
        correct += (outputs.argmax(1) == combined_labels).sum().item()

    n = 2 * len(loader.dataset)
    return total_loss / n, correct / n


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = get_loaders()
    criterion = nn.CrossEntropyLoss()

    # Load the baseline model weights as starting point
    baseline_model = CNN().to(device)
    baseline_model.load_state_dict(torch.load("results/model.pth", map_location=device))

    # Fine-tune with adversarial training (start from baseline weights)
    defended_model = CNN().to(device)
    defended_model.load_state_dict(torch.load("results/model.pth", map_location=device))
    optimizer = optim.Adam(defended_model.parameters(), lr=5e-4)  # lower LR for fine-tuning

    epsilon_train = 0.2
    epochs = 5

    print(f"\nAdversarial training (epsilon={epsilon_train})...")
    train_accs, clean_accs, adv_accs = [], [], []

    for epoch in range(1, epochs + 1):
        _, train_acc = adversarial_train_epoch(
            defended_model, train_loader, optimizer, criterion, device, epsilon=epsilon_train
        )
        clean_acc = evaluate(defended_model, test_loader, device)
        adv_acc   = evaluate_under_attack(defended_model, test_loader, criterion, epsilon_train, device)

        train_accs.append(train_acc)
        clean_accs.append(clean_acc)
        adv_accs.append(adv_acc)

        print(f"Epoch {epoch}/{epochs} | Train Acc: {train_acc:.4f} | "
              f"Clean: {clean_acc:.4f} | Adv (ε={epsilon_train}): {adv_acc:.4f}")

    # Save defended model
    torch.save(defended_model.state_dict(), "results/defended_model.pth")
    print("\nDefended model saved to results/defended_model.pth")

    # Compare baseline vs defended across epsilon values
    epsilons = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    baseline_accs  = []
    defended_accs  = []

    print("\nComparing baseline vs defended model across epsilons...")
    print(f"{'Epsilon':>8} | {'Baseline':>10} | {'Defended':>10} | {'Improvement':>12}")
    print("-" * 48)
    for eps in epsilons:
        b_acc = evaluate_under_attack(baseline_model,  test_loader, criterion, eps, device)
        d_acc = evaluate_under_attack(defended_model,  test_loader, criterion, eps, device)
        baseline_accs.append(b_acc)
        defended_accs.append(d_acc)
        print(f"{eps:>8.2f} | {b_acc:>10.4f} | {d_acc:>10.4f} | {d_acc - b_acc:>+12.4f}")

    # Plot comparison
    plt.figure(figsize=(9, 5))
    plt.plot(epsilons, baseline_accs, marker="o", color="red",   label="Baseline (no defense)")
    plt.plot(epsilons, defended_accs, marker="s", color="green", label="Adversarially Trained")
    plt.xlabel("Epsilon (attack strength)")
    plt.ylabel("Accuracy")
    plt.title("Baseline vs Adversarially Trained Model Under FGSM Attack")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/defense_comparison.png", dpi=150)
    print("\nComparison plot saved to results/defense_comparison.png")

    # Training curve for defended model
    plt.figure(figsize=(9, 4))
    plt.plot(range(1, epochs + 1), clean_accs, marker="o", label="Clean accuracy")
    plt.plot(range(1, epochs + 1), adv_accs,   marker="s", label=f"Adversarial accuracy (ε={epsilon_train})")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Adversarial Training Progress")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/defense_training.png", dpi=150)
    print("Training curve saved to results/defense_training.png")


if __name__ == "__main__":
    main()
