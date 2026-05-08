"""
evaluate.py - Final evaluation summary
CTEC 450 - AI Security Project

Runs a complete side-by-side evaluation of:
  - Baseline model (clean + under attack)
  - Defended model (clean + under attack)
Prints a summary table and saves a combined results figure.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from train import CNN, get_loaders, evaluate
from attack import fgsm_attack, evaluate_under_attack


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test_loader = get_loaders()
    criterion = nn.CrossEntropyLoss()

    # Load both models
    baseline = CNN().to(device)
    baseline.load_state_dict(torch.load("results/model.pth", map_location=device))
    baseline.eval()

    defended = CNN().to(device)
    defended.load_state_dict(torch.load("results/defended_model.pth", map_location=device))
    defended.eval()

    epsilons = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    baseline_accs, defended_accs = [], []

    print("=" * 56)
    print(f"{'Epsilon':>8} | {'Baseline':>10} | {'Defended':>10} | {'Delta':>10}")
    print("=" * 56)
    for eps in epsilons:
        b = evaluate_under_attack(baseline, test_loader, criterion, eps, device)
        d = evaluate_under_attack(defended, test_loader, criterion, eps, device)
        baseline_accs.append(b)
        defended_accs.append(d)
        print(f"{eps:>8.2f} | {b:>10.4f} | {d:>10.4f} | {d - b:>+10.4f}")
    print("=" * 56)

    print(f"\nBaseline clean accuracy : {baseline_accs[0]:.4f}")
    print(f"Defended clean accuracy : {defended_accs[0]:.4f}")
    print(f"\nBaseline at ε=0.20     : {baseline_accs[4]:.4f}")
    print(f"Defended  at ε=0.20    : {defended_accs[4]:.4f}")
    print(f"Improvement             : {defended_accs[4] - baseline_accs[4]:+.4f}")

    # Combined figure
    fig = plt.figure(figsize=(10, 5))
    plt.plot(epsilons, [a * 100 for a in baseline_accs],
             marker="o", color="crimson",  linewidth=2, label="Baseline")
    plt.plot(epsilons, [a * 100 for a in defended_accs],
             marker="s", color="seagreen", linewidth=2, label="Adversarially Trained")
    plt.fill_between(epsilons,
                     [a * 100 for a in baseline_accs],
                     [a * 100 for a in defended_accs],
                     alpha=0.15, color="seagreen", label="Improvement")
    plt.xlabel("FGSM Epsilon", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.title("Security Evaluation: Baseline vs Defended Model", fontsize=13)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/final_evaluation.png", dpi=150)
    print("\nFinal evaluation plot saved to results/final_evaluation.png")


if __name__ == "__main__":
    main()
