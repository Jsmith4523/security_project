# AI Security Project — CTEC 450

Adversarial machine learning on MNIST: build, attack, and defend a CNN classifier.

## Project Structure

```
├── train.py      # Build and train the baseline CNN
├── attack.py     # FGSM adversarial attack implementation
├── defend.py     # Adversarial training defense
├── evaluate.py   # Final side-by-side evaluation
├── results/      # Generated graphs and saved models (auto-created)
└── data/         # MNIST dataset (auto-downloaded)
```

## Setup

```bash
pip install torch torchvision matplotlib
```

Resolve SSL certificate issue on Mac using:

```
/Applications/Python\ 3.14/Install\ Certificates.command
```

## How to Run (in order)

```bash
# 1. Train the baseline model (~2 min on CPU)
python train.py

# 2. Run the FGSM attack and see accuracy drop
python attack.py

# 3. Apply adversarial training defense
python defend.py

# 4. Final comparison evaluation
python evaluate.py
```

## What Each File Does

### `train.py`
Trains a small CNN on MNIST. Achieves ~99% accuracy on clean data.
Saves the model to `results/model.pth`.

### `attack.py`
Implements **FGSM (Fast Gradient Sign Method)**:
- Computes the gradient of the loss with respect to the input image
- Adds a small perturbation in the direction that maximizes the loss
- Controlled by `epsilon` — higher = stronger attack, lower = more subtle
- At ε=0.2, baseline accuracy drops from ~99% to ~30%

### `defend.py`
Implements **Adversarial Training**:
- Each training batch generates adversarial examples on-the-fly
- Mixes clean + adversarial examples 50/50
- The model learns to classify both correctly
- At ε=0.2, defended accuracy recovers to ~80%+

### `evaluate.py`
Runs a full comparison across epsilon values [0.0, 0.05, ..., 0.30]
and prints a summary table + saves the final plot.

## Results

| Model     | Clean Acc | At ε=0.20 |
|-----------|-----------|-----------|
| Baseline  | ~99%      | ~30%      |
| Defended  | ~97%      | ~80%      |

## Key Concept

FGSM exploits the fact that neural networks are sensitive to small,
targeted perturbations that humans cannot perceive. Adversarial training
is the most effective known defense — by training on adversarial examples,
the model learns more robust decision boundaries.
