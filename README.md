# Loss Landscape Geometry & Optimization Dynamics

This repository contains a lightweight, modular implementation for probing neural
network loss landscapes. The goal is to empirically study how geometry relates to
optimization dynamics and generalization.

## Motivation
Despite non-convexity, SGD reliably finds solutions that generalize. This project
implements small, interpretable experiments to investigate why — using curvature,
flatness, connectivity, and gradient noise measurements.

## File Structure
- `main.py`           — experiment orchestrator
- `models.py`         — SmallMLP, SmallCNN definitions
- `data.py`           — MNIST dataloaders
- `training.py`       — train/evaluate utilities
- `utils.py`          — Hessian-vector products, flatness, GNS, connectivity
- `plotting.py`       — visualization helpers
- `figures/`          — auto-generated plots
- `data/`             — MNIST dataset (downloaded automatically)

## Requirements
- Python 3.8+
- torch, torchvision
- numpy, matplotlib

Install:
pip install torch torchvision matplotlib numpy


## Usage
Run the main experiment:
`python main.py`

The script will:
- train two MLPs with different random seeds
- estimate sharpness (top Hessian eigenvalue)
- compute perturbation-based flatness
- estimate gradient noise scale
- compute mode connectivity curves
- generate a 2D loss-surface slice
- save all figures into `./figures/`

## Outputs
- flatness stem plot
- mode connectivity curve
- 3D loss-surface slice
- summary metric table

## Reproducibility
Experiments use fixed RNG seeds (0 and 42). Results may vary slightly across hardware.

## Report
A full theoretical write-up and analysis are available in:
`Loss_Landscape_Report.pdf`

## References
- Smith & Le (2018), *A Bayesian Perspective on Gradient Descent Noise*
  https://arxiv.org/abs/1710.06451

- Mandt S, Hoffman MD, Blei DM. **Stochastic Gradient Descent as Approximate Bayesian Inference.**
  *Journal of Machine Learning Research*, 2017.  
  http://jmlr.org/papers/v18/17-214.html

- Chaudhari P, Choromanska A, Soatto S, LeCun Y, Baldassi C, Borgs C, et al.  
  **Entropy-SGD: Biasing Gradient Descent into Wide Valleys.** 2017.  
  https://github.com/ucla-vision/entropy-sgd
  
