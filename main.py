"""
Loss Landscape Geometry & Optimization Dynamics
Implementation + Figure Generation

Requirements:
- torch
- torchvision
- matplotlib
- numpy

"""

import os
import copy
import numpy as np
import torch
import torch.nn as nn

# Replace the monolithic script with a modular orchestrator

from models import SmallMLP
from data import get_mnist_loaders
from training import train_model, evaluate
from utils import (flatten_params, compute_flatness, estimate_top_hessian_eig,
                   compute_gradient_noise_scale, mode_connectivity_curve,
                   compute_2d_loss_surface)
from plotting import (plot_loss_surface_3d, plot_1d_landscape,
                      plot_flatness_curve, plot_connectivity_curve)

def main():
    os.makedirs("figures", exist_ok=True)

    # 1. Data
    train_loader, val_loader, test_loader = get_mnist_loaders()

    # 2. Models
    mlp1 = SmallMLP()
    mlp2 = SmallMLP()

    loss_fn = nn.CrossEntropyLoss()

    # 3. Train two models with different seeds
    torch.manual_seed(0)
    mlp1, hist1 = train_model(mlp1, train_loader, val_loader,
                              epochs=5, lr=0.05)

    torch.manual_seed(42)
    mlp2, hist2 = train_model(mlp2, train_loader, val_loader,
                              epochs=5, lr=0.05)

    # 4. Evaluate final performance (train + test)
    train_loss1, train_acc1 = evaluate(mlp1, train_loader, loss_fn)
    test_loss1, test_acc1 = evaluate(mlp1, test_loader, loss_fn)

    train_loss2, train_acc2 = evaluate(mlp2, train_loader, loss_fn)
    test_loss2, test_acc2 = evaluate(mlp2, test_loader, loss_fn)

    print(f"\nMLP1 train_acc={train_acc1:.3f} test_acc={test_acc1:.3f}")
    print(f"MLP2 train_acc={train_acc2:.3f} test_acc={test_acc2:.3f}")

    # 5. Top Hessian eigenvalue (rough estimate)
    print("\nEstimating top Hessian eigenvalue (MLP1)...")
    lambda_max1 = estimate_top_hessian_eig(mlp1, val_loader, loss_fn,
                                           num_iters=20)
    print(f"Estimated top eigenvalue (MLP1): {lambda_max1:.4f}")

    print("\nEstimating top Hessian eigenvalue (MLP2)...")
    lambda_max2 = estimate_top_hessian_eig(mlp2, val_loader, loss_fn,
                                           num_iters=20)
    print(f"Estimated top eigenvalue (MLP2): {lambda_max2:.4f}")

    # 6. Flatness metric
    print("\nComputing flatness metric (MLP1)...")
    avg_increase1, delta_idx1, loss_increases1 = compute_flatness(
        mlp1, val_loader, loss_fn, epsilon=1e-3, num_samples=10
    )
    print(f"Average loss increase (MLP1): {avg_increase1:.6f}")

    print("\nComputing flatness metric (MLP2)...")
    avg_increase2, delta_idx2, loss_increases2 = compute_flatness(
        mlp2, val_loader, loss_fn, epsilon=1e-3, num_samples=10
    )
    print(f"Average loss increase (MLP2): {avg_increase2:.6f}")

    # 7. Mode connectivity between mlp1 and mlp2
    print("\nComputing mode connectivity curve (MLP1 <-> MLP2)...")
    alphas, losses = mode_connectivity_curve(mlp1, mlp2,
                                             val_loader, loss_fn,
                                             num_points=21)
    print("Mode connectivity losses (first few):", [float(l) for l in losses[:5]])

    # 8. Gradient noise scale (efficient proxy)
    print("\nEstimating gradient noise scale (MLP1)...")
    gns1 = compute_gradient_noise_scale(mlp1, train_loader, loss_fn,
                                        num_batches=20)
    if gns1 is not None:
        S1, sq_norm_mean_g1, mean_sq_norm1 = gns1
        print(f"Gradient noise scale (MLP1): {S1:.4f}")
    else:
        S1, sq_norm_mean_g1, mean_sq_norm1 = float("nan"), float("nan"), float("nan")
        print("Could not estimate gradient noise scale for MLP1 (no batches).")

    print("\nEstimating gradient noise scale (MLP2)...")
    gns2 = compute_gradient_noise_scale(mlp2, train_loader, loss_fn,
                                        num_batches=20)
    if gns2 is not None:
        S2, sq_norm_mean_g2, mean_sq_norm2 = gns2
        print(f"Gradient noise scale (MLP2): {S2:.4f}")
    else:
        S2, sq_norm_mean_g2, mean_sq_norm2 = float("nan"), float("nan"), float("nan")
        print("Could not estimate gradient noise scale for MLP2 (no batches).")

    # 9. 2D loss surface around the two models
    print("\nComputing 2D loss surface around MLP1/MLP2...")

    params1 = [p for p in mlp1.parameters() if p.requires_grad]
    params2 = [p for p in mlp2.parameters() if p.requires_grad]
    theta1_flat = flatten_params(params1).detach()
    theta2_flat = flatten_params(params2).detach()

    diff = theta2_flat - theta1_flat
    dist = diff.norm().item()
    d1 = diff / (dist + 1e-8)

    v = torch.randn_like(theta1_flat)
    v = v - (v @ d1) * d1
    d2 = v / (v.norm() + 1e-8)

    center_flat = 0.5 * (theta1_flat + theta2_flat)

    grid_size = 21
    alpha_range = np.linspace(-dist, dist, grid_size)
    beta_range = np.linspace(-0.5 * dist, 0.5 * dist, grid_size)

    Z = compute_2d_loss_surface(mlp1, val_loader, loss_fn,
                                center_flat, d1, d2,
                                alpha_range, beta_range)

    alpha_model1, beta_model1 = -dist / 2.0, 0.0
    alpha_model2, beta_model2 =  dist / 2.0, 0.0

    plot_loss_surface_3d(alpha_range, beta_range, Z,
                         alpha_model1, beta_model1,
                         alpha_model2, beta_model2,
                         filename=os.path.join("figures",
                                               "loss_surface_3d.pdf"))
    print("3D loss surface saved to figures/loss_surface_3d.pdf")

    # 10. Generate figures
    print("\nGenerating plots...")
    # plot_1d_landscape(filename=os.path.join("figures",
    #                                         "sketch_landscape.pdf"))
    plot_flatness_curve(delta_idx1, loss_increases1,
                        filename=os.path.join("figures",
                                              "sketch_flatness.pdf"))
    plot_connectivity_curve(alphas, losses,
                            filename=os.path.join("figures",
                                                  "sketch_connectivity.pdf"))
    print("Figures saved in ./figures/")

    # 11. Print table summary
    print("%% Model & lambda_max & Flatness & GradNoiseScale & Train acc & Test acc \\\\")
    print(f"MLP (seed 0) & {lambda_max1:.4f} & {avg_increase1:.6f} "
          f"& {S1:.4f} & {train_acc1:.3f} & {test_acc1:.3f} \\\\")
    print(f"MLP (seed 42) & {lambda_max2:.4f} & {avg_increase2:.6f} "
          f"& {S2:.4f} & {train_acc2:.3f} & {test_acc2:.3f} \\\\")


if __name__ == "__main__":
    main()
