import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def plot_loss_surface_3d(alpha_range, beta_range, Z,
                         alpha_model1, beta_model1,
                         alpha_model2, beta_model2,
                         filename="figures/loss_surface_3d.pdf"):
    Alpha, Beta = np.meshgrid(alpha_range, beta_range, indexing="ij")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(Alpha, Beta, Z, linewidth=0, antialiased=True, alpha=0.8)

    ax.scatter(alpha_model1, beta_model1,
               Z[np.argmin(np.abs(alpha_range - alpha_model1)),
                 np.argmin(np.abs(beta_range - beta_model1))],
               color="red", s=50, label="MLP (seed 0)")
    ax.scatter(alpha_model2, beta_model2,
               Z[np.argmin(np.abs(alpha_range - alpha_model2)),
                 np.argmin(np.abs(beta_range - beta_model2))],
               color="blue", s=50, label="MLP (seed 42)")

    ax.set_xlabel("alpha (direction along MLP1→MLP2)")
    ax.set_ylabel("beta (orthogonal direction)")
    ax.set_zlabel("loss")
    ax.set_title("2D Slice of Loss Landscape Around MNIST MLPs")
    ax.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_1d_landscape(filename="sketch_landscape.pdf"):
    theta = np.linspace(-4, 4, 400)
    L1 = 0.2 * (theta + 2) ** 2 + 0.3
    L2 = 1.5 * (theta - 1) ** 2 + 0.3
    L = np.minimum(L1, L2)

    plt.figure()
    plt.plot(theta, L, label="Toy loss landscape")
    plt.scatter([-2, 1], [0.3, 0.3], color="black")
    plt.text(-2, 0.35, "flat min", ha="center")
    plt.text(1, 0.35, "sharp min", ha="center")
    plt.xlabel("parameter direction")
    plt.ylabel("loss")
    plt.title("Toy 1D Loss Landscape")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_flatness_curve(deltas, losses, filename="sketch_flatness.pdf"):
    plt.figure()
    plt.stem(deltas, losses)
    plt.xlabel("perturbation sample index")
    plt.ylabel("loss increase")
    plt.title("Perturbation-based Flatness (random directions)")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_connectivity_curve(alphas, losses, filename="sketch_connectivity.pdf"):
    plt.figure()
    plt.plot(alphas, losses, marker="o")
    plt.xlabel("interpolation alpha")
    plt.ylabel("loss")
    plt.title("Mode Connectivity: Loss vs Interpolation")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
