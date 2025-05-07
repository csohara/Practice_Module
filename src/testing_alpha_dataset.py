from peds_model.Alpha_Dataset import AlphaFieldDataset, alpha_step_function
from peds_model.T_Alg import thomas_algorithm
import torch
import matplotlib.pyplot as plt
from peds_model.QoI import QuantityOfInterest 

DTYPE = torch.float64
torch.set_default_dtype(DTYPE)

# Step 1: Load dataset
dataset = AlphaFieldDataset(n_samples=3, L=1000, n_fibres=3, sigma=0.02, alpha_1 = 0.5)
x = dataset.x


# Define sample locations for QoI (e.g. 25%, 50%, 75%)
qoi = QuantityOfInterest(sample_points=[0.25, 0.5, 0.75], x_grid=x)

# Step 2: Solve for each alpha using unbatched Thomas algorithm
solutions = []

for i in range(len(dataset)):
    alpha, fibre_regions = dataset[i]  # alpha is shape [1000]

    # Build tridiagonal matrix components
    a = alpha[:-1] ** 2
    b = 2 + alpha ** 3           # <--- bumped up for stability
    c = alpha[1:] ** 2 + 2 * alpha[1:]
    f = torch.ones_like(alpha)

    u = thomas_algorithm(a, b, c, f)
    solutions.append(u)
    # Compute the quantity of interest for this u
    qoi_values = qoi(u)

    print(f"\nSample {i}:")
    print(f"  alpha max: {alpha.max().item():.2f}")
    print(f"  u min: {u.min().item():.2f}, u max: {u.max().item():.2f}")
    print(f"  QoI: {qoi_values.numpy()}")



    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(x, alpha.numpy(), label="alpha(x)", alpha=0.6)
    plt.plot(x, u.detach().numpy(), label="u(x)", linestyle="--")
    for start, end in fibre_regions:
        plt.axvspan(start, end, color="red", alpha=0.2)
    plt.title(f"Sample {i}")
    plt.xlabel("x")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()






