import torch
import numpy as np
from peds_model.alpha_layer import AlphaLayer, AlphaFunction
from peds_model.T_Alg import thomas_algorithm

DTYPE = torch.float64
torch.set_default_dtype(DTYPE)

def test_alpha_layer_backward_pass():
    # Set up inputs
    n = 5
    rng = np.random.default_rng(0)
    alpha = torch.tensor(rng.normal(size=n), dtype=DTYPE, requires_grad=True)
    f = torch.tensor(rng.normal(size=n), dtype=DTYPE)

    # Forward pass
    model = AlphaLayer(f)
    u = model(alpha)

    # Backward pass
    u.sum().backward()
    grad_autograd = alpha.grad.detach().numpy()

    # Finite difference
    epsilon = 1e-6
    alpha_np = alpha.detach().numpy()
    f_np = f.detach().numpy()
    grad_fd = np.zeros_like(alpha_np)  # Creates an array of of zeros same shape as alpha_np

    # Converts to tensor, solves then converts to numpy 
    def solve(alpha_arr, f_arr):
        alpha_t = torch.tensor(alpha_arr, dtype=DTYPE)
        f_t = torch.tensor(f_arr, dtype=DTYPE)
        a, b, c = AlphaFunction.tridiagonal_matrix(alpha_t)
        return thomas_algorithm(a, b, c, f_t).numpy()

    # Solution used for comparison
    u_base = solve(alpha_np, f_np)

    for j in range(n):
        alpha_plus = alpha_np.copy()
        alpha_plus[j] += epsilon
        u_plus = solve(alpha_plus, f_np)python 
        grad_fd[j] = (np.sum(u_plus) - np.sum(u_base)) / epsilon

    # Compare gradients
    assert np.allclose(grad_autograd, grad_fd, atol=1e-5), (
        f"Backward gradient mismatch.\nAutograd: {grad_autograd}\nFD: {grad_fd}"
    )
