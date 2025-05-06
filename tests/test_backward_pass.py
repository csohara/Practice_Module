import torch
import numpy as np
from peds_model.alpha_layer import AlphaLayer, AlphaFunction
from peds_model.T_Alg import thomas_algorithm

DTYPE = torch.float64
torch.set_default_dtype(DTYPE)

def test_alpha_layer_batched_backward_pass():
    # Set up batched inputs
    B, N = 3, 5
    rng = np.random.default_rng(0)
    alpha = torch.tensor(rng.normal(size=(B, N)), dtype=DTYPE, requires_grad=True)
    f = torch.tensor(rng.normal(size=(B, N)), dtype=DTYPE)

    # Forward pass
    model = AlphaLayer(f)
    u = model(alpha)

    # Backward pass
    u.sum().backward()
    grad_autograd = alpha.grad.detach().numpy()  # shape (B, N)

    # Finite difference
    epsilon = 1e-6
    alpha_np = alpha.detach().numpy()
    f_np = f.detach().numpy()
    grad_fd = np.zeros_like(alpha_np)  # shape (B, N)


    # Solve for manual gradient calculation 
    def solve(alpha_arr, f_arr):
        alpha_t = torch.tensor(alpha_arr, dtype=DTYPE)
        f_t = torch.tensor(f_arr, dtype=DTYPE)

        if alpha_t.ndim == 1:
            alpha_t = alpha_t.unsqueeze(0)
        if f_t.ndim == 1:
            f_t = f_t.unsqueeze(0)

        a, b, c = AlphaFunction.tridiagonal_matrix(alpha_t)
        return thomas_algorithm(a, b, c, f_t).squeeze(0).numpy() # # Remove fake batch dim (if added)

    # Loop over batch and each alpha index
    for b in range(B):
        u_base = solve(alpha_np[b], f_np[b]) # Solve system for original alpha
        for j in range(N):
            alpha_plus = alpha_np[b].copy()
            alpha_plus[j] += epsilon
            u_plus = solve(alpha_plus, f_np[b])
            grad_fd[b, j] = (np.sum(u_plus) - np.sum(u_base)) / epsilon

    # Compare gradients
    assert np.allclose(grad_autograd, grad_fd, atol=1e-5), (
        f"Backward gradient mismatch.\nAutograd:\n{grad_autograd}\nFD:\n{grad_fd}"
    )
