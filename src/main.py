import torch

import numpy as np

from peds_model.T_Alg import thomas_algorithm
from peds_model.alpha_layer import AlphaLayer

DTYPE = torch.float64

torch.set_default_dtype(DTYPE)

# Matrix Construction 
def tridiagonal_matrix(alpha):
    """
    Constructs the tridiagonal matrix A(alpha) where:
      a_j = alpha_j^2
      b_j = 1 + alpha_j^3
      c_j = alpha_{j+1}^2 + 2*alpha_{j+1}
    """
    alpha = alpha.to(DTYPE)
    N = alpha.shape[0]
    a = alpha[:-1] ** 2
    b = 1 + alpha ** 3
    c = alpha[1:] ** 2 + 2 * alpha[1:]
    return a, b, c


# Custom Autograd Function 
class AlphaFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, metadata, input):
        """
        Forward solve: u = A(alpha)^(-1) f
        """
        f = metadata["f"].to(DTYPE)
        a, b, c = tridiagonal_matrix(input)
        u = thomas_algorithm(a, b, c, f)
        ctx.save_for_backward(input, a, b, c, u, f)
        return u

    @staticmethod
    def backward(ctx, grad_output):
        """
        Adjoint gradient: grad_input[j] = - w^T (dA/dalpha_j * u)
        where A^T w = grad_output, solved via Thomas algorithm.
        """
        input, a, b, c, u, f = ctx.saved_tensors
        N = input.shape[0]
        w = thomas_algorithm(c, b, a, grad_output.to(DTYPE))
        grad_input = torch.zeros_like(input, dtype=DTYPE)

        for j in range(N):
            dA_dalpha_u_j = torch.zeros(N, dtype=DTYPE)

            if j < N - 1:
                dA_dalpha_u_j[j + 1] += 2 * input[j] * u[j]
            dA_dalpha_u_j[j] += 3 * input[j]**2 * u[j]
            if j > 0:
                dA_dalpha_u_j[j - 1] += (2 * input[j] + 2) * u[j]

            grad_input[j] = - torch.dot(w, dA_dalpha_u_j)

        return None, grad_input

# Test and Compare 
n = 5
rng = np.random.default_rng(42)
alpha = torch.tensor(rng.normal(size=n), dtype=DTYPE, requires_grad=True)
f = torch.tensor(rng.normal(size=n), dtype=DTYPE)

model = AlphaLayer(f)
u = model(alpha)

def solve_explicit(alpha_np, f_np):
    alpha_t = torch.tensor(alpha_np, dtype=DTYPE)
    f_t = torch.tensor(f_np, dtype=DTYPE)
    a, b, c = tridiagonal_matrix(alpha_t)
    return thomas_algorithm(a, b, c, f_t).numpy()

u_manual = solve_explicit(alpha.detach().numpy(), f.detach().numpy())

print("\nForward solution (PyTorch):", u.detach().numpy())
print("Forward solution (Manual)  :", u_manual)
print("Difference in u:", np.linalg.norm(u.detach().numpy() - u_manual))

external_grad = torch.ones_like(u)
alpha.grad = None
u.backward(gradient=external_grad)
grad_autograd = alpha.grad.detach().numpy()

epsilon = 1e-6
grad_fd = np.zeros_like(grad_autograd)
for j in range(n):
    d_alpha = np.zeros_like(grad_autograd)
    d_alpha[j] = epsilon
    alpha_plus = alpha.detach().numpy() + d_alpha
    u_plus = solve_explicit(alpha_plus, f.detach().numpy())
    grad_fd[j] = (np.sum(u_plus) - np.sum(u_manual)) / epsilon

print("\nAutograd gradient:        ", grad_autograd)
print("Finite difference gradient:", grad_fd)
print("Gradient difference norm: ", np.linalg.norm(grad_autograd - grad_fd))