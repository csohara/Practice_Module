import torch

import numpy as np

from peds_model.T_Alg import thomas_algorithm
from peds_model.alpha_layer import AlphaLayer, AlphaFunction

DTYPE = torch.float64

torch.set_default_dtype(DTYPE)


# Test and Compare 
n = 5
b = 3 # batches 
rng = np.random.default_rng(42)
alpha = torch.tensor(rng.normal(size=(b,n)), dtype=DTYPE, requires_grad=True)
f = torch.tensor(rng.normal(size=n), dtype=DTYPE)

model = AlphaLayer(f)

# forward pass
u = model(alpha)

# Manual solve for comparison
def solve_explicit_batch(alpha_np, f_np):
    results = []
    for alpha_i in alpha_np:
        a, b, c = AlphaFunction.tridiagonal_matrix(torch.tensor(alpha_i, dtype=DTYPE))
        u_i = thomas_algorithm(a, b, c, torch.tensor(f_np, dtype=DTYPE))
        results.append(u_i.detach().numpy())
    return np.stack(results)

u_manual = solve_explicit_batch(alpha.detach().numpy(), f.detach().numpy())

print("\nForward solution (PyTorch):\n", u.detach().numpy())
print("\nForward solution (Manual):\n", u_manual)
print("\nDifference in u:", np.linalg.norm(u.detach().numpy() - u_manual))


#def solve_explicit(alpha_np, f_np):
    #alpha_t = torch.tensor(alpha_np, dtype=DTYPE)
    #f_t = torch.tensor(f_np, dtype=DTYPE)
    #a, b, c = AlphaFunction.tridiagonal_matrix(alpha_t)
    #return thomas_algorithm(a, b, c, f_t).numpy()

#u_manual = solve_explicit(alpha.detach().numpy(), f.detach().numpy())

#print("\nForward solution (PyTorch):", u.detach().numpy())
#print("Forward solution (Manual)  :", u_manual)
#print("Difference in u:", np.linalg.norm(u.detach().numpy() - u_manual))

#external_grad = torch.ones_like(u)
#alpha.grad = None
#u.backward(gradient=external_grad)
#grad_autograd = alpha.grad.detach().numpy()

#epsilon = 1e-6
#grad_fd = np.zeros_like(grad_autograd)
#for j in range(n):
    #d_alpha = np.zeros_like(grad_autograd)
    #d_alpha[j] = epsilon
    #alpha_plus = alpha.detach().numpy() + d_alpha
    #u_plus = solve_explicit(alpha_plus, f.detach().numpy())
    #grad_fd[j] = (np.sum(u_plus) - np.sum(u_manual)) / epsilon

#print("\nAutograd gradient:        ", grad_autograd)
#print("Finite difference gradient:", grad_fd)
#print("Gradient difference norm: ", np.linalg.norm(grad_autograd - grad_fd))