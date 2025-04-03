
import torch 
DTYPE = torch.float64 
def thomas_algorithm(a, b, c, f):
    B, N = b.shape
    b, f = b.clone(), f.clone()

    u = torch.zeros_like(f)
    
    # Forward elimination
    for i in range(1, N):
        w = a[:, i - 1] / b[:, i - 1]
        b[:, i] = b[:, i] - w * c[:, i - 1]
        f[:, i] = f[:, i] - w * f[:, i - 1]


    # Back substitution
    u[:, -1] = f[:, -1] / b[:, -1]
    for i in reversed(range(N - 1)):
        u[:, i] = (f[:, i] - c[:, i] * u[:, i + 1]) / b[:, i]

    return u

