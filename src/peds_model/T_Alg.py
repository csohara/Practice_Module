
import torch 
DTYPE = torch.float64 
def thomas_algorithm(a, b, c, f):
    N = b.shape[0]
    b, f = b.clone(), f.clone()
    
    # Forward elimination
    for i in range(1, N):
        w = a[i-1] / b[i-1]
        b[i] -= w * c[i-1]
        f[i] -= w * f[i-1]

    # Back substitution
    u = torch.zeros(N, dtype=DTYPE)
    u[-1] = f[-1] / b[-1]
    for i in range(N-2, -1, -1):
        u[i] = (f[i] - c[i] * u[i+1]) / b[i]

    return u

